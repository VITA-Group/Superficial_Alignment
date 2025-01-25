import os
from typing import Optional, Dict, Any
import torch
import json
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers.generation.utils import (
    ModelOutput,
    # top_k_top_p_filtering,
    StoppingCriteriaList,
    LogitsProcessorList
)
import torch.optim as optim

from tqdm import trange
import torch.nn.init as init
DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant."
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"



from typing import List


def merge_prefix(prefix):
    new_prefix = []
    start=None
    end=None
    for p in prefix:
        p=p.split("_")
        if start is None:
            start=p[0]
            end=p[1]
        else:
            if end==p[0]:
                end=p[1]
            else:
                new_prefix.append(f"{start}_{end}")
                start=p[0]
                end=p[1]
    new_prefix.append(f"{start}_{end}")
    return new_prefix


def get_topk(x, topk):
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x).to(x.device)
    result.scatter_(1, top_indices, top_values)
    return result

def get_topk_softmax(x, topk):
    x = torch.softmax(x, dim=-1)
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x).to(x.device)
    result.scatter_(1, top_indices, top_values)
    return result

def get_topk_one(x, topk):
    top_values, top_indices = torch.topk(x, topk, dim=1)
    result = torch.zeros_like(x).to(x.device)
    result.scatter_(1, top_indices, 1)
    return result



class DExpertsLlama:
    def __init__(
        self,
        base_model_name_or_path: str,
        data_dir: str,
        tokenizer: AutoTokenizer,
        alpha: float = 1.0,
        topk:int=40,
        model:str="Linear",
        model_kwargs: Dict[str, Any] = None,
        save_logit_model_path=None,
    ):
        

        # NOTE: Our implementation calls each model in sequence, but these calls can
        # potentially be parallelized for faster generation (e.g., putting models on
        # different GPUs and collecting logits with GPU communication like NCCL operations)
        self.base = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path, **model_kwargs
        )
        self.base.eval()




        self.tokenizer = tokenizer
        self.topk = topk
        self.alpha = alpha
        self.device = self.base.device
        self.model_name = model
        # model:  [hidden/toplogit]-[hidden/logit]-[residual/direct]
    
        assert os.path.exists(save_logit_model_path)
        self.build_logit_model(data_dir, rank=-1)
        print(save_logit_model_path)
        state_dict = torch.load(save_logit_model_path, map_location='cpu')
        self.logit_model.load_state_dict(state_dict, strict=True)
        self.logit_model = self.logit_model.to(self.device)

    def build_logit_model(self, data_dir, rank=-1):
        W = torch.load(os.path.join(data_dir, "emb.pt")).float()
        print("W.shape=", W.shape)
        hidden_size = W.shape[-1]
        out_hidden = W.shape[0]
        print(f"out_hidden={out_hidden}")
        # hidden_size = 5120
        ms = self.model_name.split('-')
        assert len(ms)==3
        if ms[1] == "hidden":
            out_hidden = hidden_size
        elif ms[1] == "logit":
            out_hidden = out_hidden
        else:
            raise ValueError
        if "hidden" in ms[0]:
            in_hidden = hidden_size
        elif "logit" in ms[0]:
            in_hidden = out_hidden
        else:
            raise ValueError

        if rank < 0:
            self.logit_model = nn.Linear(in_hidden, out_hidden, bias=False)
        else:
            self.logit_model = nn.Sequential(nn.Linear(in_hidden, rank, bias=False), nn.Linear(rank, out_hidden, bias=False))





    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 100,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        **kwargs
    ):
        base_kwargs = kwargs.copy()
    
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        eos_token_id_tensor = torch.tensor([self.tokenizer.eos_token_id]).to(input_ids.device)


        for step in range(max_new_tokens):
            # prepare model inputs with past_key_values and attention_mask
            base_inputs = self.base.prepare_inputs_for_generation(input_ids, **base_kwargs)

            
            base_outputs = self.base(**base_inputs, return_dict=True, output_hidden_states=True)
            
            base_next_token_logits = base_outputs.logits[..., -1, :]
            base_next_token_hidden = base_outputs.hidden_states[-1][..., -1, :]



            ms = self.model_name.split('-')

            assert len(ms) == 3
            if ms[0] == "hidden":
                x=base_next_token_hidden
            elif ms[0] == "logit":
                x = base_next_token_logits
            elif ms[0] == "toplogit":
                x = get_topk(base_next_token_logits, self.topk)

            else:
                print(ms[0])
                raise ValueError

            logits = self.logit_model(x.float())

            if ms[1] == "hidden":
                logits = torch.mm(logits, self.W)

            if ms[2] == "residual":
                logits += base_next_token_logits
            next_tokens = logits.argmax(1)


            next_tokens = (
                next_tokens * unfinished_sequences +
                self.tokenizer.pad_token_id * (1 - unfinished_sequences)
            )

            # update model inputs for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # update kwargs
            base_kwargs = self._update_model_kwargs_for_generation(base_outputs, base_kwargs)
            
            # stopping criteria
            if stopping_criteria and stopping_criteria(input_ids, None):
                break

            # if eos_token was found in one sentence, set sentence to finished
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break
        return input_ids

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        # update past_key_values
        kwargs["past_key_values"] = self._extract_past_from_model_output(outputs)

        if getattr(outputs, "state", None) is not None:
            kwargs["state"] = outputs.state

        # update attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
            kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

        return kwargs

    def _extract_past_from_model_output(self, outputs: ModelOutput):
        past_key_values = None
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values
        elif "mems" in outputs:
            past_key_values = outputs.mems
        elif "past_buckets_states" in outputs:
            past_key_values = outputs.past_buckets_states

        return past_key_values




