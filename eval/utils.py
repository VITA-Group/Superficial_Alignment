import torch
import tqdm
import os
from importlib import import_module
from transformers import (
    StoppingCriteria,
    StoppingCriteriaList,
    LogitsProcessorList,
    NoBadWordsLogitsProcessor,
    SuppressTokensAtBeginLogitsProcessor
)
from typing import List


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.inference_mode()
def generate_completions_extract_logits(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    expert_prompts=None,
    **generation_kwargs
):
    expert_logits = []
    antiexpert_logits = []
    expert_hiddens = []
    antiexpert_hiddens = []
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if expert_prompts is not None:
            batch_expert_prompts = expert_prompts[i:i+batch_size]
            tokenized_expert_prompts = tokenizer(
                batch_expert_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
            )
            batch_expert_input_ids = tokenized_expert_prompts.input_ids
            expert_attention_mask = tokenized_expert_prompts.attention_mask
        else:
            batch_expert_input_ids=None
            expert_attention_mask=None

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None

        batch_outputs, expert_logit, antiexpert_logit, expert_hidden, antiexpert_hidden = model.generate(
            input_ids=batch_input_ids,
            attention_mask=attention_mask,
            stopping_criteria=stopping_criteria,
            logits_processor=logits_processor,
            temperature=temperature,
            top_p=top_p,
            save_logits = True,
            extra_expert_input_ids = batch_expert_input_ids,
            expert_attention_mask = expert_attention_mask,
            **generation_kwargs
        )
        antiexpert_logits.append(antiexpert_logit)
        expert_logits.append(expert_logit)
        antiexpert_hiddens.append(antiexpert_hidden)
        expert_hiddens.append(expert_hidden)


        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations, expert_logits, antiexpert_logits, expert_hiddens, antiexpert_hiddens

@torch.inference_mode()
def generate_completions(
    model,
    tokenizer,
    prompts,
    batch_size=1,
    stop_id_sequences=None,
    banned_id_sequences=None,
    banned_begin_ids=None,
    add_special_tokens=True,
    disable_tqdm=False,
    temperature=1.0,
    top_p=1.0,
    train=False,
    **generation_kwargs
):
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens
        )
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        stopping_criteria = StoppingCriteriaList([KeyWordsCriteria(stop_id_sequences)]) if stop_id_sequences else None

        # create logit processors
        if banned_id_sequences or banned_begin_ids:
            logit_processors = []
            if banned_id_sequences:
                logit_processors.append(
                    NoBadWordsLogitsProcessor(banned_id_sequences, eos_token_id=tokenizer.eos_token_id)
                )
            if banned_begin_ids:
                logit_processors.append(
                    SuppressTokensAtBeginLogitsProcessor(banned_begin_ids, begin_index=batch_input_ids.shape[1])
                )
            logits_processor = LogitsProcessorList(logit_processors)
        else:
            logits_processor = None
        if train:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                temperature=temperature,
                top_p=top_p,
                train=True,
                **generation_kwargs
            )
        else:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                temperature=temperature,
                top_p=top_p,
                **generation_kwargs
            )

        # the stopping criteria is applied at batch level, so if other examples are not stopped,
        # the entire batch will continue to generate. so some outputs still have the stop sequence,
        # which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated 
        # the same way as in the outputs. we changed our previous way of truncating the output token ids
        # directly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)

        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]

        generations += batch_generations

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations



def load_hf_lm_and_tokenizer(
    model_name_or_path,
    tokenizer_name_or_path=None,
    device_map="auto",
    load_in_8bit=False,
    convert_to_half=False,
    use_fast_tokenizer=True,
    padding_side="left",
):

    from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM

    model_kwargs = {
        'device_map': "auto",
        'offload_folder': 'offload_folder',
        'torch_dtype': "auto",
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
        # 'cache_dir': "../../../checkpoint/"
    }
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    if convert_to_half:
        model = model.half()
    model.eval()

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # for OPT and Pythia models, we need to set tokenizer.model_max_length to model.config.max_position_embeddings
    # to avoid wrong embedding index.
    if isinstance(model, GPTNeoXForCausalLM) or isinstance(model, OPTForCausalLM):
        tokenizer.model_max_length = model.config.max_position_embeddings
        print("Set tokenizer.model_max_length to model.config.max_position_embeddings: {}".format(model.config.max_position_embeddings))

    return model, tokenizer


def load_dexperts_and_tokenizer(
    base_model_name_or_path: str,
    data_dir: str,
    device_map: str = "auto",
    alpha: float = 1.0,
    topk: int = 40,
    load_in_8bit: bool = False,
    use_fast_tokenizer: bool = True,
    tokenizer_name_or_path: str = None,
    padding_side: str = "left",
    model:str = "hidden-logit-residual",
    save_logit_model_path=None,
    ):
    from modeling.dexperts_linear import  DExpertsLlama
    from transformers import AutoTokenizer

    if not tokenizer_name_or_path:
        tokenizer_name_or_path = base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, use_fast_tokenizer=use_fast_tokenizer)

    # set padding side to left for batch generation
    tokenizer.padding_side = padding_side

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        'device_map': device_map,
        'offload_folder': 'offload_folder',
        'torch_dtype': "auto",
        'offload_state_dict': True,
        'load_in_8bit': load_in_8bit,
        # 'cache_dir': "../../../checkpoint/"
    }

    model = DExpertsLlama(
        base_model_name_or_path=base_model_name_or_path,
        data_dir = data_dir,
        model=model,
        tokenizer=tokenizer,
        alpha=alpha,
        model_kwargs=model_kwargs,
        topk=topk,
        save_logit_model_path = save_logit_model_path,
    )

    return model, tokenizer



def dynamic_import_function(function_path):
    '''
    Dynamically import a function from a path string (e.g., "module.submodule.my_function")
    '''
    module_path, function_name = function_path.rsplit(".", 1)
    module = import_module(module_path)
    function = getattr(module, function_name)
    return function
