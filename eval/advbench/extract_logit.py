import argparse
import glob
import json
import os
import random
from collections import defaultdict

import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from eval.utils import (
generate_completions_extract_logits,
    ensure_dir
)
from eval.utils import dynamic_import_function
import pandas as pd
from eval.advbench.utils import gpt_eval, asr_eval
from modeling.dexperts import DExpertsLlamaExtract
from transformers import AutoTokenizer


def get_prompt(instruction):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    return prompt


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    harmful_data = list(pd.read_csv("data/advbench/train.csv")["goal"])
    if args.start >= 0:
        harmful_data = harmful_data[args.start:]
    if args.samples >= 0:
        harmful_data = harmful_data[:args.samples]

    prompts = []
    for data in harmful_data:
        prompt = get_prompt(data)
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            prompt = chat_formatting_function(messages, add_bos=False)
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])
    print(prompts[0], flush=True)


    # if not tokenizer_name_or_path:
    tokenizer_name_or_path = args.expert_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                                use_fast_tokenizer=not args.use_slow_tokenizer)

    # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = {
        'device_map': "auto",
        'offload_folder': 'offload_folder',
        'torch_dtype': "auto",
        'offload_state_dict': True,
        'load_in_8bit': args.load_in_8bit,
        # 'cache_dir': "../../../checkpoint/"
    }
    model = DExpertsLlamaExtract(
        expert_model_name_or_path=args.expert_model_name_or_path,
        antiexpert_model_name_or_path=args.antiexpert_model_name_or_path,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        model_kwargs=model_kwargs,
    )

    outputs, expert_logits, antiexpert_logits, expert_hiddens, antiexpert_hiddens = generate_completions_extract_logits(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        do_sample=False,
        batch_size=1,
    )

    expert = torch.cat(expert_logits, dim=0)
    antiexpert = torch.cat(antiexpert_logits, dim=0)
    antiexpert_hiddens = torch.cat(antiexpert_hiddens, dim=0)
    assert expert.shape[0] == antiexpert.shape[0]
    top = 500
    expert_logit, expert_topk = torch.topk(expert, k=top, dim=-1)
    antiexpert_logit, antiexpert_topk = torch.topk(antiexpert, k=top, dim=-1)
    # expert_softmax = expert_softmax[torch.arange(expert.shape[0])[:, None], expert_topk]
    expert_dict = {"logit": expert_logit, "indices": expert_topk}
    antiexpert_dict = {"logit": antiexpert_logit, "indices": antiexpert_topk}
    torch.save(expert_dict, os.path.join(args.logit_dir, f"expert_{args.start}_{args.start + args.samples}.bin"))
    torch.save(antiexpert_dict,
               os.path.join(args.logit_dir, f"antiexpert_{args.start}_{args.start + args.samples}.bin"))
    torch.save(antiexpert_hiddens,
               os.path.join(args.logit_dir, f"antiexpert_hidden_{args.start}_{args.start + args.samples}.bin"))
    expert_hiddens = torch.cat(expert_hiddens, dim=0)
    torch.save(expert_hiddens,
                os.path.join(args.logit_dir, f"expert_hidden_{args.start}_{args.start + args.samples}.bin"))
    torch.save(model.antiexpert.lm_head.weight.detach().cpu(), os.path.join(args.logit_dir, "emb.pt"))

    with open(os.path.join(args.save_dir, f"predictions_{args.start}_{args.start + args.samples}.jsonl"), "w") as fout:
            for i in range(len(harmful_data)):
                example = {"instruction": harmful_data[i], "response": outputs[i]}
                fout.write(json.dumps(example) + "\n")
                

    if args.eval:
        if args.metric=="gpt4":
            metric_model = "gpt-4-1106-preview"
        elif args.metric=="gpt3.5":
            metric_model = "gpt-3.5-turbo-0125"
        else:
            raise ValueError
        kw_asr = asr_eval(outputs=outputs)
        gpt_score, gpt_asr = gpt_eval(instructions=harmful_data, outputs=outputs, cache_path=os.path.join(args.save_dir, f"{args.metric}_eval_cache.jsonl"), model=metric_model)
        performance = {"samples": len(harmful_data), "kw_asr": kw_asr, "gpt_asr": gpt_asr, "gpt_score": gpt_score}
        print(performance)
        with open(os.path.join(args.save_dir, f"{args.metric}_metrics_{args.start}_{args.start + args.samples}.jsonl"), "a") as fout:
            fout.write(json.dumps(performance, indent=4)+"\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/toxigen"
    )
    parser.add_argument(
        "--logit_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the model to generate the predictions.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here.",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--antiexpert_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--chat_formatting_function",
        type=str,
        default="eval.templates.create_prompt_with_tulu_chat_format", 
        help="The function to use to create the chat format. This function will be dynamically imported. Please see examples in `eval/templates.py`."
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=100,
        help="If given, we will only use this many instances per group. Default to 500 (half the available instances).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default='gpt3.5',
    )
    parser.add_argument(
        "--start",
        type=int,
        default=-1,

    )
    parser.add_argument(
        "--samples",
        type=int,
        default=-1,

    )
    args = parser.parse_args()
    if args.logit_dir is None:
        args.logit_dir = args.save_dir

    
    main(args)
