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
    generate_completions,
    load_hf_lm_and_tokenizer,
    load_dexperts_and_tokenizer
)
from eval.utils import dynamic_import_function
import pandas as pd
from eval.advbench.utils import gpt_eval, asr_eval


def get_prompt(instruction):
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    return prompt


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    harmful_data = list(pd.read_csv("data/advbench/test.csv")["goal"])
    if args.max_examples > 0:
        harmful_data = harmful_data[:args.max_examples]

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

    if args.model_name_or_path:  # single HuggingFace model
        print("Loading model and tokenizer for generations...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path if args.model_name_or_path else args.model_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )
    else:
        model, tokenizer = load_dexperts_and_tokenizer(
            args.base_model_name_or_path,
            data_dir=args.data_dir,
            model=args.model,
            alpha=args.alpha,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
            topk=args.topk,
            save_logit_model_path=args.logit_path
        )


    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        do_sample=False,
    )


    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
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
        performance = {"logit_path":args.logit_path, "samples": len(harmful_data), "kw_asr": kw_asr, "gpt_asr": gpt_asr, "gpt_score": gpt_score}
        print(performance)
        with open(os.path.join(args.save_dir, f"{args.metric}_metrics.jsonl"), "a") as fout:
            fout.write(json.dumps(performance, indent=4)+"\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--save_dir",
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
        "--eval_batch_size", type=int, default=1, help="batch size for evaluation."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference.",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
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
        "--model",
        type=str,
        default="LR"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=200
    )
    parser.add_argument(
        "--logit_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    print(args.save_dir)

    # assert args.model_name_or_path or args.base_model_name_or_path

    main(args)
    print(args.save_dir)
