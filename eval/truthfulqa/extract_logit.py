import argparse
import os
import json
import torch
import pandas as pd
import openai

from eval.utils import (
generate_completions_extract_logits,
    dynamic_import_function,
    ensure_dir
)
from transformers import AutoTokenizer
from modeling.dexperts import DExpertsLlamaExtract


def trim_answer(answer):
    """
    Trim generated answer for open-ended evaluation setting.
    """
    # remove spaces at the beginning and end
    answer = answer.strip()
    # remove the "Answer:" prefix if it exists
    if answer.startswith('Answer:'):
        answer = answer[len('Answer:'):].strip()
    # reformat line-breaks for long-form answers
    answer = answer.replace('\n\n', '\n')
    return answer


def run_hf_model(
    test_df, model, tokenizer, batch_size=1, max_new_tokens=50
):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""
    prompts = []
    # test_df = test_df.head(100)
    for _, row in test_df.iterrows():
        prompt = row['Question']
        if args.use_chat_format:
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            messages = []
            if args.system_prompt:
                messages += [{"role": "system", "content": args.system_prompt}]
            messages += [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\n\nAnswer:"
        prompts.append(prompt)

    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    print(prompts[0], flush=True)

    # outputs = generate_completions(
    #     model,
    #     tokenizer,
    #     prompts,
    #     batch_size=batch_size,
    #     max_new_tokens=max_new_tokens,
    #     do_sample=False
    # )
    outputs, expert_logits, antiexpert_logits, expert_hiddens, antiexpert_hiddens = generate_completions_extract_logits(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        batch_size=1,
        do_sample=args.do_sample,
        temperature=args.temperature
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
    torch.save(expert_hiddens,os.path.join(args.logit_dir, f"expert_hidden_{args.start}_{args.start + args.samples}.bin"))
    torch.save(model.antiexpert.lm_head.weight.detach().cpu(), os.path.join(args.logit_dir, "emb.pt"))


    assert len(outputs) == len(prompts)
    test_df['output'] = [trim_answer(o) for o in outputs]

    test_df.to_json(os.path.join(args.save_dir, f"open_results_{args.start}_{args.start + args.samples}.jsonl"), lines=True, orient='records')

    return test_df



def format_results(results):
    results = results[[x for x in results.columns if (results[x].dtype != object)]]
    return results


def main(args):
    ensure_dir(args.save_dir)
    test_df = pd.read_csv(os.path.join("data/truthfulqa", "train.csv"))

    if args.start >= 0:
        test_df = test_df[args.start:]
    if args.samples >= 0:
        test_df = test_df[:args.samples]

    # load individual HF models
    from transformers import AutoTokenizer

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

    if args.do_sample:
        model_kwargs = {
            'device_map': "auto",
            'offload_folder': 'offload_folder',
            'torch_dtype': torch.bfloat16,
            'offload_state_dict': True,
            'load_in_8bit': args.load_in_8bit,
            # 'cache_dir': "../../../checkpoint/"
        }
    else:
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
        chat_response_prefix="Answer:"
    )
    print("Running generations!")
    run_hf_model(
        test_df, model, tokenizer, batch_size=1, max_new_tokens=500
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The HuggingFace model to be evaluated."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="If specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/truthfulqa/",
        help="The directory to save the results."
    )
    parser.add_argument(
        "--logit_dir",
        type=str,
        default=None
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="The number of instances to evaluate. If not given, we will evaluate all instances."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8bit mode, which will reduce memory and speed up inference."
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
        "--start",
        type=int,
        default=-1,

    )
    parser.add_argument(
        "--samples",
        type=int,
        default=-1,

    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--do_sample",
        action='store_true',
        default=False
    )

    args = parser.parse_args()
    if args.logit_dir is None:
        args.logit_dir = args.save_dir
    main(args)
