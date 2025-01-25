import argparse
import os
import json
import torch
import pandas as pd
import openai

from eval.truthfulqa.metrics import run_end2end_GPT3, run_end2end_llama
from eval.utils import (
    load_hf_lm_and_tokenizer,
    generate_completions,
    dynamic_import_function,
    ensure_dir,
    load_dexperts_and_tokenizer,
)


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
    # answer = answer.replace('\n\n', '\n')
    stop_sequence = ['Comment:', '\n\n']
    for prefix in stop_sequence:
        if prefix in answer:
            answer = answer.split(prefix)[0]
    return answer


def run_hf_model(
    test_df, model, tokenizer, batch_size=1, max_new_tokens=50
):
    """Stores answers from autoregressive HF models (GPT-2, GPT-Neo)"""
    prompts = []
    test_df = test_df.head(100)
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

    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        do_sample=False
    )

    assert len(outputs) == len(prompts)
    test_df['output'] = [trim_answer(o) for o in outputs]


    print("Running LLama-based evaluation metrics!")
    try:
        test_df = run_end2end_llama('GPT-true', test_df, info=False)
        test_df = run_end2end_llama('GPT-info', test_df, info=True)
    except Exception as err:
        print(err)

    test_df["GPT-true-info acc"] = test_df["GPT-true acc"] * test_df["GPT-info acc"]
    test_df.to_json(os.path.join(args.save_dir, "open_results_llama.jsonl"), lines=True, orient='records')

    # format and print basic results
    results = format_results(test_df)
    results = results.mean(axis=0).to_dict()
    results["logit_path"] = args.logit_path
    print(results)

    with open(os.path.join(args.save_dir, 'open_metrics_llama.json'), 'a') as f:
        json.dump(results, f, indent=2)
        f.write('\n')
    print(os.path.join(args.save_dir, 'open_metrics_llama.json'))
    return test_df



def format_results(results):
    results = results[[x for x in results.columns if (results[x].dtype != object)]]
    return results


def main(args):
    ensure_dir(args.save_dir)
    test_df = pd.read_csv(os.path.join("data/truthfulqa", "test.csv"))


    if args.max_examples is not None:
        # test_df = test_df.sample(args.max_examples, random_state=42)
        test_df = test_df[:args.max_examples]

    # load individual HF models
    if args.model_name_or_path:
        print("Loading HF model and tokenizer...")
        model, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            load_in_8bit=args.load_in_8bit,
            device_map="balanced_low_0" if torch.cuda.device_count() > 1 else "auto",
            use_fast_tokenizer=not args.use_slow_tokenizer,
        )

    if 'open' in args.settings:
        if args.base_model_name_or_path:
            print("Loading DExperts model and tokenizer...")

            model, tokenizer = load_dexperts_and_tokenizer(
                args.base_model_name_or_path,
                data_dir=args.data_dir,
                model=args.model,
                alpha=args.alpha,
                load_in_8bit=args.load_in_8bit,
                use_fast_tokenizer=not args.use_slow_tokenizer,
                topk=args.topk,
                save_logit_model_path=args.logit_path,
            )
        print("Running generations!")
        run_hf_model(
            test_df, model, tokenizer, batch_size=args.eval_batch_size, max_new_tokens=500
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
        "--data_dir",
        type=str,
        default=None,
        help="The directory containing the truthfulqa data. Download from https://github.com/sylinrl/TruthfulQA/tree/main/data."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/truthfulqa/",
        help="The directory to save the results."
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
        "--eval_batch_size",
        type=int,
        default=1,
        help="batch size for evaluation."
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=None
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
        '--settings',
        nargs='+',
        choices=['open', 'mc'],
        help='Settings',
        default=["open"]
    )
    parser.add_argument(
        '--gpt_true_model_name',
        type=str,
        help='If `open` setting is used, the trained GPT truthfulness model name should be provided.',
        default="ft:davinci-002:vita-lab:gpt-judge:9JbgCkev",
    )
    parser.add_argument(
        '--gpt_info_model_name',
        type=str,
        help='If `open` setting is used, the trained GPT informativeness model name should be provided.',
        default="ft:davinci-002:vita-lab:gpt-info:9JbeIvu5"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="LR"
    )
    parser.add_argument(
        "--train",
        action="store_true",
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

    assert args.model_name_or_path or args.base_model_name_or_path

    main(args)
    print(args.save_dir)
