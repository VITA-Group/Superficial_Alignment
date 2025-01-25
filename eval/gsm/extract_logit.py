import argparse
import os
import re
import json
import random
import torch
import evaluate
from eval.utils import (
    generate_completions_extract_logits,
    dynamic_import_function,
    ensure_dir
)
from modeling.dexperts import DExpertsLlamaExtract
from transformers import AutoTokenizer


exact_match = evaluate.load("exact_match")


def trim_output(output):
    instruction_prefix = "Answer the following question"
    question_prefix = 'Question:'
    comment_prefix = 'Comment:'  # for some reason, Llama 13B likes to generate these comments indefinitely

    for prefix in [instruction_prefix, question_prefix, comment_prefix]:
        if prefix in output:
            output = output.split(prefix)[0]

    return output


def main(args):
    random.seed(42)

    print("Loading data...")
    train_data = []
    with open(os.path.join("data/gsm", "train.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            train_data.append({
                "question": example["question"],
                "answer": example["answer"].split("####")[1].strip()
            })
    if args.start >= 0:
        train_data = train_data[args.start:]
    if args.samples >= 0:
        train_data = train_data[:args.samples]
    for example in train_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])
        # assert float(example["answer"]), f"answer is not a valid number: {example['answer']}"


    ensure_dir(args.save_dir)
    ensure_dir(args.logit_dir)

    prompt_prefix = "Answer the following question.\n\n"

    prompts = []
    chat_formatting_function = dynamic_import_function(args.chat_formatting_function) if args.use_chat_format else None
    for example in train_data:
        prompt = prompt_prefix + "Question: " + example["question"].strip()
        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = chat_formatting_function(messages, add_bos=False)
            if prompt[-1] in ["\n", " "]:
                prompt += "Answer:"
            else:
                prompt += " Answer:"
        else:
            prompt += "\nAnswer:"
        prompts.append(prompt)

    # with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
    #     fout.write(prompts[0])

    


    # if not tokenizer_name_or_path:
    tokenizer_name_or_path = args.expert_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast_tokenizer=not args.use_slow_tokenizer)

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
        system_prompt=None,
        chat_response_prefix="Answer:",
        model_kwargs=model_kwargs,
    )

    outputs, expert_logits, antiexpert_logits, expert_hiddens, antiexpert_hiddens = generate_completions_extract_logits(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=1,
        do_sample=args.do_sample,
        temperature=args.temperature
    )
    expert = torch.cat(expert_logits, dim=0)
    antiexpert = torch.cat(antiexpert_logits, dim=0)
    antiexpert_hiddens = torch.cat(antiexpert_hiddens, dim=0)
    assert expert.shape[0] == antiexpert.shape[0]
    top=500
    expert_logit, expert_topk = torch.topk(expert, k=top, dim=-1)
    antiexpert_logit, antiexpert_topk = torch.topk(antiexpert, k=top, dim=-1)
    # expert_softmax = expert_softmax[torch.arange(expert.shape[0])[:, None], expert_topk]
    expert_dict = {"logit": expert_logit, "indices": expert_topk}
    antiexpert_dict = {"logit": antiexpert_logit, "indices": antiexpert_topk}
    torch.save(expert_dict, os.path.join(args.logit_dir, f"expert_{args.start}_{args.start + args.samples}.bin"))
    torch.save(antiexpert_dict, os.path.join(args.logit_dir, f"antiexpert_{args.start}_{args.start + args.samples}.bin"))
    torch.save(antiexpert_hiddens, os.path.join(args.logit_dir, f"antiexpert_hidden_{args.start}_{args.start + args.samples}.bin"))
    expert_hiddens = torch.cat(expert_hiddens, dim=0)
    torch.save(expert_hiddens, os.path.join(args.logit_dir, f"expert_hidden_{args.start}_{args.start + args.samples}.bin"))

    torch.save(model.antiexpert.lm_head.weight.detach().cpu(), os.path.join(args.logit_dir, "emb.pt"))



    outputs = [trim_output(o) for o in outputs]

    predictions = []
    for output in outputs:
        # replace numbers like `x,xxx` with `xxxx`
        output = re.sub(r"(\d),(\d)", r"\1\2", output)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
        if numbers:
            predictions.append(numbers[-1])
        else:
            predictions.append(output)

    print("Calculating accuracy...")
    targets = [example["answer"] for example in train_data]

    em_score = exact_match.compute(predictions=predictions, references=targets, ignore_case=True, ignore_punctuation=True)["exact_match"]
    print(f"Exact match : {em_score}")





    predictions = [{
        "question": example["question"],
        "answer": example["answer"],
        "model_output": output,
        "prediction": pred
    } for example, output, pred in zip(train_data, outputs, predictions)]

    with open(os.path.join(args.save_dir, f"predictions_{args.start}_{args.start + args.samples}.jsonl"), "w") as fout:
        for prediction in predictions:
            fout.write(json.dumps(prediction) + "\n")

    with open(os.path.join(args.save_dir, f"metrics_{args.start}_{args.start + args.samples}.json"), "w") as fout:
        json.dump({
            "exact_match": em_score
        }, fout, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="maximum number of examples to evaluate."
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
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
        help="if specified, we will load the model to generate the predictions."
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
        help="if specified, we will load the tokenizer from here."
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If given, we will use the slow tokenizer."
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="load model in 8bit mode, which will reduce memory and speed up inference."
    )
    parser.add_argument(
        "--expert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-chat-hf',
    )
    parser.add_argument(
        "--antiexpert_model_name_or_path",
        type=str,
        default='meta-llama/Llama-2-7b-hf',
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
