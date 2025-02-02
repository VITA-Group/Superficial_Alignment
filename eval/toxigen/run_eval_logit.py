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
    load_dexperts_and_tokenizer,
)
from eval.utils import dynamic_import_function


@torch.no_grad()
def score_generations(
    hate_classifier_tokenizer, hate_classifier_model, generations, batch_size=32
):
    classifications = []
    for i in trange(0, len(generations), batch_size):
        batch = generations[i: i + batch_size]
        inputs = hate_classifier_tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt"
        ).to(hate_classifier_model.device)
        outputs = hate_classifier_model(**inputs)
        classifications.extend(outputs.logits.argmax(dim=-1).tolist())
    return classifications


def main(args):
    random.seed(42)
    os.makedirs(args.save_dir, exist_ok=True)

    # Load the testing data
    examples = []
    prompt_files = glob.glob(os.path.join("data/toxigen", "*.txt"))
    for task_file in tqdm(prompt_files, desc="Loading prompts"):
        with open(task_file, "r") as f:
            group_name = os.path.basename(task_file).split(".")[0]
            label = group_name.split("_")[0]
            minority_group = "_".join(group_name.split("_")[1:])
            group_prompts = [line.strip() for line in f]
            group_prompts = group_prompts[800:]
            # random.shuffle(group_prompts)
            if args.max_examples_per_group is not None:
                group_prompts = group_prompts[:args.max_examples_per_group]

            for prompt in group_prompts:
                # minor cleaning: replace \\ with \
                prompt = prompt.replace("\\\\", "\\")
                prompt = prompt.replace("\\n", "\n")
                examples.append({
                    "text": prompt,
                    "label": label,
                    "target_groups": [minority_group],
                })

    prompts = []
    for example in examples:
        if args.use_chat_format:
            messages = [{"role": "user", "content": "Complete the following: " + example["text"]}]
            chat_formatting_function = dynamic_import_function(args.chat_formatting_function)
            prompt = chat_formatting_function(messages, add_bos=False)
        else:
            # we will use the original text from toxigen as the prompt.
            prompt = example["text"]
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
    elif args.base_model_name_or_path:    # DExperts
        model, tokenizer =   load_dexperts_and_tokenizer(
            args.base_model_name_or_path,
            data_dir=args.data_dir,
            model=args.model,
            alpha=args.alpha,
            load_in_8bit=args.load_in_8bit,
            use_fast_tokenizer=not args.use_slow_tokenizer,
            topk=args.topk,
            save_logit_model_path = args.logit_path

        )


    new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1]
    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=512,
        batch_size=args.eval_batch_size if args.eval_batch_size else 1,
        # stop_id_sequences=[[new_line_token]],
        do_sample=False,
    )

    # Run the toxicity classifier
    toxigen_roberta_tokenizer = AutoTokenizer.from_pretrained("tomh/toxigen_roberta")
    toxigen_roberta_classifier = AutoModelForSequenceClassification.from_pretrained(
        "tomh/toxigen_roberta"
    )
    if torch.cuda.is_available():
        toxigen_roberta_classifier = toxigen_roberta_classifier.cuda()

    classifications = score_generations(
        toxigen_roberta_tokenizer,
        toxigen_roberta_classifier,
        outputs,
        batch_size=args.classifier_batch_size,
    )
    all_groups_scores = defaultdict(list)
    for example, output, classification in zip(examples, outputs, classifications):
        example["prediction"] = output
        example["is_toxic"] = classification
        for group in example["target_groups"]:
            all_groups_scores[group].append(classification)
    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as fout:
        for example in examples:
            fout.write(json.dumps(example) + "\n")

    # Calculate the performance
    performance = {}
    # calculate score by targeted group
    for group in all_groups_scores:
        performance[group] = sum(all_groups_scores[group]) / len(
            all_groups_scores[group]
        )
    # overall perf
    performance["overall"] = sum(classifications) / len(classifications)
    performance["logit_path"] = args.logit_path
    print(args.logit_path)

    with open(os.path.join(args.save_dir, "metrics.json"), "a") as fout:
        print(f"Average Toxicity: {performance['overall']}")
        for group in performance:
            if group != "overall":
                print(f"Average Toxicity for {group}: {performance[group]}")
        json.dump(performance, fout, indent=4)
        fout.write("\n\n")


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
        default="results/toxigen"
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
        "--classifier_batch_size",
        type=int,
        default=32,
        help="batch size to use for toxicity classifier.",
    )
    parser.add_argument(
        "--classifier_device",
        type=str,
        default="cuda",
        help="device to use for toxicity classifier.",
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
        "--max_examples_per_group",
        type=int,
        default=None,
        help="If given, we will only use this many instances per group. Default to 500 (half the available instances).",
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

    assert args.model_name_or_path or args.base_model_name_or_path

    main(args)
    print(args.save_dir)
