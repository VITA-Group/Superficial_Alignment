export CUDA_VISIBLE_DEVICES=2
python  -m eval.gsm.run_eval_logit \
--data_dir  logits/gsm/dexperts-7B-chat7B \
--save_dir results/gsm/dexperts-7B-chat7B \
--base_model_name_or_path meta-llama/Llama-2-7b-hf \
--eval_batch_size 10 \
--model "hidden-logit-residual" \
--logit_path "[logit_path]"

python  -m eval.advbench.run_eval_logit \
--data_dir  logits/advbench/dexperts-7B-chat7B \
--save_dir results/advbench/dexperts-7B-chat7B \
--base_model_name_or_path meta-llama/Llama-2-7b-hf \
--eval_batch_size 10 \
--model "hidden-logit-residual" \
--eval \
--metric gpt3.5 \
--logit_path "[logit_path]"

python  -m eval.toxigen.run_eval_logit \
--data_dir  logits/toxigen/dexperts-7B-chat7B \
--save_dir results/toxigen/dexperts-7B-chat7B \
--base_model_name_or_path meta-llama/Llama-2-7b-hf \
--eval_batch_size 10 \
--model "hidden-logit-residual" \
--logit_path "[logit_path]"


python  -m eval.truthfulqa.run_eval_logit \
--data_dir  logits/truthfulqa/dexperts-7B-chat7B \
--save_dir results/truthfulqa/dexperts-7B-chat7B \
--base_model_name_or_path meta-llama/Llama-2-7b-hf \
--eval_batch_size 10 \
--model "hidden-logit-residual" \
--logit_path "[logit_path]"