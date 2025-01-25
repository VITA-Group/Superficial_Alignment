python -m eval.gsm.extract_logit \
--save_dir logits/gsm/dexperts-7B-chat7B/ \
--expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--antiexpert_model_name_or_path meta-llama/Llama-2-7b-hf \
--start 0 \
--samples 1000


python -m eval.advbench.extract_logit \
--save_dir logits/advbench/dexperts-7B-chat7B \
--expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--antiexpert_model_name_or_path meta-llama/Llama-2-7b-hf \
--start 0 \
--samples 500


python -m eval.toxigen.extract_logit \
--save_dir logits/toxigen/dexperts-7B-chat7B \
--expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--antiexpert_model_name_or_path meta-llama/Llama-2-7b-hf \
--start 0 \
--samples 1000


python -m eval.truthfulqa.extract_logit \
--save_dir logits/truthfulqa/dexperts-7B-chat7B \
--expert_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
--antiexpert_model_name_or_path meta-llama/Llama-2-7b-hf \
--start 0 \
--samples 800

python softlink.py --src_dir logits/gsm/dexperts-7B-chat7B/ --dst_dir logits/all/dexperts-7B-chat7B/ --src_prefix 0_1000 --dst_prefix 0_1000
python softlink.py --src_dir logits/advbench/dexperts-7B-chat7B/ --dst_dir logits/all/dexperts-7B-chat7B/ --src_prefix 0_500 --dst_prefix 1000_1500
python softlink.py --src_dir logits/toxigen/dexperts-7B-chat7B/ --dst_dir logits/all/dexperts-7B-chat7B/ --src_prefix 0_1000 --dst_prefix 1500_2500
python softlink.py --src_dir logits/truthfulqa/dexperts-7B-chat7B/ --dst_dir logits/all/dexperts-7B-chat7B/ --src_prefix 0_800 --dst_prefix 2500_3300


