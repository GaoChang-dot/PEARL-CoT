# PEARL-CoT
This repository will provide the codes and data used in PEARL-CoT.
## Helpfulness Scorer
To train the helpfulness scoring module, run:
```
cd scorer
python scorer_train.py
```
To perform inference on the test set, run:
```
python scorer_infer.py
```
## Policy Warm Up
To initialize Qwen/Qwen2.5-1.5B-Instruct with basic reasoning capabilities, run:
```
cd sft
nohup deepspeed --include=localhost:0 sft_train.py \
--base_model_type qwen > sft_train.log 2>&1 &
```
To merge LoRA and base model weights, run:
```
python merge_model.py --base_model_type qwen
```
To perform inference using the trained model, run:
```
nohup deepspeed --include=localhost:0 sft_infer.py \
--base_model_type qwen > sft_infer.log 2>&1 &
```
## Reasoning-to-Response GRPO
To further enhance the reasoning capability of the policy model using GRPO fine-tuning on Qwen/Qwen2.5-1.5B-Instruct, run:
```
cd grpo
bash qwen_grpo_finetune.sh
```
To merge LoRA and base model weights after fine-tuning, run:
```
python merge_model.py --base_model_type qwen
```
To conduct inference on the GRPO fine-tuned model, run:
```
nohup deepspeed --include=localhost:0 grpo_infer.py \
--base_model_type qwen > grpo_infer.log 2>&1 &
```
To use the Meta-Llama/Llama-3.2-1B-Instruct model instead of Qwen, simply replace all instances of ``qwen`` with ``llama`` in the commands above.