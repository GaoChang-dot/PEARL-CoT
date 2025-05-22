export DEBUG_MODE="true"
export CUDA_VISIBLE_DEVICES=0,1

RUN_NAME="GRPO"
export LOG_PATH="./debug_log_$RUN_NAME.txt"

nohup torchrun --nproc_per_node="2" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="24866" \
    grpo_finetune.py \
    --base_model_type "llama" \
    > llama_grpo_train.out 2>&1 &