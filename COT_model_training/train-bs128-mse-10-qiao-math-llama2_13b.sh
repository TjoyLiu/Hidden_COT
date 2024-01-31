#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=$(dirname -- "$(readlink -f -- "$0")")

pip install transformers==4.34.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

MODEL="/mnt/pfs/zitao_team/big_model/raw_models/Llama-2-13b-hf" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/COT_model_training/data/data_config_math_0126.json"

GPUS_PER_NODE=8
# NNODES=1
# NODE_RANK=0
# MASTER_ADDR=localhost
# MASTER_PORT=6001
# DISTRIBUTED_ARGS="
#     --nproc_per_node $GPUS_PER_NODE \
#     --nnodes $NNODES \
#     --node_rank $NODE_RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT
# "
is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]]; then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi

torchrun $DISTRIBUTED_ARGS $DIR/train.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --bf16 True \
    --output_dir $DIR/output/7b-bs-mse-10-0-llama2-13b \
    --num_train_epochs 10 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 100 \
    --learning_rate 2e-5 \
    --mse_ratio 10.0 \
    --weight_decay 0.0 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --gradient_checkpointing True \
    --deepspeed /mnt/pfs/zitao_team/tianqiaoliu/Project/llm_team/source/training/sft/train_scripts/config/default_offload_opt_param.json
