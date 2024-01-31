#! /bin/bash

# Runs the "345M" parameter model
# git config --global http.proxy http://10.202.1.3:18000
# git config --global https.proxy http://10.202.1.3:18000
pip install transformers==4.34.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
# pip install rouge_chinese nltk jieba datasets transformers==4.31.0 deepspeed==0.10.0 accelerate==0.21.0 torch==2.0.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

GPUS_PER_NODE=8
# WORLD_SIZE=1
# MASTER_PORT=6000
# RANK=0
# MASTER_ADDR="localhost"
# export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]]; then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi
ROOT_DIR=$(dirname -- "$(readlink -f -- "$0")")
echo $ROOT_DIR
DATA_ARGS="--data_path /mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/data_config/train_math_direct_0128.json"

output_dir="$ROOT_DIR/outputs/mistral-7B-math-normal-cot-0128"
model_name_or_path="/mnt/pfs/zitao_team/big_model/raw_models/Mistral-7B-v0.1"
# --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 4 \
# --gradient_accumulation_steps 8 \
# 8机
python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    $ROOT_DIR/train_cot_model_sft_direct.py \
    --model_name_or_path $model_name_or_path \
    $DATA_ARGS \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 60 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --deepspeed "/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_config/default_no_offload_param.json" \
    --tf32 True \
    --lazy_preprocess >$ROOT_DIR/training_logs/0128_hcot_mstrial_7b_basic_normal_cot.log
