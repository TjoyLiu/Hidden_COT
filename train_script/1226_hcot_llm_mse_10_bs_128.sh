#! /bin/bash
# cd /mnt/pfs/zitao_team/tianqiaoliu/Project/papers/hidden_cot/train_script
# pip install transformers==4.32.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
GPUS_PER_NODE=8
is_master=${MASTER-"0"}
if [[ $is_master -eq 1 ]];then
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $HOSTNAME --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $WORLD_SIZE --node_rank $RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
fi
ROOT_DIR=$( dirname -- "$( readlink -f -- "$0"; )"; )
echo $ROOT_DIR

lr=2e-5

pretrained_model=None
model_name_or_path=/mnt/pfs/zitao_team/liuchangkun1/repos/tiny-gpt/1215/Llama-2-7b-hf
data_path=/mnt/pfs/zitao_team/tianqiaoliu/Project/papers/hidden_cot/data/GSM8K/hcot_llm_data.json
per_device_train_batch_size=1
gradient_accumulation_steps=2
model_max_length=4096
output_dir="$ROOT_DIR/outputs/llama2-7B-gsm8k-hcot-llm-mse-10-bs-128"
hcot_config_path=/mnt/pfs/zitao_team/tianqiaoliu/Project/teammates/hidden_cot/train_script/hcot_llm_model/config.json

deepspeed_config_file=/mnt/pfs/zitao_team/tianqiaoliu/Project/papers/hidden_cot/train_script/ds_zero2_no_offload.json


python -m torch.distributed.launch $DISTRIBUTED_ARGS \
    $ROOT_DIR/train_hcot_llm_model.py \
    --hcot_config_name ${hcot_config_path}\
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 100 \
    --learning_rate ${lr} \
    --mse_ratio 2.0 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --model_max_length ${model_max_length} \
    --gradient_checkpointing True \
    --overwrite_output_dir \
    --lazy_preprocess True >$ROOT_DIR/training_logs/1226_hcot_llm_mse_10_bs_128.log