cd /mnt/pfs/zitao_team/liuchangkun1/repos/tiny-gpt/1215
pip install transformers==4.34.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
lr=2e-5

pretrained_model=None
model_name_or_path=/mnt/pfs/zitao_team/liuchangkun1/repos/tiny-gpt/1215/Llama-2-7b-hf
data_path=/mnt/pfs/zitao_team/liuchangkun1/repos/tiny-gpt/1215/data/COT_model_train_data_kun_idea.json
per_device_train_batch_size=2
gradient_accumulation_steps=8
model_max_length=4096
output_dir=/mnt/pfs/zitao_team/liuchangkun1/repos/tiny-gpt/1215/output/7b-bs-128-mse-2-0

deepspeed_config_file=ds_zero2_no_offload.json
export WANDB_API_KEY=b1b8f5091321fdfaf46a376a95dd8938b2e0ffce


torchrun --nnodes 1 --nproc_per_node 8 --master_port=20001 train.py \
    --deepspeed ${deepspeed_config_file} \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --output_dir ${output_dir} \
    --num_train_epochs 10 \
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
    --lazy_preprocess True
    # --model_name_or_path ${model_name_or_path}