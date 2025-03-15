#!/bin/bash
#SBATCH --cpus-per-task=35
#SBATCH --gres=gpu:3
#SBATCH --mem=80000
#SBATCH --time=3-00:00:00
#SBATCH --output=Test_Run_25thFeb.txt

nvidia-smi

uid="$(date +%Y%m%d_%H%M%S)"
base_model="Qwen/Qwen2.5-1.5B-Instruct"
lr=5e-6
epochs=100
weight_decay=1e-4
micro_batch_size=4
gradient_accumulation_steps=4
max_steps=-1
push_to_hub=false
output_dir="ckpts/s1-${uid}"

# Pass LoRA parameters as arguments to sft.py
python sft.py \
    --block_size=32768 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="simplescaling/s1K_tokenized" \
    --model_name=${base_model} \
    --warmup_ratio=0.05 \
    --logging_steps=1 \
    --save_strategy="steps" \
    --save_steps=500 \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir=${output_dir} \
    --push_to_hub=${push_to_hub} \
    --save_only_model=False \
    --use_lora=True \
    --lora_r=16 \
    --lora_alpha=32 \
    --lora_dropout=0.05  
