#!/bin/bash

cuda="0,1,2,3"

task_name="sft"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$cuda \
torchrun --standalone --nnodes 1 --nproc-per-node 4 ../openvla/vla-scripts/finetune.py \
  --vla_path "../weights/openvla-7b-rlvla-warmup" \
  --data_root_dir "/home/kyzhang/workspace/kyzhang/datasets/ms3_datasets" \
  --dataset_name ${task_name} \
  --run_root_dir checkpoints/${task_name} \
  --lora_rank 32 \
  --batch_size 12 \
  --max_steps 60000 \
  --eval_steps 200 \
  --save_steps "0,2500,5000,7500,10000,15000,20000,25000,30000,35000,40000,45000,50000,55000,60000" \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project "RLVLA_sft"