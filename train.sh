#!/bin/bash

deepspeed --num_gpus 2 \
  sft.py \
  --tensorboard_path ./runs \
  --num_train_epochs 10 \
  --max_seq_len 512 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --zero_stage 2 \

 