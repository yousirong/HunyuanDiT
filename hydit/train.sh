# task_flag="dit_g2_full_1024p"                                 # the task flag is used to identify folders.
# resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt # checkpoint root for model resume
# resume_ema_root=./ckpts/t2i/model/pytorch_model_ema.pt      # checkpoint root for ema resume
# index_file=dataset/porcelain/jsons/porcelain.json             # index file for dataloader
# results_dir=./log_EXP                                         # save root for results
# batch_size=1                                                  # training batch size
# image_size=1024                                               # training image resolution
# grad_accu_steps=1                                             # gradient accumulation
# warmup_num_steps=0                                            # warm-up steps
# lr=0.0001                                                     # learning rate
# ckpt_every=9999999                                            # create a ckpt every a few steps.
# ckpt_latest_every=9999999                                     # create a ckpt named `latest.pt` every a few steps.
# ckpt_every_n_epoch=2                                          # create a ckpt every a few epochs.
# epochs=8                                                      # total training epochs


# sh $(dirname "$0")/run_g.sh \
#     --task-flag ${task_flag} \
#     --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
#     --predict-type v_prediction \
#     --uncond-p 0 \
#     --uncond-p-t5 0 \
#     --index-file ${index_file} \
#     --random-flip \
#     --lr ${lr} \
#     --batch-size ${batch_size} \
#     --image-size ${image_size} \
#     --global-seed 999 \
#     --grad-accu-steps ${grad_accu_steps} \
#     --warmup-num-steps ${warmup_num_steps} \
#     --use-flash-attn \
#     --use-fp16 \
#     --extra-fp16 \
#     --results-dir ${results_dir} \
#     --resume \
#     --resume-module-root ${resume_module_root} \
#     --resume-ema-root ${resume_ema_root} \
#     --epochs ${epochs} \
#     --ckpt-every ${ckpt_every} \
#     --ckpt-latest-every ${ckpt_latest_every} \
#     --ckpt-every-n-epoch ${ckpt_every_n_epoch} \
#     --log-every 10 \
#     --deepspeed \
#     --use-zero-stage 2 \
#     --gradient-checkpointing \
#     --cpu-offloading \
#     "$@"

#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29500
export PYTHONPATH=IndexKits:$PYTHONPATH
# Waymo 자율주행 데이터셋으로 HunyuanDiT 기본 모델 훈련 스크립트
# Usage: bash scripts/train_waymo_base.sh [sample_limit]

# 기본 설정
SAMPLE_LIMIT=${1:-50000}  # 첫 번째 인수로 샘플 수 제한 (기본값: 1000)
BATCH_SIZE=12
LEARNING_RATE=1e-4
MAX_STEPS=104000

echo "=========================================="
echo "Waymo 자율주행 데이터셋 기본 모델 훈련 시작"
echo "샘플 제한: $SAMPLE_LIMIT"
echo "배치 사이즈: $BATCH_SIZE"
echo "학습률: $LEARNING_RATE"
echo "최대 스텝: $MAX_STEPS"
echo "=========================================="
resume_module_root=/mnt/ssd/HunyuanDiT/results/waymo_base/001-DiT-g-2/checkpoints/latest.pt/mp_rank_00_model_states.pt

# DeepSpeed를 사용한 기본 모델 훈련
sh $(dirname "$0")/run_g.sh \
    --image-size 256 256 \
    --index-file /mnt/ssd/HunyuanDiT/dataset/waymo/jsons/waymo.json \
    --random-flip \
    --lr ${LEARNING_RATE} \
    --noise-schedule scaled_linear \
    --beta-start 0.00085 \
    --beta-end 0.02 \
    --predict-type v_prediction \
    --uncond-p 0.1 \
    --uncond-p-t5 0.1 \
    --log-every 10 \
    --ckpt-every 5000 \
    --ckpt-latest-every 5000 \
    --ckpt-every-n-epoch 10 \
    --epochs 200 \
    --max-training-steps ${MAX_STEPS} \
    --batch-size ${BATCH_SIZE} \
    --grad-accu-steps 1 \
    --warmup-num-steps 1000 \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    --cpu-offloading \
    --extra-fp16 \
    --use-fp16 \
    --deepspeed \
    --sample-limit ${SAMPLE_LIMIT} \
    --training-parts all \
    --results-dir /mnt/ssd/HunyuanDiT/results/waymo_base \
    --resume \
    --resume-module-root ${resume_module_root} \
    "$@"