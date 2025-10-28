#!/bin/bash

# 환경변수 설정
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=""

cd src

# 파이썬 모듈 실행
python -m others.main_other \
    --batch-size 16 \
    --workers 4 \
    --report-to wandb \
    --wandb-project-name Scar_BiomedCLIP_pretraining \
    --train-data "" \
    --val-data "" \
    --lock-image \
    --lock-text \
    --precision amp \
    --save-frequency 250 \
    --warmup 50 \
    --lr 5e-6 \
    --wd 0.1 \
    --epochs 150 \
    --local-loss \
    --model hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    --delete-previous-checkpoint \
    --save-most-recent \
    --prompt-template-setting "sentence_1" 