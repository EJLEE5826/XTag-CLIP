#!/bin/bash

cd src

# 파이썬 모듈 실행
python -m others.main_other_simple \
    --batch-size 4 \
    --workers 4 \
    --report-to wandb \
    --wandb-project-name Scar_ViT-B-32_laion400m_e32 \
    --train-data "" \
    --val-data "" \
    --precision amp \
    --save-frequency 250 \
    --warmup 50 \
    --lock-text \
    --lr 1e-5 \
    --wd 0.1 \
    --epochs 75 \
    --local-loss \
    --model ViT-B-32 \
    --pretrained laion400m_e32 \
    --prompt-template-setting "sentence_1" \
    --use-fusion \
    --use-tagging \
