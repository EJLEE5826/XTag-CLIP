#!/bin/bash

# 환경변수 설정
export CUDA_VISIBLE_DEVICES=2
export TRANSFORMERS_CACHE=""

cd src

# 파이썬 모듈 실행
python -m others.main_other_simple \
    --batch-size 4 \
    --workers 4 \
    --train-data "" \
    --val-data "" \
    --precision amp \
    --save-frequency 250 \
    --warmup 50 \
    --lr 5e-5 \
    --wd 0.1 \
    --epochs 500 \
    --local-loss \
    --model ViT-B-32 \
    --pretrained laion400m_e32 \
    --cache-dir "" \
    --prompt-template-setting "" \
    --use-tagging \
    --use-fusion \


