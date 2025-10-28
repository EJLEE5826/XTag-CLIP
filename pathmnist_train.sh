cd src

# 파이썬 모듈 실행
python -m others.main_other \
    --batch-size 16 \
    --workers 4 \
    --report-to wandb \
    --wandb-project-name Scar_ViT-B-32_laion400m_e32 \
    --train-data "" \
    --val-data "" \
    --precision amp \
    --save-frequency 100 \
    --warmup 50 \
    --lr 5e-6 \
    --wd 0.1 \
    --epochs 500 \
    --local-loss \
    --model ViT-B-32 \
    --delete-previous-checkpoint \
    --save-most-recent \