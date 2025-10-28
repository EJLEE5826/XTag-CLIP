cd src

python -m others.main_other \
    --batch-size 1 \
    --force-image-size 224 \
    --name Scar_ViT-B-32_laion400m_e32_9 \
    --val-data "" \
    --model ViT-B-32 \
    --pretrained laion400m_e32 \
    --save-embed