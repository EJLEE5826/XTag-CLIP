#!/bin/bash

# 한글 폰트 설치 및 캐시 갱신 (옵션 - 필요시 주석 해제)
# echo "나눔 폰트 설치 중..."
# FONT_DIR=~/.fonts
# mkdir -p $FONT_DIR
# wget -q https://github.com/naver/nanumfont/releases/download/VER2.5/NanumGothicCoding-2.5.zip -O /tmp/NanumFont.zip
# unzip -q /tmp/NanumFont.zip -d /tmp/NanumFont
# cp /tmp/NanumFont/*.ttf $FONT_DIR/
# fc-cache -f -v
# rm -rf /tmp/NanumFont /tmp/NanumFont.zip

# 결과 저장 디렉토리
OUTPUT_DIR=""
mkdir -p ${OUTPUT_DIR}

# 공통 인자
MODEL_PATH=""
CSV_PATH=""
IMG_DIR=""

# 최대 예측 확률 히트맵 생성
echo "최대 예측 확률 히트맵 생성 중..."
python visualize_max_prob_heatmap.py \
    --model-path ${MODEL_PATH} \
    --csv-path ${CSV_PATH} \
    --img-dir ${IMG_DIR} \
    --output-dir ${OUTPUT_DIR}


# 각 흉터 종류별 주요 특징 분포 차트 생성
echo "각 흉터 종류별 주요 특징 분포 차트 생성 중..."
python visualize_class_feature_distribution.py \
    --model-path ${MODEL_PATH} \
    --csv-path ${CSV_PATH} \
    --img-dir ${IMG_DIR} \
    --output-dir ${OUTPUT_DIR}

echo "모든 시각화 완료! 결과는 ${OUTPUT_DIR} 디렉토리에 저장되었습니다."
