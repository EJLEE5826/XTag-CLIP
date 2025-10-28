#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
흉터 태그별 최대 예측 확률 히트맵 시각화 스크립트
모델이 특정 흉터 종류를 예측할 때 어떤 tag에 가장 주목하는지 시각화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# 파일 경로 설정
tagging_output_path = ''
class_output_path = ''
output_dir = ''

# 출력 디렉토리가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

def parse_tagging_output(file_path, columns=None, start_line=0, end_line=None):
    """
    태깅 출력 파일에서 특정 열과 특정 범위의 행만 파싱
    
    Args:
        file_path: 파일 경로
        columns: 읽어올 특정 열 인덱스 리스트 (None인 경우 모든 열 읽기)
        start_line: 읽기 시작할 줄 번호 (0-기반 인덱스)
        end_line: 읽기를 종료할 줄 번호 (None인 경우 파일 끝까지)
    """
    print(f"태깅 데이터 로드 중: {file_path} (행 범위: {start_line}~{end_line if end_line is not None else '끝'})")
    
    data = []
    with open(file_path, 'r') as file:
        # 모든 줄을 읽음
        lines = file.readlines()
        
        # 범위 확인 및 조정
        if end_line is None or end_line > len(lines):
            end_line = len(lines)
        
        # 지정된 범위의 줄만 처리
        for line_idx, line in enumerate(lines[start_line:end_line]):
            line = line.strip()
            if not line:
                continue
                
            # GT 태그 - 예측 태그 형식 파싱
            parts = line.split(' - ')
            if len(parts) == 2:
                # 특정 열만 추출
                if columns is not None:
                    gt_tags = []
                    pred_tags = []
                    
                    gt_parts = parts[0].split(',')
                    pred_parts = parts[1].split(',')
                    
                    # 요청된 열 인덱스만 추출
                    for col_idx in columns:
                        if col_idx < len(gt_parts) and col_idx < len(pred_parts):
                            gt_tags.append(gt_parts[col_idx].strip())
                            pred_tags.append(pred_parts[col_idx].strip())
                else:
                    # 모든 열 추출
                    gt_tags = [tag.strip() for tag in parts[0].split(',')]
                    pred_tags = [tag.strip() for tag in parts[1].split(',')]
                
                data.append({
                    'gt_tags': gt_tags,
                    'pred_tags': pred_tags
                })
    
    print(f"총 {len(data)}개의 태깅 항목을 파싱했습니다.")
    return data

def parse_class_output(file_path, columns=None, start_line=0, end_line=None):
    """
    클래스 출력 파일에서 특정 열과 특정 범위의 행만 파싱
    
    Args:
        file_path: 파일 경로
        columns: 예측 확률에서 읽어올 특정 열 인덱스 리스트 (None인 경우 모든 열 읽기)
        start_line: 읽기 시작할 줄 번호 (0-기반 인덱스)
        end_line: 읽기를 종료할 줄 번호 (None인 경우 파일 끝까지)
    """
    print(f"클래스 데이터 로드 중: {file_path} (행 범위: {start_line}~{end_line if end_line is not None else '끝'})")
    
    data = []
    classes = set()
    
    with open(file_path, 'r') as file:
        # 모든 줄을 읽음
        lines = file.readlines()
        
        # 범위 확인 및 조정
        if end_line is None or end_line > len(lines):
            end_line = len(lines)
        
        # 지정된 범위의 줄만 처리
        for line in lines[start_line:end_line]:
            line = line.strip()
            if not line:
                continue
                
            # GT 클래스 - 예측 클래스 - 예측값 형식 파싱
            parts = line.split(' - ')
            if len(parts) == 3:
                gt_class = parts[0].strip()
                pred_class = parts[1].strip()
                
                # 예측값 파싱 (문자열 "[13.208, 11.157, 12.532]" -> 리스트 [13.208, 11.157, 12.532])
                pred_values_str = parts[2].strip('[]')
                all_pred_values = [float(val) for val in pred_values_str.split(',')]
                
                # 특정 열만 추출
                if columns is not None:
                    pred_values = []
                    for col_idx in columns:
                        if col_idx < len(all_pred_values):
                            pred_values.append(all_pred_values[col_idx])
                else:
                    pred_values = all_pred_values
                
                data.append({
                    'gt_class': gt_class,
                    'pred_class': pred_class,
                    'pred_values': pred_values
                })
                
                classes.add(gt_class)
    
    print(f"총 {len(data)}개의 클래스 항목을 파싱했습니다.")
    print(f"발견된 GT 클래스: {', '.join(sorted(classes))}")
    
    return data, sorted(list(classes))

def combine_data(tagging_data, class_data):
    """
    태깅 데이터와 클래스 데이터 연결
    두 데이터의 행 수가 다를 경우 처리
    """
    print("태깅 데이터와 클래스 데이터 통합 중...")
    print(f"태깅 데이터: {len(tagging_data)}개, 클래스 데이터: {len(class_data)}개")
    
    # 태그 그룹 정의
    tag_groups_dict = {
        'Width': ['Linear Width', 'Widened Width', 'Linear bulging Width'],
        'Color': ['Normal Color', 'Pink Color', 'Red Color', 'Purple Color'],
        'Pigmentation': ['Normal Pigmentation', 'Pigmented Pigmentation', 'Hypopigmented Pigmentation'],
        'Surface': ['Flat Surface', 'Hypertrophic Surface', 'Keloid Surface', 'Atrophic Surface'],
        'Irregular Color': ['no Irregular Color', 'mild Irregular Color', 'moderate Irregular Color', 'severe Irregular Color'],
        'Irregular Height': ['no Irregular Height', 'mild Irregular Height', 'moderate Irregular Height', 'severe Irregular Height']
    }
    tag_groups = list(tag_groups_dict.keys())
    
    # 데이터 길이가 다를 경우의 처리 방식
    combined_data = []
    min_length = min(len(tagging_data), len(class_data))
    
    for i in range(min_length):
        tag_item = tagging_data[i]
        class_item = class_data[i]
        
        # 태그 분류
        gt_grouped_tags, _ = categorize_tags(tag_item['gt_tags'], tag_groups_dict)
        pred_grouped_tags, _ = categorize_tags(tag_item['pred_tags'], tag_groups_dict)
        
        combined_data.append({
            'gt_tags': gt_grouped_tags,
            'pred_tags': pred_grouped_tags,
            'gt_class': class_item['gt_class'],
            'pred_class': class_item['pred_class'],
            'pred_values': class_item['pred_values']
        })
    
    # 통합 데이터 개수 출력
    print(f"총 {len(combined_data)}개의 데이터가 통합되었습니다.")
    
    return combined_data, tag_groups

def categorize_tags(tags, tag_groups_dict=None):
    """
    태그를 특성 그룹별로 분류
    
    Args:
        tags: 태그 리스트
        tag_groups_dict: 태그 그룹 정의 딕셔너리 (없으면 기본값 사용)
    """
    # 태그 그룹 정의 (인자로 받지 않은 경우 기본값 사용)
    if tag_groups_dict is None:
        tag_groups_dict = {
            'Width': ['Linear Width', 'Widened Width', 'Linear bulging Width'],
            'Color': ['Normal Color', 'Pink Color', 'Red Color', 'Purple Color'],
            'Pigmentation': ['Normal Pigmentation', 'Pigmented Pigmentation', 'Hypopigmented Pigmentation'],
            'Surface': ['Flat Surface', 'Hypertrophic Surface', 'Keloid Surface', 'Atrophic Surface'],
            'Irregular Color': ['no Irregular Color', 'mild Irregular Color', 'moderate Irregular Color', 'severe Irregular Color'],
            'Irregular Height': ['no Irregular Height', 'mild Irregular Height', 'moderate Irregular Height', 'severe Irregular Height']
        }
    
    # 각 그룹별로 분류된 태그 저장 구조
    grouped_tags = {}
    
    # 각 태그에 대해 어떤 그룹에 속하는지 확인
    for tag in tags:
        for group, group_tags in tag_groups_dict.items():
            if tag in group_tags:
                # 해당 그룹에 태그 저장 (그룹:태그)
                grouped_tags[group] = tag
                break
    
    return grouped_tags, tag_groups_dict

def analyze_tag_max_probabilities(combined_data, classes, tag_groups):
    """클래스별, 태그 그룹별 최대 예측 확률 계산"""
    print("클래스별 태그 그룹 최대 예측 확률 분석 중...")
    
    # 클래스별, 태그 그룹별 최대 예측 확률 저장 구조
    # {클래스: {태그그룹: [정확도 리스트]}}
    class_tag_max_probs = {cls: {group: [] for group in tag_groups} for cls in classes}
    
    for item in combined_data:
        gt_class = item['gt_class']
        gt_tags = item['gt_tags']
        pred_tags = item['pred_tags']
        
        # 각 태그 그룹별로 최대 예측 확률 계산
        for group in tag_groups:
            # 해당 그룹의 태그가 있는 경우만 계산
            if group in gt_tags and group in pred_tags:
                # 예측이 맞았으면 1, 틀렸으면 0
                accuracy = 1 if gt_tags[group] == pred_tags[group] else 0
                class_tag_max_probs[gt_class][group].append(accuracy)
    
    # 평균 계산
    class_tag_avg_max_probs = {}
    for cls in classes:
        class_tag_avg_max_probs[cls] = {}
        for group in tag_groups:
            accuracies = class_tag_max_probs[cls][group]
            if accuracies:  # 데이터가 있는 경우
                class_tag_avg_max_probs[cls][group] = np.mean(accuracies)
            else:  # 데이터가 없는 경우
                class_tag_avg_max_probs[cls][group] = 0.0
    
    return class_tag_avg_max_probs

def plot_max_prob_heatmap(class_tag_avg_max_probs, classes, tag_groups):
    """
    태그 그룹별 (행) x 예측 클래스별 (열) 히트맵 생성 (6x3 형태)
    행: 6개 태그 그룹
    열: 3개 예측 클래스
    """
    print("6x3 최대 예측 확률 히트맵 생성 중...")
    
    # 사용할 클래스 제한 (최대 3개)
    used_classes = classes[:3] if len(classes) >= 3 else classes
    
    # 데이터프레임 생성 (클래스 x 태그 그룹) - 전치된 형태
    df = pd.DataFrame(index=used_classes, columns=tag_groups)
    
    # 데이터 채우기
    for cls in used_classes:
        for group in tag_groups:
            if cls in class_tag_avg_max_probs and group in class_tag_avg_max_probs[cls]:
                df.loc[cls, group] = float(class_tag_avg_max_probs[cls][group])
            else:
                df.loc[cls, group] = 0.0
    
    # 데이터타입을 float으로 명시적 변환
    df = df.astype(float)
    
    # 히트맵 생성
    plt.figure(figsize=(12, 7))
    ax = sns.heatmap(
        df,
        annot=True,
        cmap='YlGnBu',
        vmin=0,
        vmax=1,
        fmt='.2f',
        cbar=False,
        annot_kws={"size": 15},
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    
    # 제목 및 축 레이블 제거
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # 레이아웃 조정
    plt.tight_layout()
    
    # 파일 저장
    output_path = os.path.join(output_dir, 'scar_tag_max_prob_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"히트맵이 저장되었습니다: {output_path}")
    
    # CSV로도 저장
    csv_path = os.path.join(output_dir, 'scar_tag_max_prob_data.csv')
    df.to_csv(csv_path)
    print(f"데이터가 CSV 파일로 저장되었습니다: {csv_path}")
    
    return plt.gcf()

def main():
    """메인 함수"""
    print("흉터 태그별 최대 예측 확률 히트맵 분석 시작...")
    
    # 태깅 데이터와 클래스 데이터의 행 범위를 각각 다르게 설정
    tagging_start_line = 1803    # 태깅 데이터 시작 행
    tagging_end_line = 1899    # 태깅 데이터 종료 행
    
    class_start_line = 1667      # 클래스 데이터 시작 행
    class_end_line = 1763       # 클래스 데이터 종료 행
    
    # 특정 열 인덱스 설정 (필요한 경우)
    tagging_columns = None    # 모든 열 사용
    class_columns = None      # 모든 열 사용
    
    # 데이터 로드 및 파싱 (각각 다른 행 범위 사용)
    tagging_data = parse_tagging_output(
        tagging_output_path,
        columns=tagging_columns,
        start_line=tagging_start_line,
        end_line=tagging_end_line
    )
    
    class_data, classes = parse_class_output(
        class_output_path,
        columns=class_columns,
        start_line=class_start_line,
        end_line=class_end_line
    )
    
    # 행 수가 다를 경우 combine_data 함수에서 자동으로 처리
    combined_data, tag_groups = combine_data(tagging_data, class_data)
    
    # 클래스별 태그 그룹 최대 예측 확률 분석
    class_tag_avg_max_probs = analyze_tag_max_probabilities(combined_data, classes, tag_groups)
    
    # 히트맵 생성
    plot_max_prob_heatmap(class_tag_avg_max_probs, classes, tag_groups)
    
    print("분석 완료!")

if __name__ == "__main__":
    main()