#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
특징별 흉터 종류 분포 시각화 코드 (Class와 Tag 정보 모두 활용)
특정 줄만 선택적으로 로드하는 기능 추가
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# 파일 경로 설정
tagging_file = 'output.txt'
class_file = 'output.txt'
output_dir = ''

# 기본 줄 범위 설정
tag_start_line = 1803
tag_end_line = 1899  # None은 파일 끝까지 읽기
class_start_line = 1667
class_end_line = 1763  # None은 파일 끝까지 읽기

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

def read_lines_in_range(file_path, start_line=0, end_line=None):
    """특정 범위의 줄만 파일에서 읽기"""
    result = []
    try:
        with open(file_path, 'r') as file:
            # 모든 줄을 읽음
            all_lines = file.readlines()
            
            # 지정된 범위의 줄만 선택
            if end_line is None:
                selected_lines = all_lines[start_line:]
            else:
                selected_lines = all_lines[start_line:end_line]
            
            # 리스트로 변환
            result = [line.strip() for line in selected_lines]
            
        print(f"파일에서 {len(result)}개 줄 읽음: {file_path} (줄 {start_line}부터 {end_line if end_line else '끝'}까지)")
    except Exception as e:
        print(f"파일 읽기 오류: {e}")
    
    return result

def parse_class_data(file_path, start_line=0, end_line=None):
    """클래스 정보 파일 파싱"""
    print(f"클래스 파일 로드 중: {file_path} (줄 {start_line}부터 {end_line if end_line else '끝'}까지)")
    class_data = []
    
    # 특정 범위의 줄만 읽기
    lines = read_lines_in_range(file_path, start_line, end_line)
    
    for line in lines:
        if ' - ' in line:
            parts = line.split(' - ')
            if len(parts) >= 2:
                gt_class = parts[0]
                pred_class = parts[1]
                
                # 점수 정보가 있다면 제거 (예: "Keloid scar - Hypertrophic scar - [19.42326545715332, ...]")
                if '[' in pred_class:
                    pred_class = pred_class.split('[')[0].strip()
                
                class_data.append({
                    'gt_class': gt_class,
                    'pred_class': pred_class
                })
    
    print(f"클래스 데이터: {len(class_data)}개 항목 처리됨")
    return class_data

def parse_tagging_data(file_path, start_line=0, end_line=None):
    """태깅 정보 파일 파싱"""
    print(f"태깅 파일 로드 중: {file_path} (줄 {start_line}부터 {end_line if end_line else '끝'}까지)")
    tagging_data = []
    
    # 특정 범위의 줄만 읽기
    lines = read_lines_in_range(file_path, start_line, end_line)
    
    for line in lines:
        if ' - ' in line:
            parts = line.split(' - ')
            if len(parts) == 2:
                gt_tags = parts[0].split(',')
                pred_tags = parts[1].split(',')
                
                # 태그 정보 추출 및 정리
                gt_tags = [tag.strip() for tag in gt_tags]
                pred_tags = [tag.strip() for tag in pred_tags]
                
                # 태그 그룹화
                gt_tag_dict = categorize_tags(gt_tags)
                pred_tag_dict = categorize_tags(pred_tags)
                
                tagging_data.append({
                    'gt_tags': gt_tag_dict,
                    'pred_tags': pred_tag_dict,
                    'raw_gt_tags': gt_tags,
                    'raw_pred_tags': pred_tags
                })
    
    print(f"태깅 데이터: {len(tagging_data)}개 항목 처리됨")
    return tagging_data

def categorize_tags(tags):
    """태그를 카테고리별로 분류"""
    result = {
        'Width': None,
        'Color': None,
        'Pigmentation': None,
        'Surface': None,
        'Irregular_Color': None,
        'Irregular_Height': None
    }
    
    tag_categories = {
        'Width': ['Linear Width', 'Widened Width', 'Linear bulging Width'],
        'Color': ['Normal Color', 'Pink Color', 'Red Color', 'Purple Color'],
        'Pigmentation': ['Normal Pigmentation', 'Pigmented Pigmentation', 'Hypopigmented Pigmentation'],
        'Surface': ['Flat Surface', 'Hypertrophic Surface', 'Keloid Surface', 'Atrophic Surface'],
        'Irregular_Color': ['no Irregular Color', 'mild Irregular Color', 'moderate Irregular Color', 'severe Irregular Color'],
        'Irregular_Height': ['no Irregular Height', 'mild Irregular Height', 'moderate Irregular Height', 'severe Irregular Height']
    }
    
    for tag in tags:
        for category, category_tags in tag_categories.items():
            if tag in category_tags:
                result[category] = tag
                break
    
    return result

def merge_data(class_data, tagging_data):
    """클래스와 태깅 데이터 병합"""
    merged_data = []
    
    # 두 데이터의 길이가 같은지 확인
    min_len = min(len(class_data), len(tagging_data))
    print(f"병합할 데이터: {min_len}개 항목")
    
    for i in range(min_len):
        merged_item = {
            'gt_class': class_data[i]['gt_class'],
            'pred_class': class_data[i]['pred_class'],
            'gt_tags': tagging_data[i]['gt_tags'],
            'pred_tags': tagging_data[i]['pred_tags'],
            'raw_gt_tags': tagging_data[i]['raw_gt_tags'],
            'raw_pred_tags': tagging_data[i]['raw_pred_tags']
        }
        merged_data.append(merged_item)
    
    return merged_data

def create_stacked_bar_chart(merged_data, output_dir, prefix=''):
    """태그별 흉터 클래스 분포를 스택 막대 차트로 시각화"""
    # 정의된 클래스 목록
    scar_classes = ['Hypertrophic scar', 'Keloid scar', 'Others']
    
    # seaborn Paired 팔레트 설정 (커스텀 색상 사용)
    import seaborn as sns
    paired_palette = sns.color_palette("Paired")
    gt_colors = [paired_palette[0], paired_palette[2], paired_palette[6]]  # GT 색상: 0, 2, 6번째 색
    pred_colors = [paired_palette[1], paired_palette[3], paired_palette[7]]  # Pred 색상: 1, 3, 7번째 색
    
    # 태그 카테고리 및 값 목록
    tag_categories = {
        'Width': ['Linear Width', 'Widened Width', 'Linear bulging Width'],
        'Color': ['Normal Color', 'Pink Color', 'Red Color', 'Purple Color'],
        'Pigmentation': ['Normal Pigmentation', 'Pigmented Pigmentation', 'Hypopigmented Pigmentation'],
        'Surface': ['Flat Surface', 'Hypertrophic Surface', 'Keloid Surface', 'Atrophic Surface'],
        'Irregular_Color': ['no Irregular Color', 'mild Irregular Color', 'moderate Irregular Color', 'severe Irregular Color'],
        'Irregular_Height': ['no Irregular Height', 'mild Irregular Height', 'moderate Irregular Height', 'severe Irregular Height']
    }
    
    # 결과 파일 경로 목록
    result_files = []
    
    # 긴 태그 이름에 줄바꿈 추가하는 함수
    def wrap_tag_name(tag_name):
        # 카테고리 이름과 태그 이름 분리
        if ":" in tag_name:
            category, tag = tag_name.split(":", 1)
        else:
            category, tag = "", tag_name
        
        # 일정 길이 이상이면 두 줄로 나누기
        if len(tag) > 12 and ' ' in tag:
            parts = tag.split(' ')
            if len(parts) >= 3:  # 세 단어 이상이면 두 번째 단어 뒤에서 줄바꿈
                return f"{category}:\n{parts[0]} {parts[1]}\n{' '.join(parts[2:])}"
            else:  # 두 단어면 단어 사이에 줄바꿈
                # 문제가 되는 부분 수정
                joined_parts = "\n".join(parts)
                return f"{category}:\n{joined_parts}"
        return f"{category}:\n{tag}"
    
    # 각 카테고리별로 GT와 Pred 막대 차트 생성
    for category, tag_values in tag_categories.items():
        # GT 분포 계산
        gt_distribution = {}
        pred_distribution = {}
        
        for tag_value in tag_values:
            gt_distribution[tag_value] = {cls: 0 for cls in scar_classes}
            pred_distribution[tag_value] = {cls: 0 for cls in scar_classes}
        
        # GT 클래스와 GT 태그 기준 분포 집계
        for item in merged_data:
            gt_class = item['gt_class']
            pred_class = item['pred_class']
            
            if gt_class in scar_classes:
                # GT 데이터 집계
                gt_tag_value = item['gt_tags'].get(category)
                if gt_tag_value in tag_values:
                    gt_distribution[gt_tag_value][gt_class] += 1
                
                # Pred 데이터 집계
                pred_tag_value = item['pred_tags'].get(category)
                if pred_tag_value in tag_values:
                    pred_distribution[pred_tag_value][pred_class] += 1
        
        # 모든 태그 값을 사용 (원래 순서 유지)
        # 데이터가 있는지 여부와 관계없이 모든 태그를 표시
        tag_values_to_display = tag_values
        
        if not any(sum(gt_distribution[v].values()) > 0 or sum(pred_distribution[v].values()) > 0 for v in tag_values_to_display):
            print(f"카테고리 {category}에 데이터가 없습니다. 빈 차트 생성.")
        
        # 시각화
        plt.figure(figsize=(14, 8))
        
        # 데이터 준비
        bar_width = 0.4
        indices = np.arange(len(tag_values_to_display))
        
        # GT 막대 (왼쪽)
        bottoms_gt = np.zeros(len(tag_values_to_display))
        for i, cls in enumerate(scar_classes):
            values = [gt_distribution[tag][cls] for tag in tag_values_to_display]
            plt.bar(indices - bar_width/2, values, bar_width, label=f'GT {cls}', 
                   bottom=bottoms_gt, color=gt_colors[i])
            bottoms_gt += values
        
        # Pred 막대 (오른쪽)
        bottoms_pred = np.zeros(len(tag_values_to_display))
        for i, cls in enumerate(scar_classes):
            values = [pred_distribution[tag][cls] for tag in tag_values_to_display]
            plt.bar(indices + bar_width/2, values, bar_width, label=f'Pred {cls}', 
                   bottom=bottoms_pred, color=pred_colors[i])
            bottoms_pred += values
        
        # 차트 레이블 및 축 설정
        plt.xlabel('feature', fontsize=12)
        plt.ylabel('number', fontsize=12)
        
        # 제목 없음 (요구사항에 따라 제거)
        
        # 긴 태그 이름 두 줄로 변경
        wrapped_labels = [wrap_tag_name(tag) for tag in tag_values_to_display]
        plt.xticks(indices, wrapped_labels, rotation=0, ha='center', fontsize=10)
        
        # y축 범위 설정 - 최소 0에서 시작하도록
        plt.ylim(bottom=0)
        
        # 만약 데이터가 모두 0이면 y축 최대값 설정
        if max(bottoms_gt) == 0 and max(bottoms_pred) == 0:
            plt.ylim(top=1)  # 빈 차트에 대한 적절한 y축 범위 설정
        
        # 범례 설정 
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # 파일명에 접두어 추가 (선택적 범위 표시용)
        if prefix:
            filename = f'{prefix}_scar_class_distribution_by_{category}.png'
        else:
            filename = f'scar_class_distribution_by_{category}.png'
            
        # 파일 저장
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"차트 저장됨: {output_path}")
        result_files.append(output_path)
        
        # 차트 닫기
        plt.close()
    
    return result_files

def create_combined_chart(merged_data, output_dir, prefix=''):
    """모든 태그 카테고리를 포함하는 하나의 통합 차트 생성"""
    print("통합 차트 생성 중...")
    
    # 정의된 클래스 목록
    scar_classes = ['Hypertrophic scar', 'Keloid scar', 'Others']
    
    # seaborn Paired 팔레트 설정 (커스텀 색상 사용)
    import seaborn as sns
    paired_palette = sns.color_palette("Paired")
    gt_colors = [paired_palette[0], paired_palette[2], paired_palette[6]]  # GT 색상: 0, 2, 6번째 색
    pred_colors = [paired_palette[1], paired_palette[3], paired_palette[7]]  # Pred 색상: 1, 3, 7번째 색
    
    # 태그 카테고리 및 값 목록
    tag_categories = {
        'Width': ['Linear Width', 'Widened Width', 'Linear bulging Width'],
        'Color': ['Normal Color', 'Pink Color', 'Red Color', 'Purple Color'],
        'Pigmentation': ['Normal Pigmentation', 'Pigmented Pigmentation', 'Hypopigmented Pigmentation'],
        'Surface': ['Flat Surface', 'Hypertrophic Surface', 'Keloid Surface', 'Atrophic Surface'],
        'Irregular_Color': ['no Irregular Color', 'mild Irregular Color', 'moderate Irregular Color', 'severe Irregular Color'],
        'Irregular_Height': ['no Irregular Height', 'mild Irregular Height', 'moderate Irregular Height', 'severe Irregular Height']
    }
    
    # 모든 태그 카테고리를 병합하여 하나의 그래프에 표시
    all_tags = []
    for category, tags in tag_categories.items():
        # 카테고리 이름을 태그에 추가하여 구분
        all_tags.extend([f"{category}:{tag}" for tag in tags])
    
    # 긴 태그 이름에 줄바꿈 추가하는 함수
    def wrap_tag_name(tag_name):
        # 카테고리 이름과 태그 이름 분리
        if ":" in tag_name:
            category, tag = tag_name.split(":", 1)
        else:
            category, tag = "", tag_name
        
        # 일정 길이 이상이면 두 줄로 나누기
        if len(tag) > 12 and ' ' in tag:
            parts = tag.split(' ')
            if len(parts) >= 3:  # 세 단어 이상이면 두 번째 단어 뒤에서 줄바꿈
                return f"{category}:\n{parts[0]} {parts[1]}\n{' '.join(parts[2:])}"
            else:  # 두 단어면 단어 사이에 줄바꿈
                # 문제가 되는 부분 수정
                joined_parts = "\n".join(parts)
                return f"{category}:\n{joined_parts}"
        return f"{category}:\n{tag}"
    
    # GT 분포 및 Pred 분포 초기화
    gt_distribution = {}
    pred_distribution = {}
    
    for tag in all_tags:
        category, tag_value = tag.split(":", 1)
        gt_distribution[tag] = {cls: 0 for cls in scar_classes}
        pred_distribution[tag] = {cls: 0 for cls in scar_classes}
    
    # 데이터 집계
    for item in merged_data:
        gt_class = item['gt_class']
        pred_class = item['pred_class']
        
        if gt_class in scar_classes:
            # 각 카테고리별로 GT 및 Pred 데이터 집계
            for category, tag_values in tag_categories.items():
                # GT 데이터 집계
                gt_tag_value = item['gt_tags'].get(category)
                if gt_tag_value in tag_values:
                    full_tag = f"{category}:{gt_tag_value}"
                    gt_distribution[full_tag][gt_class] += 1
                
                # Pred 데이터 집계
                pred_tag_value = item['pred_tags'].get(category)
                if pred_tag_value in tag_values:
                    full_tag = f"{category}:{pred_tag_value}"
                    pred_distribution[full_tag][pred_class] += 1
    
    # 시각화
    plt.figure(figsize=(24, 12))  # 가로로 긴 그래프
    
    # 데이터 준비
    bar_width = 0.4
    indices = np.arange(len(all_tags))
    
    # GT 막대 (왼쪽)
    bottoms_gt = np.zeros(len(all_tags))
    for i, cls in enumerate(scar_classes):
        values = [gt_distribution[tag][cls] for tag in all_tags]
        plt.bar(indices - bar_width/2, values, bar_width, label=f'GT {cls}', 
               bottom=bottoms_gt, color=gt_colors[i])
        bottoms_gt += values
    
    # Pred 막대 (오른쪽)
    bottoms_pred = np.zeros(len(all_tags))
    for i, cls in enumerate(scar_classes):
        values = [pred_distribution[tag][cls] for tag in all_tags]
        plt.bar(indices + bar_width/2, values, bar_width, label=f'Pred {cls}', 
               bottom=bottoms_pred, color=pred_colors[i])
        bottoms_pred += values
    
    # 차트 레이블 및 축 설정
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    
    # 제목 없음
    
    # 긴 태그 이름 두 줄로 변경
    wrapped_labels = [wrap_tag_name(tag) for tag in all_tags]
    plt.xticks(indices, wrapped_labels, rotation=45, ha='right', fontsize=8)
    
    # y축 범위 설정
    plt.ylim(bottom=0)
    
    # 범례 설정
    plt.legend(fontsize=12, loc='upper right')
    plt.tight_layout()
    
    # 파일 저장
    if prefix:
        filename = f'{prefix}_combined_scar_class_distribution.png'
    else:
        filename = 'combined_scar_class_distribution.png'
        
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"통합 차트 저장됨: {output_path}")
    
    # 차트 닫기
    plt.close()
    
    return output_path

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='특정 줄 범위의 데이터만 분석하는 흉터 분류 분석 도구')
    
    parser.add_argument('--tag-file', type=str, default=tagging_file,
                        help='태깅 파일 경로')
    parser.add_argument('--class-file', type=str, default=class_file,
                        help='클래스 파일 경로')
    parser.add_argument('--output-dir', type=str, default=output_dir,
                        help='출력 디렉터리 경로')
                        
    parser.add_argument('--tag-start', type=int, default=tag_start_line,
                        help='태깅 파일에서 시작할 줄 번호 (0부터 시작)')
    parser.add_argument('--tag-end', type=int, default=tag_end_line,
                        help='태깅 파일에서 끝나는 줄 번호 (None이면 파일 끝까지)')
    parser.add_argument('--class-start', type=int, default=class_start_line,
                        help='클래스 파일에서 시작할 줄 번호 (0부터 시작)')
    parser.add_argument('--class-end', type=int, default=class_end_line,
                        help='클래스 파일에서 끝나는 줄 번호 (None이면 파일 끝까지)')
    
    args = parser.parse_args()
    
    # None 값 처리 (argparse는 None을 직접 처리하지 못함)
    if args.tag_end == 0:
        args.tag_end = None
    if args.class_end == 0:
        args.class_end = None
    
    return args

def main():
    """메인 함수"""
    # 명령줄 인자 처리
    args = parse_args()
    
    # 인자에서 값 추출
    tag_file = args.tag_file
    class_file = args.class_file
    output_dir = args.output_dir
    tag_start = args.tag_start
    tag_end = args.tag_end
    class_start = args.class_start
    class_end = args.class_end
    
    # 파라미터 정보 출력
    print("특징별 흉터 종류 분포 분석 시작...")
    print(f"태깅 파일: {tag_file} (줄 {tag_start}부터 {tag_end if tag_end else '끝'}까지)")
    print(f"클래스 파일: {class_file} (줄 {class_start}부터 {class_end if class_end else '끝'}까지)")
    print(f"출력 디렉터리: {output_dir}")
    
    # 출력 디렉터리 생성 확인
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터 로드 (특정 범위만)
    class_data = parse_class_data(class_file, class_start, class_end)
    tagging_data = parse_tagging_data(tag_file, tag_start, tag_end)
    
    # 데이터 병합
    merged_data = merge_data(class_data, tagging_data)
    
    # 결과 파일명에 사용할 접두어 생성 (범위 정보 포함)
    prefix = f"lines_{tag_start}-{tag_end if tag_end else 'end'}"
    
    # 개별 스택 막대 차트 생성
    result_files = create_stacked_bar_chart(merged_data, output_dir, prefix)
    
    # 통합 차트 생성 (추가)
    combined_chart_path = create_combined_chart(merged_data, output_dir, prefix)
    
    # 요약 정보 출력
    print("분석 완료!")
    print(f"총 {len(result_files) + 1}개 파일이 생성됨: {output_dir}")
    
    # 요약 정보를 텍스트 파일로 저장
    summary_path = os.path.join(output_dir, f"{prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"흉터 분류 분석 요약\n")
        f.write(f"실행 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"태깅 파일: {tag_file} (줄 {tag_start}부터 {tag_end if tag_end else '끝'}까지)\n")
        f.write(f"클래스 파일: {class_file} (줄 {class_start}부터 {class_end if class_end else '끝'}까지)\n\n")
        f.write(f"처리된 클래스 데이터: {len(class_data)}개 항목\n")
        f.write(f"처리된 태깅 데이터: {len(tagging_data)}개 항목\n")
        f.write(f"병합된 데이터: {len(merged_data)}개 항목\n\n")
        f.write(f"생성된 파일:\n")
        f.write(f"1. {os.path.basename(combined_chart_path)} (통합 차트)\n")
        for i, file in enumerate(result_files, 2):
            f.write(f"{i}. {os.path.basename(file)}\n")
    
    print(f"요약 정보 저장됨: {summary_path}")

if __name__ == "__main__":
    main()