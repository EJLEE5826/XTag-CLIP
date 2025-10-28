import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime
import matplotlib.colors as mcolors

# 파일에서 특정 줄 범위만 읽어오는 함수
def read_lines_in_range(file_path, start_line=0, end_line=None):
    with open(file_path, 'r') as f:
        if end_line is None:
            # 끝 줄이 지정되지 않으면 전체 파일 읽기
            lines = f.readlines()[start_line:]
        else:
            # 시작 줄부터 끝 줄까지만 읽기
            lines = f.readlines()[start_line:end_line]
    return lines

# 결과 저장 디렉토리 생성
def create_output_directory():
    output_dir = f"./analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# 긴 태그 이름을 두 줄로 변경하는 함수
def wrap_tag_name(tag):
    # 태그가 공백을 포함하고 충분히 긴 경우
    if ' ' in tag and len(tag) > 12:
        words = tag.split()
        if len(words) >= 3:
            # 첫 단어와 두 번째 단어를 첫 줄에
            return f"{words[0]} {words[1]}\n{' '.join(words[2:])}"
        else:
            # 단어가 2개인 경우 중간에 줄바꿈
            return '\n'.join(words)
    return tag

# 태그를 심각성에 따라 정렬하는 함수
def sort_tags_by_severity(tags):
    # 심각성 순서 정의 (태그 유형별)
    severity_order = {
        # Width 관련 태그
        'Linear Width': 0, 
        'Linear bulging Width': 1, 
        'Widened Width': 2,
        
        # Color 관련 태그
        'Normal Color': 0, 
        'Pink Color': 1, 
        'Red Color': 2, 
        'Purple Color': 3,
        
        # Pigmentation 관련 태그
        'Normal Pigmentation': 0, 
        'Hypopigmented Pigmentation': 1, 
        'Pigmented Pigmentation': 2,
        
        # Surface 관련 태그
        'Flat Surface': 0, 
        'Atrophic Surface': 1, 
        'Hypertrophic Surface': 2, 
        'Keloid Surface': 3,
        
        # Irregular Color 관련 태그
        'no Irregular Color': 0, 
        'mild Irregular Color': 1, 
        'moderate Irregular Color': 2, 
        'severe Irregular Color': 3,
        
        # Irregular Height 관련 태그
        'no Irregular Height': 0, 
        'mild Irregular Height': 1, 
        'moderate Irregular Height': 2, 
        'severe Irregular Height': 3
    }
    
    # 태그를 심각성 순서대로 정렬
    sorted_tags = sorted(tags, key=lambda x: severity_order.get(x, 999))
    return sorted_tags

# 태그 그룹 정의
tag_groups = {
    'Width': ['Linear Width', 'Linear bulging Width', 'Widened Width'],
    'Color': ['Normal Color', 'Pink Color', 'Red Color', 'Purple Color'],
    'Pigmentation': ['Normal Pigmentation', 'Hypopigmented Pigmentation', 'Pigmented Pigmentation'],
    'Surface': ['Flat Surface', 'Atrophic Surface', 'Hypertrophic Surface', 'Keloid Surface'],
    'Irregular Color': ['no Irregular Color', 'mild Irregular Color', 'moderate Irregular Color', 'severe Irregular Color'],
    'Irregular Height': ['no Irregular Height', 'mild Irregular Height', 'moderate Irregular Height', 'severe Irregular Height']
}

# 심각성에 따라 태그 그룹 정렬
for group in tag_groups:
    tag_groups[group] = sort_tags_by_severity(tag_groups[group])

# 태그가 속한 그룹 찾기
def find_tag_group(tag):
    for group, tags in tag_groups.items():
        if tag in tags:
            return group
    return None

# 태그의 심각성 레벨 찾기 (같은 그룹 내에서)
def get_tag_severity_level(tag):
    group = find_tag_group(tag)
    if group:
        return tag_groups[group].index(tag)
    return 0

# 읽어올 줄 범위 설정
tag_start_line = 1803  # 태그 파일 시작 줄 (0부터 시작)
tag_end_line = 1899  # None으로 설정하면 파일 끝까지 읽음
class_start_line = 1667  # 클래스 파일 시작 줄 (0부터 시작)
class_end_line = 1763  # None으로 설정하면 파일 끝까지 읽음

# 출력 디렉토리 생성
output_dir = create_output_directory()

# 태그 데이터 파일 경로
tag_file_path = ''
# 클래스 데이터 파일 경로
class_file_path = ''

# 태그 데이터 파일 읽기
tag_lines = read_lines_in_range(tag_file_path, tag_start_line, tag_end_line)

# 클래스 데이터 파일 읽기 
class_lines = read_lines_in_range(class_file_path, class_start_line, class_end_line)

# 클래스 데이터 파싱하여 GT와 예측 레이블 추출
class_data = []
for line in class_lines:
    line = line.strip()
    # 전체 정확도와 같은 메트릭 라인은 건너뛰기
    if line.startswith("전체 정확도") or line.startswith("그룹별 메트릭") or line.startswith("val data"):
        continue
        
    # "실제 클래스 - 예측 클래스" 형식 파싱
    parts = line.split(' - ')
    if len(parts) >= 2:
        gt = parts[0].strip()
        pred = parts[1].strip()
        # 점수 부분은 무시
        class_data.append({'gt': gt, 'pred': pred})

# 클래스 레이블 고유값 추출
unique_classes = set()
for item in class_data:
    unique_classes.add(item['gt'])
    unique_classes.add(item['pred'])
unique_classes = list(unique_classes)

# 태그 데이터 파싱
tag_data = []
current_tags = None

for line in tag_lines:
    line = line.strip()
    
    # 메트릭 정보 건너뛰기
    if line.startswith("전체 정확도") or line.startswith("그룹별 메트릭") or line.startswith("val data"):
        continue
    
    # 태그 파싱
    if ' - ' in line:
        parts = line.split(' - ')
        if len(parts) == 2:
            gt_tags_str = parts[0].strip()
            pred_tags_str = parts[1].strip()
            
            # 쉼표로 구분된 태그를 리스트로 변환
            gt_tags = [tag.strip() for tag in gt_tags_str.split(',')]
            pred_tags = [tag.strip() for tag in pred_tags_str.split(',')]
            
            # 태그 쌍 저장
            tag_data.append({'gt_tags': gt_tags, 'pred_tags': pred_tags})

# 데이터 일관성 확인
print(f"클래스 데이터 수: {len(class_data)}")
print(f"태그 데이터 수: {len(tag_data)}")

# 태그 빈도 계산을 위한 함수
def count_tag_frequencies(tags_list):
    tag_counts = {}
    for tags in tags_list:
        for tag in tags:
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1
    return tag_counts

# 각 클래스별 태그 빈도 계산
class_tag_frequencies = {}
for cls in unique_classes:
    class_tag_frequencies[cls] = {'gt': {}, 'pred': {}}

# 클래스 데이터와 태그 데이터 결합 (같은 수의 항목이 있을 경우에만)
min_len = min(len(class_data), len(tag_data))
if min_len > 0:
    for i in range(min_len):
        gt_class = class_data[i]['gt']
        pred_class = class_data[i]['pred']
        
        # GT 클래스의 GT 태그 빈도 업데이트
        for tag in tag_data[i]['gt_tags']:
            if tag in class_tag_frequencies[gt_class]['gt']:
                class_tag_frequencies[gt_class]['gt'][tag] += 1
            else:
                class_tag_frequencies[gt_class]['gt'][tag] = 1
        
        # 예측 클래스의 예측 태그 빈도 업데이트
        for tag in tag_data[i]['pred_tags']:
            if tag in class_tag_frequencies[pred_class]['pred']:
                class_tag_frequencies[pred_class]['pred'][tag] += 1
            else:
                class_tag_frequencies[pred_class]['pred'][tag] = 1
else:
    print("경고: 클래스 데이터와 태그 데이터의 수가 일치하지 않습니다.")

# 가장 흔한 태그 N개 추출
def get_top_tags(tag_dict, n=5):
    sorted_tags = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_tags[:n]

# 분석 결과를 텍스트 파일로 저장
result_file_path = os.path.join(output_dir, "analysis_results.txt")
with open(result_file_path, "w") as result_file:
    result_file.write(f"데이터 분석 결과\n")
    result_file.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    result_file.write(f"태그 파일: {tag_file_path} (줄 {tag_start_line}부터 {tag_end_line if tag_end_line else '끝'}까지)\n")
    result_file.write(f"클래스 파일: {class_file_path} (줄 {class_start_line}부터 {class_end_line if class_end_line else '끝'}까지)\n\n")
    
    result_file.write(f"총 분석된 데이터 개수: {min_len}\n")
    result_file.write(f"발견된 고유 클래스: {len(unique_classes)}\n")
    result_file.write(f"고유 클래스 목록: {', '.join(unique_classes)}\n\n")
    
    for cls in unique_classes:
        result_file.write(f"\n--- 클래스: {cls} ---\n")
        
        # GT 태그 상위 10개
        gt_top_tags = get_top_tags(class_tag_frequencies[cls]['gt'], 10)
        result_file.write("GT 태그 상위 10개:\n")
        for i, (tag, count) in enumerate(gt_top_tags, 1):
            result_file.write(f"  {i}. {tag}: {count}회\n")
        
        # 예측 태그 상위 10개
        pred_top_tags = get_top_tags(class_tag_frequencies[cls]['pred'], 10)
        result_file.write("예측 태그 상위 10개:\n")
        for i, (tag, count) in enumerate(pred_top_tags, 1):
            result_file.write(f"  {i}. {tag}: {count}회\n")

# tab20c 색상 팔레트 설정
# GT: 파란색(3~0), pred: 초록색(11~8), 심각성에 따라 색이 진해지도록
blue_colors = plt.cm.tab20c(np.array([3, 2, 1, 0]))  # 파란색 계열 (짙은 색 -> 연한 색)
green_colors = plt.cm.tab20c(np.array([11, 10, 9, 8]))  # 초록색 계열 (짙은 색 -> 연한 색)

# 클래스별 개별 그래프 저장 (심각성에 따라 색상 적용)
for cls in unique_classes:
    if cls in class_tag_frequencies:
        plt.figure(figsize=(12, 4))
        
        # 모든 태그 그룹의 모든 태그를 포함하는 전체 태그 목록 생성
        all_possible_tags = []
        for group, tags in tag_groups.items():
            all_possible_tags.extend(tags)
        
        # 전체 태그 목록을 그룹과 심각성에 따라 정렬
        all_possible_tags = sorted(all_possible_tags, key=lambda x: (find_tag_group(x) or "", get_tag_severity_level(x)))
        
        # x 위치 설정
        x = np.arange(len(all_possible_tags))
        width = 0.35
        
        # 각 태그별 심각성 레벨에 따른 색상 인덱스 결정
        gt_colors = []
        pred_colors = []
        
        for tag in all_possible_tags:
            severity = min(get_tag_severity_level(tag), 3)  # 최대 4단계 색상 (0,1,2,3)
            gt_colors.append(blue_colors[severity])
            pred_colors.append(green_colors[severity])
        
        # GT 태그 빈도 바 그래프 (데이터가 없는 태그는 0으로 설정)
        gt_values = [class_tag_frequencies[cls]['gt'].get(tag, 0) for tag in all_possible_tags]
        gt_bars = plt.bar(x - width/2, gt_values, width, label='Ground Truth', color=gt_colors)
        
        # 예측 태그 빈도 바 그래프 (데이터가 없는 태그는 0으로 설정)
        pred_values = [class_tag_frequencies[cls]['pred'].get(tag, 0) for tag in all_possible_tags]
        pred_bars = plt.bar(x + width/2, pred_values, width, label='Prediction', color=pred_colors)
        
        plt.xlabel('Tags', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.title(cls, fontsize=16)  # 클래스 이름만 제목으로
        
        # 긴 태그 이름 두 줄로 변경
        wrapped_labels = [wrap_tag_name(tag) for tag in all_possible_tags]
        plt.xticks(x, wrapped_labels, rotation=45, ha='right', fontsize=10)  # 폰트 크기 줄임 (많은 태그가 표시되므로)
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # 개별 클래스 그래프 저장
        class_plot_path = os.path.join(output_dir, f"tag_distribution_{cls.replace(' ', '_')}.png")
        plt.savefig(class_plot_path, dpi=300, bbox_inches='tight')
        plt.close()

# 전체 클래스 통합 그래프 생성 (클래스별로 서브플롯)
fig, axs = plt.subplots(len(unique_classes), 1, figsize=(14, 6*len(unique_classes)))

for i, cls in enumerate(unique_classes):
    if cls in class_tag_frequencies:
        # 현재 서브플롯
        if len(unique_classes) > 1:
            ax = axs[i]
        else:
            ax = axs  # 클래스가 하나뿐인 경우
        
        # GT 태그 중 상위 10개
        gt_top_tags = get_top_tags(class_tag_frequencies[cls]['gt'], 10)
        
        # 예측 태그 중 상위 10개
        pred_top_tags = get_top_tags(class_tag_frequencies[cls]['pred'], 10)
        
        # 두 리스트를 합치고 중복 제거
        all_top_tags = []
        for tag, _ in gt_top_tags[:5]:  # 상위 5개만
            all_top_tags.append(tag)
        for tag, _ in pred_top_tags[:5]:  # 상위 5개만
            if tag not in all_top_tags:
                all_top_tags.append(tag)
        
        # 태그 그룹별로 정렬
        all_top_tags = sorted(all_top_tags, key=lambda x: (find_tag_group(x) or "", get_tag_severity_level(x)))
        
        # x 위치 설정
        x = np.arange(len(all_top_tags))
        width = 0.35
        
        # 각 태그별 심각성 레벨에 따른 색상 인덱스 결정
        gt_colors = []
        pred_colors = []
        
        for tag in all_top_tags:
            severity = min(get_tag_severity_level(tag), 3)  # 최대 4단계 색상 (0,1,2,3)
            gt_colors.append(blue_colors[severity])
            pred_colors.append(green_colors[severity])
        
        # GT 태그 빈도 바 그래프
        gt_values = [class_tag_frequencies[cls]['gt'].get(tag, 0) for tag in all_top_tags]
        gt_bars = ax.bar(x - width/2, gt_values, width, label='Ground Truth', color=gt_colors)
        
        # 예측 태그 빈도 바 그래프
        pred_values = [class_tag_frequencies[cls]['pred'].get(tag, 0) for tag in all_top_tags]
        pred_bars = ax.bar(x + width/2, pred_values, width, label='Prediction', color=pred_colors)
        
        ax.set_xlabel('Tags', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        ax.set_title(cls, fontsize=14)  # 클래스 이름만 제목으로
        
        # 긴 태그 이름 두 줄로 변경
        wrapped_labels = [wrap_tag_name(tag) for tag in all_top_tags]
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_labels, rotation=45, ha='right', fontsize=10)
        
        ax.legend(fontsize=10)

plt.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# 통합 그래프 저장
combined_plot_path = os.path.join(output_dir, "combined_tag_distribution.png")
plt.savefig(combined_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"분석 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
print(f"- 텍스트 결과: {result_file_path}")
print(f"- 통합 그래프: {combined_plot_path}")
print(f"- 클래스별 그래프: {len(unique_classes)}개 파일이 저장됨")


tag_frequencies = {}
for group, tags in tag_groups.items():
    for tag in tags:
        tag_frequencies[tag] = {'gt': 0, 'pred': 0}

# 태그 빈도 계산
for i in range(min_len):
    # GT 태그 빈도 업데이트
    for tag in tag_data[i]['gt_tags']:
        if tag in tag_frequencies:
            tag_frequencies[tag]['gt'] += 1
    
    # Pred 태그 빈도 업데이트
    for tag in tag_data[i]['pred_tags']:
        if tag in tag_frequencies:
            tag_frequencies[tag]['pred'] += 1

# 태그별 정규화된 막대 그래프 생성
plt.figure(figsize=(18, 10))

# 태그 목록을 그룹과 심각성에 따라 정렬
all_tags = sorted(list(tag_frequencies.keys()), 
                 key=lambda x: (find_tag_group(x) or "", get_tag_severity_level(x)))

# x 위치 설정
x = np.arange(len(all_tags))
width = 0.4

# GT와 Pred 빈도 추출
gt_values = [tag_frequencies[tag]['gt'] for tag in all_tags]
pred_values = [tag_frequencies[tag]['pred'] for tag in all_tags]

# 태그 그룹별 색상 설정
tag_colors = []
for tag in all_tags:
    group = find_tag_group(tag)
    severity = min(get_tag_severity_level(tag), 3)
    if group:
        # 태그 그룹별로 다른 색상 사용
        group_idx = list(tag_groups.keys()).index(group) % 10
        color = plt.cm.tab10(group_idx)
        tag_colors.append(color)
    else:
        tag_colors.append(plt.cm.tab10(9))  # 기본 색상

# GT와 Pred 막대 그리기
gt_bars = plt.bar(x - width/2, gt_values, width, color=tag_colors, alpha=0.7, label='Ground Truth')
pred_bars = plt.bar(x + width/2, pred_values, width, color=tag_colors, hatch='///', alpha=0.7, label='Prediction')

# 축 설정 및 레이블
plt.xlabel('Tags', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Tag Distribution (Ground Truth vs Prediction)', fontsize=16)

# 태그 이름 및 그룹 경계 표시
wrapped_labels = [wrap_tag_name(tag) for tag in all_tags]
plt.xticks(x, wrapped_labels, rotation=45, ha='right', fontsize=9)

# 태그 그룹 경계 및 이름 표시
group_boundaries = []
current_group = None
for i, tag in enumerate(all_tags):
    group = find_tag_group(tag)
    if current_group != group:
        group_boundaries.append(i)
        current_group = group

# 태그 그룹 경계에 수직선 추가
for boundary in group_boundaries[1:]:  # 첫 번째 경계는 건너뜀
    plt.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.3)

# 그룹 이름 표시
group_names = list(tag_groups.keys())
group_positions = []
for i in range(len(group_boundaries)):
    start = group_boundaries[i]
    end = group_boundaries[i+1] if i+1 < len(group_boundaries) else len(all_tags)
    mid = (start + end - 1) / 2
    group_positions.append(mid)
    
    if i < len(group_names):
        plt.text(mid, -max(gt_values + pred_values) * 0.08, group_names[i], 
                 ha='center', fontsize=12, fontweight='bold')

# 범례 설정
plt.legend(loc='upper right', fontsize=12, ncol=2)

# 그리드 추가
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 레이아웃 조정
plt.tight_layout()
plt.subplots_adjust(bottom=0.2)  # 그룹 이름을 위한 공간 확보

# 저장
tag_distribution_path = os.path.join(output_dir, "tag_distribution.png")
plt.savefig(tag_distribution_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"- 태그 분포 그래프: {tag_distribution_path}")

# 태그 그룹별로 하나의 막대그래프로 누적 시각화
plt.figure(figsize=(14, 10))

# 태그 그룹 준비 (순서대로)
ordered_groups = list(tag_groups.keys())

# X 위치 설정
x = np.arange(len(ordered_groups))
width = 0.4

# 각 그룹별 태그 데이터 준비
gt_stacked_data = []
pred_stacked_data = []
all_tag_labels = []
all_tag_colors = []

# 각 그룹별 최대 태그 개수 확인
max_tags_in_group = max(len(tags) for tags in tag_groups.values())

# 서로 다른 태그 패턴에 대한 색상 및 패턴 준비
color_map = plt.cm.viridis  # 연속적인 색상 맵 사용
colors = [color_map(i/max_tags_in_group) for i in range(max_tags_in_group)]

# 각 태그 그룹별로 누적 데이터 준비
for group_idx, group in enumerate(ordered_groups):
    tags_in_group = sort_tags_by_severity(tag_groups[group])
    
    # 이 그룹의 태그별 GT/Pred 데이터
    group_gt_data = []
    group_pred_data = []
    group_labels = []
    group_colors = []
    
    # 각 태그의 값 수집
    for i, tag in enumerate(tags_in_group):
        if tag in tag_frequencies:
            group_gt_data.append(tag_frequencies[tag]['gt'])
            group_pred_data.append(tag_frequencies[tag]['pred'])
            group_labels.append(tag)
            group_colors.append(colors[i])
    
    # 데이터 저장
    gt_stacked_data.append(group_gt_data)
    pred_stacked_data.append(group_pred_data)
    all_tag_labels.extend(group_labels)
    all_tag_colors.extend(group_colors)

# GT 누적 막대 그리기
gt_bottoms = np.zeros(len(ordered_groups))
pred_bottoms = np.zeros(len(ordered_groups))

# 각 태그에 대한 핸들 저장 (범례용)
tag_handles = []
tag_labels = []

# 모든 그룹의 모든 태그를 한번에 처리
all_tags_to_plot = set()
for group in ordered_groups:
    all_tags_to_plot.update(sort_tags_by_severity(tag_groups[group]))

# 태그 심각성별로 정렬
all_tags_sorted = sort_tags_by_severity(list(all_tags_to_plot))

# 각 태그를 심각성 순서대로 그리기
for tag in all_tags_sorted:
    gt_values = []
    pred_values = []
    
    # 각 그룹에서 해당 태그의 값 찾기
    for group_idx, group in enumerate(ordered_groups):
        if tag in tag_groups[group]:
            gt_val = tag_frequencies[tag]['gt']
            pred_val = tag_frequencies[tag]['pred']
            gt_values.append(gt_val)
            pred_values.append(pred_val)
        else:
            gt_values.append(0)
            pred_values.append(0)
    
    # 태그의 심각성 레벨에 따라 색상 선택
    severity = min(get_tag_severity_level(tag), 3)
    tag_color = plt.cm.viridis(severity / 3)
    
    # GT 막대 그리기
    gt_bar = plt.bar(x - width/2, gt_values, width, bottom=gt_bottoms, 
              color=tag_color, alpha=0.7)
    
    # Pred 막대 그리기
    pred_bar = plt.bar(x + width/2, pred_values, width, bottom=pred_bottoms, 
                color=tag_color, hatch='///', alpha=0.7)
    
    # 범례 항목 추가 (각 태그마다 한 번만)
    tag_handles.append(gt_bar)
    shortened_tag = tag.replace(' ', '\n')
    tag_labels.append(shortened_tag)
    
    # 다음 막대를 위한 바닥값 업데이트
    gt_bottoms += np.array(gt_values)
    pred_bottoms += np.array(pred_values)

# 축 설정
plt.xlabel('Feature Groups', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Tag Distribution by Feature Groups', fontsize=16)

# x축 레이블
plt.xticks(x, ordered_groups, fontsize=12)

# 왼쪽 막대는 GT, 오른쪽 막대는 Pred 표시
gt_label = plt.bar(0, 0, color='gray', label='Ground Truth')
pred_label = plt.bar(0, 0, color='gray', hatch='///', label='Prediction')

# 범례 두 부분으로 나누기
# 1. GT vs Pred 범례 (위)
plt.legend(handles=[gt_label, pred_label], labels=['Ground Truth', 'Prediction'],
          loc='upper right', fontsize=12, ncol=2)

# 2. 태그별 범례 (아래)
second_legend = plt.legend(handles=tag_handles, labels=tag_labels,
                          loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fontsize=9, ncol=min(8, len(tag_handles)))

# 첫 번째 범례 추가 (다시)
plt.gca().add_artist(second_legend)
plt.gca().add_artist(plt.legend(handles=[gt_label, pred_label], 
                              labels=['Ground Truth', 'Prediction'],
                              loc='upper right', fontsize=12, ncol=2))

# 그리드 추가
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 레이아웃 조정
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # 태그 범례를 위한 공간 확보

# 저장
group_stacked_path = os.path.join(output_dir, "tag_group_stacked_distribution.png")
plt.savefig(group_stacked_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"- 태그 그룹별 누적 분포 그래프: {group_stacked_path}")


# 태그 그룹별로 하나의 막대그래프로 누적 시각화
plt.figure(figsize=(14, 10))

# 태그 그룹 준비 (순서대로)
ordered_groups = list(tag_groups.keys())

# X 위치 설정
x = np.arange(len(ordered_groups))
width = 0.4

# tab20c 색상 팔레트 설정
# GT: 파란색(3~0), pred: 초록색(11~8), 심각성에 따라 색이 진해지도록
blue_colors = plt.cm.tab20c(np.array([3, 2, 1, 0]))  # 파란색 계열 (짙은 색 -> 연한 색)
green_colors = plt.cm.tab20c(np.array([11, 10, 9, 8]))  # 초록색 계열 (짙은 색 -> 연한 색)

# GT 누적 막대 그리기
gt_bottoms = np.zeros(len(ordered_groups))
pred_bottoms = np.zeros(len(ordered_groups))

# 각 태그에 대한 핸들 저장 (범례용)
tag_handles = []
tag_labels = []

# 각 그룹의 태그를 역순으로 처리하여 심각성이 높은 태그가 하단에 오도록 함
for group in ordered_groups:
    # 태그를 심각성 순으로 정렬 (낮은 심각성 -> 높은 심각성)
    tags_in_group = sort_tags_by_severity(tag_groups[group])
    
    # 그래프 그릴 때는 역순으로 처리 (높은 심각성부터 그려서 아래에 위치하도록)
    for i, tag in enumerate(reversed(tags_in_group)):
        gt_values = []
        pred_values = []
        
        # 각 그룹에 해당 태그가 속하는지 확인하고 값 설정
        for group_idx, curr_group in enumerate(ordered_groups):
            if curr_group == group:
                gt_val = tag_frequencies[tag]['gt']
                pred_val = tag_frequencies[tag]['pred']
                gt_values.append(gt_val)
                pred_values.append(pred_val)
            else:
                gt_values.append(0)
                pred_values.append(0)
        
        # 심각성에 따른 색상 인덱스 선택
        severity_idx = min(get_tag_severity_level(tag), 3)  # 최대 4단계 색상 (0,1,2,3)
        
        # GT 색상: 심각성이 높을수록 진한 파란색 (3->0)
        gt_color = blue_colors[severity_idx]
        
        # Pred 색상: 심각성이 높을수록 진한 초록색 (11->8)
        pred_color = green_colors[severity_idx]
        
        # GT 막대 그리기
        gt_bar = plt.bar(x - width/2, gt_values, width, bottom=gt_bottoms, 
                  color=gt_color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Pred 막대 그리기
        pred_bar = plt.bar(x + width/2, pred_values, width, bottom=pred_bottoms, 
                    color=pred_color, alpha=0.85, edgecolor='black', linewidth=0.5,
                    hatch='///')
        
        # 범례 항목 추가 (각 태그마다 한 번만)
        # GT 색상으로 범례 추가
        tag_handle = plt.Rectangle((0, 0), 1, 1, color=gt_color, alpha=0.85, 
                                   linewidth=0.5)
        tag_handles.append(tag_handle)
        
        # 줄바꿈을 활용하여 긴 태그 이름 처리
        tag_labels.append(wrap_tag_name(tag))
        
        # 다음 막대를 위한 바닥값 업데이트
        gt_bottoms += np.array(gt_values)
        pred_bottoms += np.array(pred_values)

# 축 설정
plt.xlabel('Feature Groups', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.title('Tag Distribution by Feature Groups', fontsize=16)

# x축 레이블
plt.xticks(x, ordered_groups, fontsize=12)

# GT와 Pred 구분을 위한 범례 항목
gt_label = plt.Rectangle((0, 0), 1, 1, color='#4C72B0', label='Ground Truth')
pred_label = plt.Rectangle((0, 0), 1, 1, color='#55A868', label='Prediction', hatch='///')

# 범례 두 부분으로 나누기
# 1. GT vs Pred 범례 (상단)
first_legend = plt.legend(handles=[gt_label, pred_label], 
                         labels=['Ground Truth', 'Prediction'],
                         loc='upper right', fontsize=12, ncol=2)

# 그리드 추가
plt.grid(axis='y', linestyle='--', alpha=0.3)

# 레이아웃 조정
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # 태그 범례를 위한 공간 확보

# 저장
group_stacked_path = os.path.join(output_dir, "tag_group_stacked_distribution_20c.png")
plt.savefig(group_stacked_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"- 태그 그룹별 누적 분포 그래프 (탭 20c 색상 적용): {group_stacked_path}")



# 태그 그룹별로 하나의 막대그래프로 누적 분포 시각화 (정규화 버전)
plt.figure(figsize=(14, 10))

# 태그 그룹 준비 (순서대로)
ordered_groups = list(tag_groups.keys())

# X 위치 설정
x = np.arange(len(ordered_groups))
width = 0.4

# tab20c 색상 팔레트 설정
# GT: 파란색(3~0), pred: 초록색(11~8), 심각성에 따라 색이 진해지도록
blue_colors = plt.cm.tab20c(np.array([3, 2, 1, 0]))  # 파란색 계열 (짙은 색 -> 연한 색)
green_colors = plt.cm.tab20c(np.array([11, 10, 9, 8]))  # 초록색 계열 (짙은 색 -> 연한 색)

# 그룹별 총합 계산 (정규화 위함)
group_totals_gt = {group: 0 for group in ordered_groups}
group_totals_pred = {group: 0 for group in ordered_groups}

# 각 그룹의 총 빈도수 계산
for group in ordered_groups:
    for tag in tag_groups[group]:
        if tag in tag_frequencies:
            group_totals_gt[group] += tag_frequencies[tag]['gt']
            group_totals_pred[group] += tag_frequencies[tag]['pred']

# GT와 Pred 누적값 초기화
gt_bottoms = np.zeros(len(ordered_groups))
pred_bottoms = np.zeros(len(ordered_groups))

# 각 태그에 대한 핸들 저장 (범례용)
tag_handles = []
tag_labels = []

# 각 그룹의 태그를 역순으로 처리하여 심각성이 높은 태그가 하단에 오도록 함
for group in ordered_groups:
    # 태그를 심각성 순으로 정렬 (낮은 심각성 -> 높은 심각성)
    tags_in_group = sort_tags_by_severity(tag_groups[group])
    
    # 그래프 그릴 때는 역순으로 처리 (높은 심각성부터 그려서 아래에 위치하도록)
    for i, tag in enumerate(reversed(tags_in_group)):
        gt_values = []
        pred_values = []
        
        # 각 그룹에 해당 태그가 속하는지 확인하고 값 설정
        for group_idx, curr_group in enumerate(ordered_groups):
            if curr_group == group:
                # 정규화된 값 계산 (총합이 0이면 0으로 처리)
                gt_val = tag_frequencies[tag]['gt'] / group_totals_gt[curr_group] if group_totals_gt[curr_group] > 0 else 0
                pred_val = tag_frequencies[tag]['pred'] / group_totals_pred[curr_group] if group_totals_pred[curr_group] > 0 else 0
                gt_values.append(gt_val)
                pred_values.append(pred_val)
            else:
                gt_values.append(0)
                pred_values.append(0)
        
        # 심각성에 따른 색상 인덱스 선택
        severity_idx = min(get_tag_severity_level(tag), 3)  # 최대 4단계 색상 (0,1,2,3)
        
        # GT 색상: 심각성이 높을수록 진한 파란색 (3->0)
        gt_color = blue_colors[severity_idx]
        
        # Pred 색상: 심각성이 높을수록 진한 초록색 (11->8)
        pred_color = green_colors[severity_idx]
        
        # GT 막대 그리기
        gt_bar = plt.bar(x - width/2, gt_values, width, bottom=gt_bottoms, 
                  facecolor=gt_color, alpha=0.85, linewidth=0.5)
        
        # Pred 막대 그리기
        pred_bar = plt.bar(x + width/2, pred_values, width, bottom=pred_bottoms, 
                    facecolor=pred_color, alpha=0.85, linewidth=0.5)
        
        # 범례 항목 추가 (각 태그마다 한 번만)
        # GT 색상으로 범례 추가
        tag_handle = plt.Rectangle((0, 0), 1, 1, facecolor=gt_color, alpha=1, 
                                   edgecolor='black', linewidth=0.5)
        tag_handles.append(tag_handle)
        
        # 줄바꿈을 활용하여 긴 태그 이름 처리
        tag_labels.append(wrap_tag_name(tag))
        
        # 다음 막대를 위한 바닥값 업데이트
        gt_bottoms += np.array(gt_values)
        pred_bottoms += np.array(pred_values)

# 축 설정
plt.xlabel('', fontsize=14, labelpad=15)
plt.ylabel('Distribution', fontsize=14, labelpad=15)  # 빈도(Frequency) 대신 비율(Proportion)로 레이블 변경
plt.title('', fontsize=16)

# x축 레이블
plt.xticks(x, ordered_groups, fontsize=12)

# y축 범위 설정 (0~1)
plt.ylim(0, 1.0)

# GT와 Pred 구분을 위한 범례 항목
gt_label = plt.Rectangle((0, 0), 1, 1, facecolor=blue_colors[3], edgecolor='black', 
                        label='Ground Truth')
pred_label = plt.Rectangle((0, 0), 1, 1, facecolor=green_colors[3], edgecolor='black', 
                          label='Prediction')

# 범례 두 부분으로 나누기
# 1. GT vs Pred 범례 (상단)
first_legend = plt.legend(handles=[gt_label, pred_label], 
                         labels=['Ground Truth', 'Prediction'],
                         loc='upper right', fontsize=12, ncol=2)


# 그리드 추가


# 레이아웃 조정
plt.tight_layout()
plt.subplots_adjust(bottom=0.3)  # 태그 범례를 위한 공간 확보

# 저장
normalized_stacked_path = os.path.join(output_dir, "normalized_tag_group_distribution.png")
plt.savefig(normalized_stacked_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"- 정규화된 태그 그룹별 누적 분포 그래프 (탭 20c 색상, 01 적용): {normalized_stacked_path}")



# 태그 그룹별로 하나의 막대그래프로 누적 분포 시각화 (정규화 버전) - 태그 레이블 추가
plt.figure(figsize=(18, 10))

# 태그 그룹 준비 (순서대로)
ordered_groups = list(tag_groups.keys())

# X 위치 설정
x = np.arange(len(ordered_groups))
width = 0.425  # 기존 0.4에서 0.35로 조정 (필요에 따라 변경 가능)
gap = 0.00    # 막대 사이 간격 명시적 설정
bar_distance = width + gap  # GT와 Pred 막대 사이 거리

# tab20c 색상 팔레트 설정
# GT: 파란색(3~0), pred: 초록색(11~8), 심각성에 따라 색이 진해지도록
blue_colors = plt.cm.tab20c(np.array([3, 2, 1, 0]))  # 파란색 계열 (짙은 색 -> 연한 색)
green_colors = plt.cm.tab20c(np.array([11, 10, 9, 8]))  # 초록색 계열 (짙은 색 -> 연한 색)

# 그룹별 총합 계산 (정규화 위함)
group_totals_gt = {group: 0 for group in ordered_groups}
group_totals_pred = {group: 0 for group in ordered_groups}

# 각 그룹의 총 빈도수 계산
for group in ordered_groups:
    for tag in tag_groups[group]:
        if tag in tag_frequencies:
            group_totals_gt[group] += tag_frequencies[tag]['gt']
            group_totals_pred[group] += tag_frequencies[tag]['pred']

# GT와 Pred 누적값 초기화
gt_bottoms = np.zeros(len(ordered_groups))
pred_bottoms = np.zeros(len(ordered_groups))

# 각 태그에 대한 핸들 저장 (범례용)
tag_handles = []
tag_labels = []

# 각 그룹의 태그 위치와 높이를 저장할 딕셔너리 - 텍스트 레이블용
gt_tag_positions = {}  # {(group_idx, tag): (bottom, height)}
pred_tag_positions = {}  # {(group_idx, tag): (bottom, height)}

# 각 그룹의 태그를 역순으로 처리하여 심각성이 높은 태그가 하단에 오도록 함
for group in ordered_groups:
    # 태그를 심각성 순으로 정렬 (낮은 심각성 -> 높은 심각성)
    tags_in_group = sort_tags_by_severity(tag_groups[group])
    
    # 그래프 그릴 때는 역순으로 처리 (높은 심각성부터 그려서 아래에 위치하도록)
    for i, tag in enumerate(reversed(tags_in_group)):
        gt_values = []
        pred_values = []
        
        # 각 그룹에 해당 태그가 속하는지 확인하고 값 설정
        for group_idx, curr_group in enumerate(ordered_groups):
            if curr_group == group:
                # 정규화된 값 계산 (총합이 0이면 0으로 처리)
                gt_val = tag_frequencies[tag]['gt'] / group_totals_gt[curr_group] if group_totals_gt[curr_group] > 0 else 0
                pred_val = tag_frequencies[tag]['pred'] / group_totals_pred[curr_group] if group_totals_pred[curr_group] > 0 else 0
                gt_values.append(gt_val)
                pred_values.append(pred_val)
                
                # 태그 위치 저장 (나중에 레이블 표시용)
                gt_tag_positions[(group_idx, tag)] = (gt_bottoms[group_idx], gt_val)
                pred_tag_positions[(group_idx, tag)] = (pred_bottoms[group_idx], pred_val)
            else:
                gt_values.append(0)
                pred_values.append(0)
        
        # 심각성에 따른 색상 인덱스 선택
        severity_idx = min(get_tag_severity_level(tag), 3)  # 최대 4단계 색상 (0,1,2,3)
        
        # GT 색상: 심각성이 높을수록 진한 파란색 (3->0)
        gt_color = blue_colors[severity_idx]
        
        # Pred 색상: 심각성이 높을수록 진한 초록색 (11->8)
        pred_color = green_colors[severity_idx]
        
        # GT 막대 그리기 - 위치 조정
        gt_bar = plt.bar(x - bar_distance/2, gt_values, width, bottom=gt_bottoms, 
                  facecolor=gt_color, alpha=0.85)
        
        # Pred 막대 그리기 - 위치 조정
        pred_bar = plt.bar(x + bar_distance/2, pred_values, width, bottom=pred_bottoms, 
                    facecolor=pred_color, alpha=0.85)
  
        # 범례 항목 추가 (각 태그마다 한 번만)
        # GT 색상으로 범례 추가
        tag_handle = plt.Rectangle((0, 0), 1, 1, facecolor=gt_color, alpha=0.85, 
                                   edgecolor='black', linewidth=0.5)
        tag_handles.append(tag_handle)
        
        # 줄바꿈을 활용하여 긴 태그 이름 처리
        tag_labels.append(wrap_tag_name(tag))
        
        # 다음 막대를 위한 바닥값 업데이트
        gt_bottoms += np.array(gt_values)
        pred_bottoms += np.array(pred_values)

# 태그 레이블 추가 - 태그 그룹명과 겹치는 부분 제외
for (group_idx, tag), (bottom, height) in gt_tag_positions.items():
    if height > 0.00:  # 충분히 높은 영역에만 레이블 추가
        # 그룹명에서 태그와 겹치는 부분 제거
        group = ordered_groups[group_idx]
        clean_tag = tag.replace(group, '').strip()
        
        # 심각도에 해당하는 레이블만 표시 (예: "mild", "moderate" 등)
        for severity_term in ["no ", "mild ", "moderate ", "severe "]:
            if severity_term in clean_tag:
                clean_tag = severity_term.strip()  # 심각도 용어만 유지
                break
        
        # 텍스트 레이블 추가 - GT 막대
        text_x = group_idx - bar_distance/2
        text_y = bottom + height/2  # 영역 중앙에 위치
        plt.text(text_x, text_y, clean_tag, ha='center', va='center', 
                color='black', fontsize=9)

# Pred 영역 레이블 추가 
for (group_idx, tag), (bottom, height) in pred_tag_positions.items():
    if height > 0.00:  # 충분히 높은 영역에만 레이블 추가
        # 그룹명에서 태그와 겹치는 부분 제거
        group = ordered_groups[group_idx]
        clean_tag = tag.replace(group, '').strip()
        
        # 심각도에 해당하는 레이블만 표시 (예: "mild", "moderate" 등)
        for severity_term in ["no ", "mild ", "moderate ", "severe "]:
            if severity_term in clean_tag:
                clean_tag = severity_term.strip()  # 심각도 용어만 유지
                break
        
        # 텍스트 레이블 추가 - Pred 막대
        text_x = group_idx + bar_distance/2
        text_y = bottom + height/2  # 영역 중앙에 위치
        plt.text(text_x, text_y, clean_tag, ha='center', va='center', 
                color='black', fontsize=9)

# 축 설정
plt.xlabel('', fontsize=14, labelpad=15)
plt.ylabel('Distribution', fontsize=22, labelpad=15)
plt.title('', fontsize=16)

# x축 레이블
plt.xticks(x, ordered_groups, fontsize=20)

# y축 범위 설정 (0~1)
plt.ylim(0, 1.0)

# GT와 Pred 구분을 위한 범례 항목
gt_label = plt.Rectangle((0, 0), 1, 1, facecolor=blue_colors[3], edgecolor='black', 
                        label='GT', linewidth=0.5)
pred_label = plt.Rectangle((0, 0), 1, 1, facecolor=green_colors[3], edgecolor='black', 
                          label='Prediction', linewidth=0.5)

# 범례 - GT vs Pred 범례
plt.legend(handles=[gt_label, pred_label], 
         labels=['Ground Truth', 'Prediction'],
         loc='upper right', fontsize=20, ncol=2)

# 레이아웃 조정
plt.tight_layout(pad=4.0)
plt.subplots_adjust(bottom=0.1, left=0.15)

# 저장
labeled_stacked_path = os.path.join(output_dir, "normalized_tag_group_distribution_labeled.png")
plt.savefig(labeled_stacked_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"- 태그 레이블이 추가된 정규화된 분포 그래프: {labeled_stacked_path}")