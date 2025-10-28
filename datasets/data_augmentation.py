import os
import torch
import pandas as pd
import numpy as np
import random
from collections import Counter, defaultdict
from torchvision import transforms
from torchvision.io import read_image, write_jpeg
from torchvision.transforms import functional as F

# Spatial-level augmentation transforms
class SpatialAugmentation:
    def __init__(self, flip_prob=0.5, rotation_degrees=30, scale_range=(0.8, 1.2), seed=None):
        self.flip_prob = flip_prob
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.seed = seed
        self.rng = torch.Generator()
        if self.seed is not None:
            self.rng.manual_seed(self.seed)

    def __call__(self, img, index=0):
        # 이미지별로 다른 시드 사용 (같은 이미지는 매번 동일한 변형)
        if self.seed is not None:
            img_seed = self.seed + index
            img_rng = torch.Generator().manual_seed(img_seed)
        else:
            img_rng = self.rng
        
        # Random horizontal flip
        if torch.rand(1, generator=img_rng) < self.flip_prob:
            img = F.hflip(img)
        
        # Random vertical flip
        if torch.rand(1, generator=img_rng) < self.flip_prob:
            img = F.vflip(img)
        
        # Random rotation
        angle = torch.empty(1).uniform_(-self.rotation_degrees, self.rotation_degrees, generator=img_rng).item()
        img = F.rotate(img, angle)
        
        # Random zoom (scale)
        scale = torch.empty(1).uniform_(*self.scale_range, generator=img_rng).item()
        h, w = img.shape[1:]
        new_h, new_w = int(h * scale), int(w * scale)
        img = F.resize(img, [new_h, new_w])
        
        # Center crop or pad to original size
        if scale >= 1.0:
            img = F.center_crop(img, [h, w])
        else:
            pad_h = max(0, h - new_h)
            pad_w = max(0, w - new_w)
            img = F.pad(img, [pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2])
        
        return img

def class_balanced_augment_and_save(csv_file, img_dir, output_dir, target_samples_per_class=None, 
                                  max_augment_per_img=10, output_csv=None, seed=None):
    """
    클래스 불균형을 고려한 데이터 증강 함수 - 단일 클래스만 처리하고 랜덤 시드 사용
    
    Args:
        csv_file: 이미지 파일명과 클래스 정보가 담긴 CSV 파일 경로
        img_dir: 원본 이미지 디렉토리
        output_dir: 증강된 이미지를 저장할 디렉토리
        target_samples_per_class: 각 클래스당 목표 샘플 수 (None인 경우 최대 클래스 수로 설정)
        max_augment_per_img: 각 이미지당 최대 증강 횟수
        output_csv: 증강된 이미지 정보를 포함한 CSV 파일 경로 (None인 경우 원본 CSV 파일명에 _augmented를 붙임)
        seed: 랜덤 시드 (None인 경우 랜덤성 유지)
    """
    # 시드 설정
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"랜덤 시드가 {seed}로 설정되었습니다.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 출력 CSV 파일 경로 설정
    if output_csv is None:
        base_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_csv = os.path.join(os.path.dirname(csv_file), f"{base_name}_augmented.csv")
    
    # CSV 파일 로드 및 'Use'가 'yes'인 데이터 필터링
    df = pd.read_csv(csv_file)
    df = df[df['Use'] == 'yes'].reset_index(drop=True)
    
    # 증강된 이미지 정보를 저장할 데이터프레임 생성
    augmented_df = pd.DataFrame(columns=df.columns)
    
    # 클래스별 이미지 리스트 생성 (균등한 증강을 위함)
    class_to_images = defaultdict(list)
    
    # 각 이미지의 클래스 분석 및 클래스별 카운트 계산
    for idx, row in df.iterrows():
        cls = str(row['Class'])  # 단일 클래스 처리
        class_to_images[cls].append(idx)
    
    # 클래스별 이미지 개수 계산
    class_counts = {cls: len(images) for cls, images in class_to_images.items()}
    class_counts = pd.Series(class_counts)
    print(f"클래스별 이미지 개수:\n{class_counts}")
    
    # 목표 샘플 수 설정 (설정되지 않은 경우 최대 클래스의 샘플 수로 설정)
    if target_samples_per_class is None:
        target_samples_per_class = class_counts.max()
    print(f"클래스당 목표 샘플 수: {target_samples_per_class}")
    
    # 클래스별 필요한 추가 샘플 수 계산
    needed_samples = {}
    for cls, count in class_counts.items():
        if count < target_samples_per_class:
            needed_samples[cls] = target_samples_per_class - count
        else:
            needed_samples[cls] = 0
    
    print(f"클래스별 추가로 필요한 샘플 수:\n{needed_samples}")
    
    # 증강 객체 생성 (시드 적용)
    aug = SpatialAugmentation(seed=seed)
    
    # 클래스별 증강 이미지 카운트 추적
    augmented_counts = {cls: 0 for cls in class_counts.index}
    
    # 전체 이미지 인덱스 (고유한 시드 생성용)
    global_img_idx = 0
    
    # 각 클래스에 대해 필요한 추가 샘플을 균등하게 생성
    for cls, needed in needed_samples.items():
        if needed <= 0:
            continue
        
        # 이 클래스에 속한 이미지 개수
        cls_images = class_to_images[cls]
        if not cls_images:
            continue
        
        # 이미지당 증강할 기본 횟수 계산
        base_aug_per_img = needed // len(cls_images)
        
        # 나머지 증강 횟수 (추가로 증강할 이미지 수)
        extra_aug_imgs = needed % len(cls_images)
        
        print(f"클래스 {cls}의 각 이미지당 기본 증강 횟수: {base_aug_per_img}, 추가 증강 이미지 수: {extra_aug_imgs}")
        
        # 각 이미지별 증강 횟수 결정
        aug_counts_per_img = {}
        for i, idx in enumerate(cls_images):
            # 기본 증강 횟수 + 추가 증강 여부 결정
            aug_counts_per_img[idx] = base_aug_per_img + (1 if i < extra_aug_imgs else 0)
        
        # 증강 수행
        for idx, aug_count in aug_counts_per_img.items():
            if aug_count <= 0:
                continue
                
            # 이미지 정보 가져오기
            row = df.iloc[idx]
            img_name = row['Name']
            img_class = str(row['Class'])  # 단일 클래스
            
            # 이미지 로드
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                print(f"경고: 이미지 파일을 찾을 수 없습니다 - {img_path}")
                continue
            
            try:
                img = read_image(img_path)
                # 이미지가 4채널인 경우 RGB 채널만 사용
                if img.shape[0] == 4:
                    img = img[:3]
            except Exception as e:
                print(f"이미지 {img_path} 로딩 중 오류 발생: {e}")
                continue
            
            # 내부 경로와 파일 이름 분리
            rel_path = os.path.dirname(img_name)
            file_name = os.path.basename(img_name)
            
            # 출력 디렉토리에 동일한 폴더 구조 생성
            if rel_path:
                out_subdir = os.path.join(output_dir, rel_path)
                os.makedirs(out_subdir, exist_ok=True)
            else:
                out_subdir = output_dir
            
            # 증강 수행
            for i in range(min(aug_count, max_augment_per_img)):
                # 각 이미지와 증강 인덱스별로 고유한 시드 적용
                aug_idx = global_img_idx * 100 + i
                aug_img = aug(img, index=aug_idx)
                
                # 이미지가 4채널인 경우 RGB 채널만 사용
                if aug_img.shape[0] == 4:
                    aug_img = aug_img[:3]
                
                # 파일 이름 및 경로 생성
                out_file = f"{os.path.splitext(file_name)[0]}_aug{i}.jpg"
                if rel_path:
                    out_fname = os.path.join(rel_path, out_file)
                    out_path = os.path.join(out_subdir, out_file)
                else:
                    out_fname = out_file
                    out_path = os.path.join(output_dir, out_file)
                
                # 이미지 저장
                write_jpeg(aug_img, out_path)
                
                # 해당 클래스의 증강 카운트 증가
                augmented_counts[img_class] += 1
                
                # 증강된 이미지 정보를 데이터프레임에 추가
                new_row = row.copy()
                new_row['Name'] = out_fname  # 새 파일 이름으로 변경
                augmented_df = pd.concat([augmented_df, pd.DataFrame([new_row])], ignore_index=True)
            
            # 전역 이미지 인덱스 증가
            global_img_idx += 1
    
    # 최종 클래스별 증강 이미지 수 출력
    print(f"클래스별 증강된 이미지 수:\n{augmented_counts}")
    print(f"최종 클래스별 이미지 수(원본 + 증강):")
    for cls in class_counts.index:
        print(f"클래스 {cls}: {class_counts[cls] + augmented_counts[cls]}")
    
    # 원본 데이터와 증강 데이터를 합친 새 CSV 파일 저장
    combined_df = pd.concat([df, augmented_df], ignore_index=True)
    
    # CSV 파일로 저장
    combined_df.to_csv(output_csv, index=False)
    print(f"증강된 이미지 정보가 포함된 CSV 파일이 저장되었습니다: {output_csv}")
    
    return combined_df

if __name__ == "__main__":
    # 예시 실행 - 단일 클래스만 처리 + 시드 설정
    class_balanced_augment_and_save(
        '',  # 라벨 CSV 파일
        '',  # 입력 이미지 디렉토리
        '',  # 출력 디렉토리
        target_samples_per_class=18,  # 각 클래스당 목표 샘플 수
        max_augment_per_img=4,  # 이미지당 최대 증강 횟수
        seed=42  # 랜덤 시드 설정 (재현성 보장)
    )