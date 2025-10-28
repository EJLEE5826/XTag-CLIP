import pandas as pd
import os
import random

def split_csv_by_class(input_csv_path, train_csv_path, val_csv_path, test_size=0.2, random_seed=42, class_column='Class'):
    """
    CSV 파일에서 데이터를 읽어와 클래스별로 train과 validation으로 나눈 후 각각의 CSV 파일로 저장합니다.
    
    Args:
        input_csv_path: 입력 CSV 파일 경로
        train_csv_path: 저장할 train CSV 파일 경로
        val_csv_path: 저장할 validation CSV 파일 경로
        test_size: validation 세트의 비율 (기본값: 0.2, 즉 1/5)
        random_seed: 랜덤 시드
        class_column: 클래스 정보가 있는 컬럼명 (기본값: 'Class')
    """
    # 랜덤 시드 설정
    random.seed(random_seed)
    
    # CSV 파일 읽기
    df = pd.read_csv(input_csv_path)
    
    # 클래스 컬럼 확인
    if class_column not in df.columns:
        print(f"클래스 컬럼 '{class_column}'을 찾을 수 없습니다. 가능한 컬럼: {df.columns.tolist()}")
        print("클래스 컬럼명을 직접 지정해주세요.")
        return
    
    # 클래스별로 데이터 분할
    train_dfs = []
    val_dfs = []
    
    # 클래스별 데이터 수 확인
    class_counts = df[class_column].value_counts()
    print("각 클래스별 데이터 수:")
    for class_value, count in class_counts.items():
        print(f"- 클래스 {class_value}: {count}개")
    
    for class_value in df[class_column].unique():
        class_df = df[df[class_column] == class_value].reset_index(drop=True)
        print(f"클래스 {class_value} 처리 중... (총 {len(class_df)}개)")
        
        # 각 클래스의 인덱스를 섞기
        indices = list(range(len(class_df)))
        random.shuffle(indices)
        
        # validation 데이터 개수 계산 (전체의 20%)
        val_size = int(len(indices) * test_size)
        
        # 인덱스를 나눠서 train과 validation 세트 생성
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        train_class_df = class_df.iloc[train_indices].reset_index(drop=True)
        val_class_df = class_df.iloc[val_indices].reset_index(drop=True)
        
        train_dfs.append(train_class_df)
        val_dfs.append(val_class_df)
        
        print(f"  - 학습 데이터: {len(train_class_df)}개, 검증 데이터: {len(val_class_df)}개")
    
    # 분할된 데이터 프레임 합치기
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    
    # 결과 저장
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    print(f"\n총 {len(df)} 개의 데이터가 다음과 같이 분할되었습니다:")
    print(f"- 학습 데이터: {len(train_df)} 개 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"- 검증 데이터: {len(val_df)} 개 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"학습 데이터 저장 경로: {train_csv_path}")
    print(f"검증 데이터 저장 경로: {val_csv_path}")

if __name__ == "__main__":
    # 파일 경로 설정
    input_csv_path = "updated_scar_label_250614_pretrain_augmented.csv"
    train_csv_path = "updated_scar_label_250614_pretrain_augmented_training.csv"
    val_csv_path = "updated_scar_label_250614_pretrain_augmented_val.csv"
    
    # 클래스 별로 4:1 비율로 데이터 분할
    split_csv_by_class(input_csv_path, train_csv_path, val_csv_path, test_size=0.2, random_seed=42, class_column='Class')