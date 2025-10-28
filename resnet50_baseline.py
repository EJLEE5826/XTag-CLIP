import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import json

# 모델 정보 출력 함수
def print_model_info(model):
    """
    모델의 구조와 파라미터 수를 출력하는 함수
    
    Args:
        model (nn.Module): 확인할 모델
    """
    # 모델 구조 출력
    print("\n" + "="*50)
    print("모델 구조:")
    print(model)
    print("="*50)
    
    # 각 레이어별 파라미터 수 계산
    print("\n각 레이어별 파라미터 수:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
    
    # 총 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*50)
    print(f"총 파라미터 수: {total_params:,}")
    print(f"학습 가능한 파라미터 수: {trainable_params:,}")
    print(f"고정된 파라미터 수: {total_params - trainable_params:,}")
    print("="*50 + "\n")

# 설정값들
class Config:
    batch_size = 4
    epochs = 100
    learning_rate = 1e-4
    patience = 15
    image_size = 224
    num_workers = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 흉터 데이터셋 클래스 - 단일 레이블 분류 지원
class ScarDataset(Dataset):
    def __init__(self, root, transform=None, is_train=True):
        """
        Args:
            root (str): 이미지와 라벨 정보가 있는 폴더 경로
            transform (callable, optional): 이미지 전처리 함수
            is_train (bool): 훈련 데이터셋인지 여부
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train
        
        # CSV 파일 읽기
        if self.is_train:
            csv_file = os.path.join("datasets/updated_scar_label_250218_train_augmented_human_simple.csv")
        else:
            csv_file = os.path.join("datasets/updated_scar_label_250218_val_augmented_human_simple.csv")
            
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Use"] == "yes"].reset_index(drop=True)
        
        # label 없는 정보 제거
        self.df = self.df.dropna(subset=["Width", "Color", "Pigmentation", "Surface", 
                                        "Irregular_color", "Irregular_height"])
        
        # 클래스 정보 로드
        with open(os.path.join(root, "label_info.json"), 'r') as f:
            self.label_info = json.load(f)
            
        self.classes = self.label_info["Class"]
        self.num_classes = len(self.classes)
        
        # 추가 특성 매핑
        self.Width_label = ["Linear", "Widened", "Linear bulging"]
        self.Color_label = ["Normal", "Pink", "Red", "Purple"]
        self.Pigmentation_label = ["Normal", "Pigmented", "Hypopigmented"]
        self.Surface_label = ["Flat", "Hypertrophic", "Keloid", "Atrophic"]
        self.Irregular_color_label = ["no", "mild", "moderate", "severe"]
        self.Irregular_height_label = ["no", "mild", "moderate", "severe"]
        
        # 이미지 경로 생성
        self.df["img_path"] = self.df["Name"].astype(str).str.strip().apply(
            lambda x: os.path.join(self.root, x)
        )
        
        # 훈련 데이터에서 다중 레이블 처리
        if self.is_train:
            # 다중 레이블이 있는 데이터를 복제하여 각각 단일 레이블로 처리
            multi_label_rows = []
            for i, row in self.df.iterrows():
                class_label = row["Class"]
                if isinstance(class_label, str) and ',' in class_label:
                    class_indices = [int(item.strip()) for item in class_label.split(',')]
                    for cls_idx in class_indices:
                        new_row = row.copy()
                        new_row["Class"] = cls_idx
                        multi_label_rows.append(new_row)
                    # 원래 행 삭제
                    self.df.drop(i, inplace=True)
            
            # 새로운 행들 추가
            if multi_label_rows:
                self.df = pd.concat([self.df, pd.DataFrame(multi_label_rows)], ignore_index=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img_path"]
        img_name = self.df.iloc[idx]["Name"]
        
        # 클래스 라벨 처리 - 단일 라벨로 변환
        class_label = self.df.iloc[idx]["Class"]
        
        # 단일 클래스로 처리
        if isinstance(class_label, str) and ',' in class_label:
            # 검증 데이터에 다중 레이블이 있다면 첫 번째 클래스만 사용
            class_idx = int(class_label.split(',')[0].strip())-1
        else:
            # 단일 클래스
            class_idx = int(class_label)-1
            
        # 추가 특성
        width = self.df.iloc[idx]["Width"]
        color = self.df.iloc[idx]["Color"]
        pigmentation = self.df.iloc[idx]["Pigmentation"]
        surface = self.df.iloc[idx]["Surface"]
        irregular_color = self.df.iloc[idx]["Irregular_color"]
        irregular_height = self.df.iloc[idx]["Irregular_height"]
        
        # 이미지 로드 및 변환
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # 추가 특성을 위한 텐서 생성
        additional_features = {
            "Width": self.Width_label.index(width) if width in self.Width_label else 0,
            "Color": self.Color_label.index(color) if color in self.Color_label else 0,
            "Pigmentation": self.Pigmentation_label.index(pigmentation) if pigmentation in self.Pigmentation_label else 0,
            "Surface": self.Surface_label.index(surface) if surface in self.Surface_label else 0,
            "Irregular_color": self.Irregular_color_label.index(irregular_color) if irregular_color in self.Irregular_color_label else 0,
            "Irregular_height": self.Irregular_height_label.index(irregular_height) if irregular_height in self.Irregular_height_label else 0
        }
        
        additional_tensor = torch.zeros(
            sum([len(self.Width_label), len(self.Color_label), len(self.Pigmentation_label),
                 len(self.Surface_label), len(self.Irregular_color_label), len(self.Irregular_height_label)]),
            dtype=torch.float32
        )
        
        # 추가 특성 원-핫 인코딩
        current_pos = 0
        for category, value in additional_features.items():
            if category == "Width":
                size = len(self.Width_label)
            elif category == "Color":
                size = len(self.Color_label)
            elif category == "Pigmentation":
                size = len(self.Pigmentation_label)
            elif category == "Surface":
                size = len(self.Surface_label)
            elif category == "Irregular_color":
                size = len(self.Irregular_color_label)
            elif category == "Irregular_height":
                size = len(self.Irregular_height_label)
                
            if 0 <= value < size:
                additional_tensor[current_pos + value] = 1
            current_pos += size
            
        return image, class_idx, additional_tensor, img_name

# ResNet50 기반 분류 모델 - 단일 레이블 분류용
class ScarClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(ScarClassifier, self).__init__()
        
        # 사전 훈련된 ResNet50 로드
        self.backbone = models.resnet50(weights='DEFAULT')
        
        # 마지막 fully connected layer 수정
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # 기존 fc layer 제거
        
        # 새로운 분류 헤드 추가 - 단일 레이블 분류용
        self.classification_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classification_head(features)
        return logits

# 조기 종료 클래스
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

# 훈련 함수 - 단일 레이블 분류용
def train_model(train_loader, val_loader, model, criterion, optimizer, config):
    early_stopping = EarlyStopping(patience=config.patience)
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(config.epochs):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels, additional_features, _ in train_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)  # 정수 형태의 레이블
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        # 평균 손실과 정확도 계산
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_accuracy = correct / total
        
        # 검증 단계
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, additional_features, _ in val_loader:
                images = images.to(config.device)
                labels = labels.to(config.device)  # 정수 형태의 레이블
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # F1 스코어용 데이터 수집
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 평균 손실과 정확도 계산
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        
        # F1 스코어 계산
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{config.epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f} | F1: {f1:.4f}')
        
        # 조기 종료 확인
        if early_stopping(avg_val_loss):
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        # 최고 모델 저장
        if avg_val_loss == min(val_losses):
            torch.save(model.state_dict(), 'best_scar_model.pth')
    
    return train_losses, val_losses, train_accs, val_accs

# 평가 함수 - 단일 레이블 평가용
def evaluate_model(model, test_loader, config):
    model.eval()
    all_outputs = []
    all_preds = []
    all_labels = []
    img_names = []
    
    with torch.no_grad():
        for images, labels, _, batch_img_names in test_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_outputs.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            img_names.extend(batch_img_names)
    
    # 넘파이 배열로 변환
    all_outputs = np.array(all_outputs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 결과 저장
    results = {}
    
    # 정확도 계산
    accuracy = np.mean(all_preds == all_labels)
    results['accuracy'] = accuracy
    
    # 클래스별 메트릭
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, zero_division=0
    )
    
    # 클래스별 정확도 계산 추가
    class_accuracies = []
    for cls in range(len(precision)):
        # 해당 클래스에 속한 샘플들만 선택
        cls_indices = np.where(all_labels == cls)[0]
        
        # 해당 클래스 샘플이 있는 경우에만 계산
        if len(cls_indices) > 0:
            # 해당 클래스 샘플들에 대한 예측과 실제 레이블 비교
            cls_acc = np.mean(all_preds[cls_indices] == all_labels[cls_indices])
            class_accuracies.append(cls_acc)
        else:
            class_accuracies.append(0.0)
    
    class_metrics = {}
    for cls in range(len(precision)):
        class_metrics[cls] = {
            'accuracy': class_accuracies[cls],  # 클래스별 정확도 추가
            'precision': precision[cls],
            'recall': recall[cls],
            'f1': f1[cls],
            'support': support[cls]
        }
    
    results['class_metrics'] = class_metrics
    
    # 전체 F1 및 정밀도, 재현율
    macro_f1 = np.mean(f1)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    
    results['macro_f1'] = macro_f1
    results['macro_precision'] = macro_precision
    results['macro_recall'] = macro_recall
    
    # 결과 출력
    print(f'정확도: {accuracy:.4f}')
    print(f'매크로 정밀도: {macro_precision:.4f}')
    print(f'매크로 재현율: {macro_recall:.4f}')
    print(f'매크로 F1 점수: {macro_f1:.4f}')
    
    print('\n클래스별 성능:')
    for cls, metrics in class_metrics.items():
        print(f'Class {cls+1}: Acc={metrics["accuracy"]:.4f}, P={metrics["precision"]:.4f}, ' + 
              f'R={metrics["recall"]:.4f}, F1={metrics["f1"]:.4f}, Support={metrics["support"]}')
    
    return results

# 모델 구조 및 파라미터 출력 함수

def count_parameters(model):
    """
    모델의 총 파라미터 수와 학습 가능한 파라미터 수를 계산합니다.
    Args:
        model (nn.Module): 파라미터 수를 계산할 PyTorch 모델
    Returns:
        dict: 총 파라미터 수, 학습 가능한 파라미터 수, 모듈별 파라미터 수를 포함하는 딕셔너리
    """
    # 전체 파라미터 수 계산
    total_params = sum(p.numel() for p in model.parameters())
    
    # 학습 가능한 파라미터 수 계산
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 모듈별 파라미터 수 계산
    module_params = {}
    
    # 주요 모듈 목록 (CLIP 모델의 주요 컴포넌트)
    module_list = [

    ]
    
    for name, module in module_list:
        if module is not None:
            # 모듈의 총 파라미터 수
            total = sum(p.numel() for p in module.parameters())
            # 모듈의 학습 가능한 파라미터 수
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            # 결과 저장
            module_params[name] = {
                'total': total,
                'trainable': trainable,
                'frozen': total - trainable
            }
    
    # 결과 반환
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': total_params - trainable_params,
        'modules': module_params
    }

def print_parameter_summary(model):
    """
    모델의 파라미터 수 요약 정보를 출력합니다.
    Args:
        model (nn.Module): 파라미터 요약을 출력할 PyTorch 모델
    """
    param_info = count_parameters(model)
    
    total_params = param_info['total_params']
    trainable_params = param_info['trainable_params']
    frozen_params = param_info['frozen_params']
    
    print("\n" + "="*50)
    print("MODEL PARAMETER SUMMARY")
    print("="*50)
    
    print(f"\nTotal Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"Frozen Parameters: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
    print(f"Trainable Parameters Ratio: {trainable_params/total_params*100:.2f}%")
    
    print("\nModule-wise Breakdown:")
    print("-"*50)
    
    for name, info in param_info['modules'].items():
        total = info['total']
        trainable = info['trainable']
        frozen = info['frozen']
        
        print(f"\n{name}:")
        print(f"  Total: {total:,} ({total/1e6:.2f}M)")
        print(f"  Trainable: {trainable:,} ({trainable/1e6:.2f}M)")
        print(f"  Frozen: {frozen:,} ({frozen/1e6:.2f}M)")
        if total > 0:
            print(f"  Trainable Ratio: {trainable/total*100:.2f}%")
            print(f"  % of Model Total: {total/param_info['total_params']*100:.2f}%")
    
    print("\n" + "="*50)

# 메인 실행 함수
def main():
    config = Config()
    
    # 데이터셋 경로
    dataset_root = ''
    
    # 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = ScarDataset(root=dataset_root, transform=transform, is_train=True)
    val_dataset = ScarDataset(root=dataset_root, transform=transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    
    # 모델 초기화
    model = ScarClassifier(num_classes=8).to(config.device)
    
    # 모델 구조 및 파라미터 출력
    print_parameter_summary(model)
    
    # 단일 레이블 분류를 위한 교차 엔트로피 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print("Starting training...")
    
    # 모델 훈련
    train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader, model, criterion, optimizer, config
    )
    
    # 최고 모델 로드
    model.load_state_dict(torch.load('best_scar_model.pth'))
    
    # 검증 세트 평가
    print("\nEvaluating on validation set...")
    results = evaluate_model(model, val_loader, config)
    
    # 결과 시각화
    plt.figure(figsize=(16, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 2, 3)
    class_indices = list(results['class_metrics'].keys())
    class_f1s = [results['class_metrics'][cls]['f1'] for cls in class_indices]
    plt.bar(range(len(class_f1s)), class_f1s)
    plt.xlabel('Class')
    plt.ylabel('F1 Score')
    plt.title('Class-wise F1 Score')
    plt.xticks(range(len(class_f1s)), [f'Class {i+1}' for i in class_indices], rotation=45)
    
    plt.subplot(2, 2, 4)
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    values = [results['accuracy'], results['macro_precision'], 
              results['macro_recall'], results['macro_f1']]
    plt.bar(range(len(values)), values)
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(range(len(values)), metrics)
    
    plt.tight_layout()
    plt.savefig('scar_classifier_results.png')
    plt.show()

if __name__ == '__main__':
    main()
