import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import open_clip
from tqdm import tqdm
# BiomedCLIP을 위한 추가 임포트
from transformers import AutoProcessor, AutoModel
import torchvision.transforms as transforms


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
        ('clip_image_encoder', model.clip_image_encoder if hasattr(model, 'clip_image_encoder') else None),
        ('clip_text_encoder', model.clip_text_encoder if hasattr(model, 'clip_text_encoder') else None),
        ('additional_embedding', model.additional_embedding if hasattr(model, 'additional_embedding') else None),
        ('fusion_layer', model.fusion_layer if hasattr(model, 'fusion_layer') else None),
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


# 설정값
class Config:
    batch_size = 4
    num_workers = 4
    learning_rate = 5e-6
    weight_decay = 1e-4
    num_epochs = 100
    patience = 5  # Early stopping patience
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 224
    # BiomedCLIP 모델 설정
    use_biomedclip = False  # BiomedCLIP 사용 여부 - 문제가 있어서 일단 비활성화
    # 대체 방법: pretrained='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224' 사용 고려
    biomedclip_model_name = "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    # 기존 CLIP 설정 (BiomedCLIP 미사용 시)
    model_name = 'ViT-B/32'  # CLIP 모델 선택
    pretrained = 'laion400m_e32'  # 사전 학습 가중치
    is_train = True  # 훈련 데이터셋 여부
    lock_text_encoder = False  # 텍스트 인코더 파라미터 고정(True) 또는 업데이트 허용(False)
    
    def __init__(self):
        # 안전장치: biomedclip 사용 시 필요한 라이브러리 체크
        if self.use_biomedclip:
            try:
                import transformers
                print(f"Transformers 라이브러리 버전: {transformers.__version__}")
            except ImportError:
                print("경고: transformers 라이브러리를 찾을 수 없어 기본 CLIP 모델을 사용합니다.")
                self.use_biomedclip = False

# SCAR 데이터셋 클래스
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
            csv_file = os.path.join("./datasets/updated_scar_label_250218_train_augmented_human_simple.csv")
        else:
            csv_file = os.path.join("./datasets/updated_scar_label_250218_val_augmented_human_simple.csv")
            
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df["Use"] == "yes"].reset_index(drop=True)
        
        # label 없는 정보 제거
        self.df = self.df.dropna(subset=["Width", "Color", "Pigmentation", "Surface", 
                                        "Irregular_color", "Irregular_height"])
        
        # 이미지 경로 생성
        self.df["img_path"] = self.df["Name"].astype(str).str.strip().apply(
            lambda x: os.path.join(self.root, x)
        )
        
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
        
        # 다중 레이블이 있는 데이터 처리 (훈련 데이터의 경우)
        if self.is_train:
            # 다중 레이블이 있는 행 찾기
            multi_label_rows = []
            rows_to_drop = []
            
            for i, row in self.df.iterrows():
                class_label = row["Class"]
                if isinstance(class_label, str) and ',' in class_label:
                    # 다중 레이블을 가진 행은 첫 번째 레이블만 사용하도록 변경
                    class_idx = int(class_label.split(',')[0].strip())
                    new_row = row.copy()
                    new_row["Class"] = class_idx
                    multi_label_rows.append(new_row)
                    rows_to_drop.append(i)
            
            # 원래 다중 레이블 행 삭제
            if rows_to_drop:
                self.df = self.df.drop(rows_to_drop, axis=0)
                
            # 수정된 단일 레이블 행 추가
            if multi_label_rows:
                self.df = pd.concat([self.df, pd.DataFrame(multi_label_rows)], ignore_index=True)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]["img_path"]
        
        # 클래스 라벨 처리 - 단일 레이블로 변환
        class_label = self.df.iloc[idx]["Class"]
        if isinstance(class_label, str) and ',' in class_label:
            # 다중 레이블이 있으면 첫 번째 클래스만 사용
            class_idx = int(class_label.split(',')[0].strip())-1
        else:
            # 단일 레이블
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
            size = 0
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
            
        # 단일 레이블 정수 클래스 인덱스 반환
        return image, class_idx, additional_tensor

# CLIP 기반 분류 모델
class CLIPScarClassifier(nn.Module):
    def __init__(self, model_name=None, pretrained=None, num_classes=8, use_biomedclip=False, biomedclip_model_name=None, lock_text_encoder=True):
        super(CLIPScarClassifier, self).__init__()
        
        self.use_biomedclip = use_biomedclip
        self.lock_text_encoder = lock_text_encoder
        
        try:
            if use_biomedclip and biomedclip_model_name:
                # BiomedCLIP을 open_clip의 hf-hub 기능을 통해 로드 시도
                print(f"BiomedCLIP 모델을 초기화하려고 시도합니다: {biomedclip_model_name}")
                
                try:
                    # 직접 Visual Encoder를 초기화하는 대신 전체 모델 사용 
                    model, _, _ = open_clip.create_model_and_transforms(
                        model_name if model_name else 'ViT-B/32',
                        pretrained=f"hf-hub:{biomedclip_model_name}"
                    )
                    self.clip_image_encoder = model.visual
                    self.clip_text_encoder = model.transformer  # 텍스트 인코더 저장
                    
                    # 텍스트 인코더 잠금 (파라미터 고정)
                    if self.lock_text_encoder:
                        print("텍스트 인코더를 lock합니다 (학습 파라미터에서 제외)")
                        for param in self.clip_text_encoder.parameters():
                            param.requires_grad = False
                    
                    # CLIP의 출력 차원 얻기
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        feature_dim = self.clip_image_encoder(dummy_input).shape[-1]
                    print(f"BiomedCLIP의 특징 차원: {feature_dim}")
                    
                except Exception as e:
                    print(f"BiomedCLIP 모델 로드 실패: {str(e)}")
                    print("기본 CLIP 모델로 대체합니다.")
                    self.use_biomedclip = False
                    
                    # 기본 CLIP 모델로 대체
                    model, _, _ = open_clip.create_model_and_transforms(
                        'ViT-B/32' if model_name is None else model_name, 
                        pretrained='laion400m_e32' if pretrained is None else pretrained
                    )
                    self.clip_image_encoder = model.visual
                    self.clip_text_encoder = model.transformer  # 텍스트 인코더 저장
                    
                    # 텍스트 인코더 잠금
                    if self.lock_text_encoder:
                        print("텍스트 인코더를 lock합니다 (학습 파라미터에서 제외)")
                        for param in self.clip_text_encoder.parameters():
                            param.requires_grad = False
                    
                    # 특징 차원 얻기
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        feature_dim = self.clip_image_encoder(dummy_input).shape[-1]
            else:
                # Open CLIP 모델 로드
                print(f"기본 CLIP 모델 로드: {model_name}, {pretrained}")
                model, _, _ = open_clip.create_model_and_transforms(
                    'ViT-B/32' if model_name is None else model_name, 
                    pretrained='laion400m_e32' if pretrained is None else pretrained
                )
                self.clip_image_encoder = model.visual
                self.clip_text_encoder = model.transformer  # 텍스트 인코더 저장
                
                # 텍스트 인코더 잠금
                if self.lock_text_encoder:
                    print("텍스트 인코더를 lock합니다 (학습 파라미터에서 제외)")
                    for param in self.clip_text_encoder.parameters():
                        param.requires_grad = False
                
                # CLIP의 출력 차원 얻기
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    feature_dim = self.clip_image_encoder(dummy_input).shape[-1]
                print(f"CLIP의 특징 차원: {feature_dim}")
                
        except Exception as e:
            print(f"모델 초기화 중 예상치 못한 오류 발생: {str(e)}")
            # 비상용: 오류 발생시 기본 모델 사용
            print("비상용 CLIP 모델로 대체합니다.")
            
            model, _, _ = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
            self.clip_image_encoder = model.visual
            self.clip_text_encoder = model.transformer  # 텍스트 인코더 저장
            
            # 텍스트 인코더 잠금
            if self.lock_text_encoder:
                print("텍스트 인코더를 lock합니다 (학습 파라미터에서 제외)")
                for param in self.clip_text_encoder.parameters():
                    param.requires_grad = False
            
            # 특징 차원 얻기
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                feature_dim = self.clip_image_encoder(dummy_input).shape[-1]
        
        # 추가 특성을 처리하기 위한 임베딩 융합 레이어
        self.additional_embedding = nn.Sequential(
            nn.Linear(22, 128),  # 추가 특성의 총 차원 수 (각 카테고리 레이블 크기의 합)
            nn.ReLU(),
            nn.Dropout(0.3),
        )
            
        # 융합 레이어 및 분류 헤드
        self.fusion_layer = nn.Sequential(
            nn.Linear(feature_dim + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, additional_features):
        try:
            # CLIP 이미지 인코더를 통한 특징 추출
            # BiomedCLIP이든 일반 CLIP이든 동일한 방식으로 특징 추출
            image_features = self.clip_image_encoder(x)
        except Exception as e:
            print(f"이미지 인코더 추론 중 오류 발생: {str(e)}")
            # 오류 발생시 평균 풀링으로 대체 (비상용)
            image_features = x.mean([2, 3])  # 비상용: 평균 풀링으로 특징 추출
        
        # 추가 특성 임베딩
        additional_embedding = self.additional_embedding(additional_features)
        
        # 특징 벡터 결합
        combined_features = torch.cat([image_features, additional_embedding], dim=1)
        
        # 분류 헤드
        logits = self.fusion_layer(combined_features)
        
        return logits

# 조기 종료 클래스
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'best_clip_scar_model.pth')
        self.val_loss_min = val_loss

# 훈련 함수
def train_model(train_loader, val_loader, model, criterion, optimizer, config):
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    early_stopping = EarlyStopping(patience=config.patience, verbose=True)
    
    for epoch in range(config.num_epochs):
        # 훈련 단계
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Train]")
        
        for images, labels, additional in train_iter:
            images = images.to(config.device)
            labels = labels.to(config.device)
            additional = additional.to(config.device)
            
            optimizer.zero_grad()
            
            outputs = model(images, additional)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            train_iter.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct / total if total > 0 else 0
        
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        
        # 검증 단계
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.num_epochs} [Val]")
        
        with torch.no_grad():
            for images, labels, additional in val_iter:
                images = images.to(config.device)
                labels = labels.to(config.device)
                additional = additional.to(config.device)
                
                outputs = model(images, additional)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                val_iter.set_postfix(loss=loss.item(), acc=correct/total if total > 0 else 0)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total if total > 0 else 0
        
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f"Epoch {epoch+1}/{config.num_epochs} - "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")
        
        # 조기 종료 검사
        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    return train_losses, val_losses, train_accs, val_accs

# 평가 함수
def evaluate_model(model, test_loader, config):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, additional in tqdm(test_loader, desc="Evaluating"):
            images = images.to(config.device)
            labels = labels.to(config.device)
            additional = additional.to(config.device)
            
            outputs = model(images, additional)
            _, predicted = torch.max(outputs, 1)
            
            # 예측과 라벨 저장
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 전체 정확도 계산
    accuracy = accuracy_score(all_labels, all_preds)
    
    # 가중 평균 메트릭
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # 클래스별 메트릭
    class_metrics = {}
    num_classes = model.fusion_layer[-1].out_features
    
    # 클래스별로 성능 지표 계산
    for cls in range(num_classes):
        # 해당 클래스에 대한 이진 레이블 및 예측 생성
        cls_labels = (all_labels == cls).astype(int)
        cls_preds = (all_preds == cls).astype(int)
        
        # 클래스별 성능 지표
        cls_support = np.sum(cls_labels)
        
        if cls_support > 0:
            cls_accuracy = accuracy_score(cls_labels, cls_preds)
            cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
                cls_labels, cls_preds, average='binary', zero_division=0
            )
            
            # 클래스별 지표 저장
            class_metrics[cls] = {
                'accuracy': cls_accuracy,
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'support': cls_support
            }
    
    # 결과 저장
    results = {
        'accuracy': accuracy,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'class_metrics': class_metrics
    }
    
    return results

# 결과 시각화 함수
def plot_results(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(16, 6))
    
    # Loss 그래프
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Accuracy 그래프
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('clip_scar_classifier_results.png')
    plt.close()

# 메인 실행 함수
def main():
    config = Config()  # Config 객체 생성 및 초기화
    
    # 환경 변수를 통한 BiomedCLIP 활성화/비활성화
    import os
    use_biomedclip_env = os.environ.get('USE_BIOMEDCLIP', 'false').lower()
    if use_biomedclip_env == 'true':
        config.use_biomedclip = True
        print("환경 변수에 의해 BiomedCLIP이 활성화되었습니다.")
    elif use_biomedclip_env == 'false':
        config.use_biomedclip = False
        print("환경 변수에 의해 BiomedCLIP이 비활성화되었습니다.")
    
    # 데이터셋 경로
    dataset_root = ''
    
    # 전처리 함수 로드
    try:
        if config.use_biomedclip:
            print("BiomedCLIP 모델을 사용하려고 시도합니다...")
            
            # BiomedCLIP을 open_clip의 hf-hub 기능을 통해 로드 시도
            hf_model_name = f"hf-hub:{config.biomedclip_model_name}"
            print(f"모델 ID: {hf_model_name}")
            
            try:
                # open_clip을 통해 BiomedCLIP 로드 시도
                _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                    config.model_name, 
                    pretrained=hf_model_name
                )
                print("BiomedCLIP 모델이 성공적으로 로드되었습니다!")
            except Exception as e:
                print(f"open_clip을 통한 BiomedCLIP 로드 실패: {str(e)}")
                raise e
        else:
            # 기존 CLIP 모델 및 전처리 함수 로드
            print(f"기본 CLIP 모델 로드: {config.model_name}, {config.pretrained}")
            _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
                config.model_name, 
                pretrained=config.pretrained,
            )
    except Exception as e:
        print(f"모델 로드 실패: {str(e)}")
        print("기본 CLIP 전처리 함수를 사용합니다.")
        # 기본 CLIP 전처리 함수 정의
        preprocess_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        preprocess_val = preprocess_train
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = ScarDataset(root=dataset_root, transform=preprocess_train, is_train=True)
    val_dataset = ScarDataset(root=dataset_root, transform=preprocess_val, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    
    # 모델 초기화
    model = CLIPScarClassifier(
        model_name=None if config.use_biomedclip else config.model_name,
        pretrained=None if config.use_biomedclip else config.pretrained,
        use_biomedclip=config.use_biomedclip,
        biomedclip_model_name=config.biomedclip_model_name if config.use_biomedclip else None,
        num_classes=len(train_dataset.classes),
        lock_text_encoder=config.lock_text_encoder
    ).to(config.device)
    
    # 모델 파라미터 요약 정보 출력
    print_parameter_summary(model)
    
    # 단일 레이블 분류를 위한 크로스 엔트로피 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("Starting training...")
    
    # 모델 훈련
    train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader, model, criterion, optimizer, config
    )
    
    # 결과 시각화
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # 최고 모델 로드
    model.load_state_dict(torch.load('best_clip_scar_model.pth'))
    
    # 검증 세트 평가
    print("\nEvaluating on validation set...")
    results = evaluate_model(model, val_loader, config)
    
    # 결과 출력
    print("\nResults:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Weighted Precision: {results['weighted_precision']:.4f}")
    print(f"Weighted Recall: {results['weighted_recall']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")

    print("\nClass-wise results:")
    for cls, metrics in results['class_metrics'].items():
        print(f"Class {cls+1} ({train_dataset.classes[cls]}):")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Precision: {metrics['precision']:.4f}")
        print(f"  - Recall: {metrics['recall']:.4f}")
        print(f"  - F1 Score: {metrics['f1']:.4f}")
        print(f"  - Support: {metrics['support']}")

    # 파라미터 수 요약 출력
    print_parameter_summary(model)

if __name__ == '__main__':
    main()

