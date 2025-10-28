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
import torchvision.transforms as transforms
from tqdm import tqdm
# BiomedCLIP 관련 임포트
from transformers import AutoProcessor, AutoModel
from transformers.models.clip import CLIPProcessor, CLIPModel 
import open_clip

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
    biomedclip_model_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    is_train = True  # 훈련 데이터셋 여부
    lock_text_encoder = False  # 텍스트 인코더 고정 여부 (True: 고정, False: 학습 가능)

# SCAR 데이터셋 클래스
class ScarDataset(Dataset):
    def __init__(self, root, processor=None, transform=None, is_train=True):
        """
        Args:
            root (str): 이미지와 라벨 정보가 있는 폴더 경로
            processor (callable, optional): BiomedCLIP 프로세서
            transform (callable, optional): 기본 이미지 전처리 함수 (프로세서가 없을 때 사용)
            is_train (bool): 훈련 데이터셋인지 여부
        """
        self.root = root
        self.processor = processor
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
        
        # 이미지 로드
        image = Image.open(img_path).convert('RGB')
        
        # 이미지 변환 (프로세서 또는 transform 사용)
        if self.processor:
            # BiomedCLIP processor 사용
            try:
                # 이미지만 전처리
                inputs = self.processor(images=image, return_tensors="pt", padding=True)
                if 'pixel_values' in inputs:
                    processed_image = inputs.pixel_values[0]
                else:
                    # pixel_values가 없다면 일반 전처리로 폴백
                    processed_image = self._fallback_transform(image)
            except Exception as e:
                print(f"프로세서 오류, 일반 전처리로 대체: {str(e)}")
                processed_image = self._fallback_transform(image)
        elif self.transform:
            # 일반 전처리 사용
            processed_image = self.transform(image)
        else:
            # 기본 전처리
            processed_image = self._fallback_transform(image)
            
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
        return processed_image, class_idx, additional_tensor
    
    def _fallback_transform(self, image):
        """기본 전처리 함수"""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                              std=[0.26862954, 0.26130258, 0.27577711])
        ])
        return transform(image)

# BiomedCLIP 기반 분류 모델
class BiomedCLIPScarClassifier(nn.Module):
    def __init__(self, model_name, num_classes=8, lock_text_encoder=True):
        super(BiomedCLIPScarClassifier, self).__init__()
        
        print(f"BiomedCLIP 모델을 초기화합니다: {model_name}")
        self.lock_text_encoder = lock_text_encoder
        
        # BiomedCLIP 모델 로드 시도 - test_biomedclip.py에서 작동하는 방법 사용
        model = None
        feature_dim = None
        
        # 방법 1: create_model_from_pretrained 사용 (test_biomedclip.py에서 동작하는 방법)
        try:
            print("방법 1: open_clip.create_model_from_pretrained 함수 사용 (권장)")
            
            # 테스트 스크립트에서 동작하는 방식
            from open_clip import create_model_from_pretrained
            model, _ = create_model_from_pretrained(f'hf-hub:{model_name}')
            self.vision_encoder = model.visual
            
            # 텍스트 인코더 저장
            self.text_encoder = model.transformer
            
            # 텍스트 인코더 고정 (lock_text_encoder 옵션에 따라)
            if self.lock_text_encoder:
                print("✅ 텍스트 인코더를 고정합니다 (requires_grad=False)")
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            else:
                print("⚠️ 텍스트 인코더가 학습 가능한 상태입니다 (requires_grad=True)")
                
            print("✅ BiomedCLIP 모델 로드 성공!")
            
            # 특징 차원 확인
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.vision_encoder(dummy_input)
                feature_dim = features.shape[-1]
            print(f"특징 차원: {feature_dim}")
            
            # 인코더 타입 저장
            self.encoder_type = "openclip"
            
        except Exception as e:
            print(f"방법 1 실패: {str(e)}")
            
            # 방법 2: create_model_and_transforms 사용
            try:
                print("방법 2: open_clip.create_model_and_transforms 함수 사용")
                
                model, _, _ = open_clip.create_model_and_transforms(
                    'ViT-B/32',  # 다른 아키텍처
                    pretrained=f"hf-hub:{model_name}"
                )
                self.vision_encoder = model.visual
                
                # 텍스트 인코더 저장
                self.text_encoder = model.transformer
                
                # 텍스트 인코더 고정 (lock_text_encoder 옵션에 따라)
                if self.lock_text_encoder:
                    print("✅ 텍스트 인코더를 고정합니다 (requires_grad=False)")
                    for param in self.text_encoder.parameters():
                        param.requires_grad = False
                else:
                    print("⚠️ 텍스트 인코더가 학습 가능한 상태입니다 (requires_grad=True)")
                
                print("✅ 방법 2로 BiomedCLIP 모델 로드 성공!")
                
                # 특징 차원 확인
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    features = self.vision_encoder(dummy_input)
                    feature_dim = features.shape[-1]
                print(f"특징 차원: {feature_dim}")
                
                # 인코더 타입 저장
                self.encoder_type = "openclip"
                
            except Exception as e2:
                print(f"방법 2 실패: {str(e2)}")
                
                # 방법 3: CLIPModel 사용
                try:
                    print("방법 3: transformers CLIPModel 사용")
                    
                    self.biomedclip = CLIPModel.from_pretrained(model_name)
                    self.vision_encoder = self.biomedclip.vision_model
                    
                    # 텍스트 인코더 저장
                    self.text_encoder = self.biomedclip.text_model
                    
                    # 텍스트 인코더 고정 (lock_text_encoder 옵션에 따라)
                    if self.lock_text_encoder:
                        print("✅ 텍스트 인코더를 고정합니다 (requires_grad=False)")
                        for param in self.text_encoder.parameters():
                            param.requires_grad = False
                    else:
                        print("⚠️ 텍스트 인코더가 학습 가능한 상태입니다 (requires_grad=True)")
                    
                    print("✅ 방법 3으로 BiomedCLIP 모델 로드 성공!")
                    
                    # 특징 차원 확인
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224).to(next(self.vision_encoder.parameters()).device)
                        outputs = self.vision_encoder(pixel_values=dummy_input)
                        feature_dim = outputs.last_hidden_state[:, 0].shape[-1]
                    print(f"특징 차원: {feature_dim}")
                    
                    # 인코더 타입 저장
                    self.encoder_type = "transformers"
                    
                except Exception as e3:
                    print(f"방법 3 실패: {str(e3)}")
                    print("BiomedCLIP 로드 실패! 기본 OpenAI CLIP으로 대체합니다.")
                    
                    # 모든 로드 방법 실패시, 일반 OpenAI CLIP으로 대체
                    model, _, _ = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
                    self.vision_encoder = model.visual
                    
                    # 텍스트 인코더 저장
                    self.text_encoder = model.transformer
                    
                    # 텍스트 인코더 고정 (lock_text_encoder 옵션에 따라)
                    if self.lock_text_encoder:
                        print("✅ 텍스트 인코더를 고정합니다 (requires_grad=False)")
                        for param in self.text_encoder.parameters():
                            param.requires_grad = False
                    else:
                        print("⚠️ 텍스트 인코더가 학습 가능한 상태입니다 (requires_grad=True)")
                    
                    print("⚠️ OpenAI CLIP 모델로 대체되었습니다.")
                    
                    # 특징 차원 확인
                    with torch.no_grad():
                        dummy_input = torch.randn(1, 3, 224, 224)
                        features = self.vision_encoder(dummy_input)
                        feature_dim = features.shape[-1]
                    print(f"특징 차원: {feature_dim}")
                    
                    # 인코더 타입 저장
                    self.encoder_type = "openclip_fallback"
            
            # 방법 3: OpenCLIP의 기본 ViT 모델 사용
            try:
                model, _, _ = open_clip.create_model_and_transforms('ViT-B/16', pretrained='openai')
            except Exception as e3:
                print(f"OpenAI ViT-B/16 로드 실패: {str(e3)}")
                print("다른 기본 모델로 시도...")
                model, _, _ = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
            
            self.vision_encoder = model.visual
            
            # 텍스트 인코더 저장
            self.text_encoder = model.transformer
            
            # 텍스트 인코더 고정 (lock_text_encoder 옵션에 따라)
            if self.lock_text_encoder:
                print("✅ 텍스트 인코더를 고정합니다 (requires_grad=False)")
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
            else:
                print("⚠️ 텍스트 인코더가 학습 가능한 상태입니다 (requires_grad=True)")
            
            print("OpenAI ViT-B/16 모델로 대체 성공")
            
            feature_dim = 512  # 기본 OpenAI ViT-B/16 특징 차원
            print(f"대체 모델 특징 차원: {feature_dim}")
            
            # 인코더 타입 저장
            self.encoder_type = "openclip_fallback"
        
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
        # BiomedCLIP 이미지 인코더를 통한 특징 추출
        try:
            if self.encoder_type == "transformers":
                # Transformers 스타일 인코더
                outputs = self.vision_encoder(
                    pixel_values=x,
                    output_hidden_states=False,
                    return_dict=True
                )
                
                # 풀링된 특징 벡터 추출
                if hasattr(outputs, 'pooler_output'):
                    image_features = outputs.pooler_output
                elif hasattr(outputs, 'last_hidden_state'):
                    # [CLS] 토큰 사용
                    image_features = outputs.last_hidden_state[:, 0]
                else:
                    raise ValueError("Transformers 출력에서 특징 벡터를 찾을 수 없습니다")
            
            elif self.encoder_type == "openclip" or self.encoder_type == "openclip_fallback":
                # OpenCLIP 스타일 인코더
                image_features = self.vision_encoder(x)
            
            else:
                raise ValueError(f"알 수 없는 인코더 타입: {self.encoder_type}")
                
        except Exception as e:
            print(f"특징 추출 오류: {str(e)}")
            # 오류 발생시 임시 방편으로 평균 풀링 사용
            print("백업 방법으로 평균 풀링을 사용합니다")
            image_features = torch.mean(x, dim=[2, 3])  # 간단한 평균 풀링
        
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
        torch.save(model.state_dict(), 'best_biomedclip_scar_model.pth')
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
    plt.savefig('biomedclip_scar_classifier_results.png')
    plt.close()

# 메인 실행 함수
def main():
    config = Config()  # Config 객체 생성 및 초기화
    
    print(f"BiomedCLIP 모델: {config.biomedclip_model_name}")
    print(f"디바이스: {config.device}")
    print(f"텍스트 인코더 고정: {config.lock_text_encoder}")
    
    # 데이터셋 경로
    dataset_root = ''
    
    processor = None
    preprocess = None
    
    # 먼저 test_biomedclip.py에서 동작하는 방식 시도
    try:
        # BiomedCLIP 전처리기 로드 (test_biomedclip.py에서 동작하는 방식)
        print("방법 1: open_clip.create_model_from_pretrained 함수를 사용해 프로세서 로드...")
        
        from open_clip import get_tokenizer, create_model_from_pretrained
        _, preprocess = create_model_from_pretrained(f'hf-hub:{config.biomedclip_model_name}')
        print("✅ BiomedCLIP 프로세서 로드 성공!")
        
    except Exception as e:
        print(f"방법 1 실패: {str(e)}")
        
        # 대체 방법: Transformers 라이브러리 사용
        try:
            print("방법 2: transformers CLIPProcessor 사용...")
            processor = CLIPProcessor.from_pretrained(config.biomedclip_model_name)
            print("✅ BiomedCLIP 프로세서 로드 성공!")
            
        except Exception as e2:
            print(f"방법 2 실패: {str(e2)}")
            print("기본 이미지 변환 함수를 사용합니다.")
            
            # 기본 전처리 함수 (OpenAI CLIP과 호환)
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                    std=[0.26862954, 0.26130258, 0.27577711])
            ])
    
    # 데이터셋 및 데이터로더 생성
    print("데이터셋을 생성합니다...")
    train_dataset = ScarDataset(root=dataset_root, processor=processor, transform=preprocess, is_train=True)
    val_dataset = ScarDataset(root=dataset_root, processor=processor, transform=preprocess, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    
    print(f"클래스 수: {len(train_dataset.classes)}")
    print(f"훈련 데이터셋 크기: {len(train_dataset)}")
    print(f"검증 데이터셋 크기: {len(val_dataset)}")
    
    # 모델 초기화
    print("BiomedCLIP 모델을 초기화합니다...")
    model = BiomedCLIPScarClassifier(
        model_name=config.biomedclip_model_name,
        num_classes=len(train_dataset.classes),
        lock_text_encoder=config.lock_text_encoder  # 텍스트 인코더 고정 옵션 전달
    ).to(config.device)
    
    # 단일 레이블 분류를 위한 크로스 엔트로피 손실 함수
    criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저 설정
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    print("모델 훈련을 시작합니다...")
    
    # 모델 훈련
    train_losses, val_losses, train_accs, val_accs = train_model(
        train_loader, val_loader, model, criterion, optimizer, config
    )
    
    # 결과 시각화
    plot_results(train_losses, val_losses, train_accs, val_accs)
    
    # 최고 모델 로드
    model.load_state_dict(torch.load('best_biomedclip_scar_model.pth'))
    
    # 검증 세트 평가
    print("\n검증 세트에서 평가 중...")
    results = evaluate_model(model, val_loader, config)
    
    # 결과 출력
    print("\n결과:")
    print(f"정확도: {results['accuracy']:.4f}")
    print(f"가중 정밀도: {results['weighted_precision']:.4f}")
    print(f"가중 재현율: {results['weighted_recall']:.4f}")
    print(f"가중 F1 점수: {results['weighted_f1']:.4f}")

    print("\n클래스별 결과:")
    for cls, metrics in results['class_metrics'].items():
        print(f"클래스 {cls+1} ({train_dataset.classes[cls]}):")
        print(f"  - 정확도: {metrics['accuracy']:.4f}")
        print(f"  - 정밀도: {metrics['precision']:.4f}")
        print(f"  - 재현율: {metrics['recall']:.4f}")
        print(f"  - F1 점수: {metrics['f1']:.4f}")
        print(f"  - 지원 샘플 수: {metrics['support']}")

if __name__ == '__main__':
    main()
