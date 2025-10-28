import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def default_loader(path):
    return Image.open(path).convert('RGB')

class PathMNISTDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, loader=default_loader):
        """
        Args:
            root (str): 이미지 파일들이 있는 폴더 경로.
            transform (callable, optional): 이미지에 적용할 전처리 함수.
            target_transform (callable, optional): 라벨에 적용할 변환 함수.
            loader (callable, optional): 이미지를 로드할 함수 (기본은 PIL.Image.open).
        """

        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
                
        samples = []
        for fname in os.listdir(root):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                if '-' in fname:
                    path = os.path.join(root, fname)
                    # 파일 이름 형식: '{class name}-{id}.tif'
                    class_name = fname.split('-')[0]
                    samples.append((path, class_name))
        if len(samples) == 0:
            raise RuntimeError(f"Found 0 files in {root}. Supported extensions are: {','.join(IMG_EXTENSIONS)}")
        
        classes = sorted({s[1] for s in samples})
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        # samples의 라벨을 인덱스로 변환
        self.imgs = [(path, class_to_idx[label]) for (path, label) in samples]
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        path, target = self.imgs[index]
        imgs = self.loader(path)
        if self.transform is not None:
            imgs = self.transform(imgs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return imgs, target

    def __len__(self):
        return len(self.imgs)


class ScarDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None, additional_labels_transform=None, loader=default_loader, is_train=True, tokenizer=None, prompt_template_settion=None):
        """
        Args:
            root (str): 이미지 파일들이 저장된 폴더 경로.
            label_json (str): label_info.json 파일 경로.
            transform (callable, optional): 이미지에 적용할 전처리 함수.
            target_transform (callable, optional): 라벨에 적용할 변환 함수.
            loader (callable, optional): 이미지를 로드할 함수 (기본: PIL.Image.open).
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.additional_labels_transform = additional_labels_transform
        self.loader = loader
        self.is_train = is_train
        self.tokenizer = tokenizer

        self.bounding_box_json = os.path.join(root, "bounding_box.json") 
        label_json = os.path.join(root, "label_info.json")
        # label_info.json 파일 로드
        with open(label_json, 'r') as f:
            label_info = json.load(f)
        # self.label_info = label_info

        self.classes = ['1. Others', '2. Hypertrophic scar', '3. Keloid scar']
        self.num_classes = len(self.classes)
        self.class_to_idx = {i+1:i for i in range(self.num_classes)}

        self.imgs, self.labels = self.load_data(label_info)

        self.Width_label = ["Linear", "Widened", "Linear bulging"]
        self.Color_label = ["Normal", "Pink", "Red", "Purple"]
        self.Pigmentation_label = ["Normal", "Pigmented", "Hypopigmented"]
        self.Surface_label = ["Flat", "Hypertrophic", "Keloid", "Atrophic"]
        self.Irregular_color_label = ["no", "mild", "moderate", "severe"]
        self.Irregular_height_label = ["no", "mild", "moderate", "severe"]
        

    def load_data(self, label_info):
        def process_class_label(x):
            x = str(x).strip()
            if ',' in x:
                try:
                    return [self.class_to_idx[int(item.strip())] for item in x.split(',')]
                except Exception as e:
                    raise ValueError(f"Invalid multiple class label: {x}") from e
            else:
                if not x.isdigit():
                    x = int(x)
                # label_info에 정의된 문자열과 매핑된 경우 우선 사용
                if x in self.class_to_idx:
                    return [self.class_to_idx[x]]
                else:
                    try:
                        return [self.class_to_idx[int(x.split('.')[0])]]
                    except Exception as e:
                        raise ValueError(f"Invalid class label: {x}") from e

        # 추가 열(Width, Color, Pigmentation, Surface, Irregular_color, Irregular_height)에 대한 매핑 생성
        additional_columns = ["Width", "Color", "Pigmentation", "Surface", "Irregular_color", "Irregular_height"]
        self.additional_mappings = {}
        for col in additional_columns:
            if col in label_info:
                self.additional_mappings[col] = {val.lower(): idx for idx, val in enumerate(label_info[col])}
        
        # CSV 파일 읽기 및 "Use"가 "yes"인 행만 필터링
        # csv_file = os.path.join(self.root, "scar_label_250109.csv")
        if self.is_train:
            csv_file = os.path.join("../datasets/updated_scar_label_250218_train_augmented_human_simple.csv")
        else:
            csv_file = os.path.join("../datasets/updated_scar_label_250218_val_augmented_human_simple.csv")
        df = pd.read_csv(csv_file)
        df = df[df["Use"] == "yes"].reset_index(drop=True)
        
        # label 없는 정보 제거
        df = df.dropna(subset=["Width", "Color", "Pigmentation", "Surface", "Irregular_color", "Irregular_height"])
        
        df["img_path"] = df["Name"].astype(str).str.strip().apply(lambda x: os.path.join(self.root, x))        
        df["class_label"] = df["Class"].astype(str).str.strip().apply(process_class_label)

        # 추가 열들에 대해 벡터화된 매핑 적용
        for col in additional_columns:
            if col in df.columns and col in self.additional_mappings:
                mapping = self.additional_mappings[col]
                df[col + "_mapped"] = df[col].astype(str).str.strip().str.lower().apply(lambda x: mapping.get(x, -1))
            else:
                df[col + "_mapped"] = -1
        
        # 추가 열들을 하나의 dict로 결합
        def combine_additional(row):
            return {col: row[col + "_mapped"] for col in additional_columns}
        df["additional_labels"] = df.apply(combine_additional, axis=1)
        
        self.imgs = df["img_path"].tolist()
        self.labels = df[["class_label", "additional_labels"]].to_records(index=False).tolist()

        return self.imgs, self.labels
        
    def dict_to_tensor(self, additional_labels, category_size=[3,4,3,4,4,4]):
        total_elements = sum(category_size)
        tensor = torch.zeros(total_elements, dtype=torch.float32)
        current_pos = 0
    
        categories = list(additional_labels.keys())
    
        for i, category in enumerate(categories):
            if i < len(category_size):  # 카테고리 사이즈 리스트 범위 내에 있는지 확인
                size = category_size[i]
                selected_idx = additional_labels[category]
                
                # 인덱스가 유효한지 확인
                if 0 <= selected_idx < size:
                    # 해당 위치에 1 설정
                    tensor[current_pos + selected_idx] = 1
                
                # 다음 카테고리의 시작 위치 업데이트
                current_pos += size
        
        return tensor
    
    def get_class_words(self, class_label):
        class_list = ["Others","Hypertrophic scar","Keloid scar"]
        if len(class_label) == 1:
            class_word = class_list[class_label[0]]
        else:
            class_word = " , ".join([class_list[label] for label in class_label])
        return class_word
    
    def parse_points(self, box):
        """좌표를 좌상단(x1, y1)과 우하단(x2, y2) 순서로 정렬."""
        x1, y1 = box[0]
        x2, y2 = box[1]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        return int(x_min), int(y_min), int(x_max), int(y_max)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path, (class_label, additional_labels) = self.imgs[index], self.labels[index]
        image = self.loader(img_path)
        
        try:
            with open(self.bounding_box_json, 'r') as f:
                bounding_box_data = json.load(f)
                
                for shape in bounding_box_data["shapes"]:
                    if shape["label"] == "scar":
                        #이미지 path로 bounding box 정보 찾기 필요
                        x_min, y_min, x_max, y_max = self.parse_points(shape["points"])
                        width = x_max - x_min
                        height = y_max - y_min

                        image = image[y_min:y_max, x_min:x_max]
            #print(f"Warning: Bounding box file {self.bounding_box_json} not found. Using the entire image.")
        except FileNotFoundError:
            pass
        
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            class_label = self.target_transform(class_label)
        if self.additional_labels_transform is not None:
            additional_labels = self.additional_labels_transform(additional_labels)

        # multi-hot encoding
        label_tensor = torch.zeros(self.num_classes, dtype=torch.float32)
        label_tensor[class_label] = 1.0

        additional_tensor = self.dict_to_tensor(additional_labels)
        label_Width = self.Width_label[additional_labels["Width"]]
        label_Color = self.Color_label[additional_labels["Color"]]
        label_Pigmentation = self.Pigmentation_label[additional_labels["Pigmentation"]]
        label_Surface = self.Surface_label[additional_labels["Surface"]]
        label_Irregular_color = self.Irregular_color_label[additional_labels["Irregular_color"]]
        label_Irregular_height = self.Irregular_height_label[additional_labels["Irregular_height"]]
        class_word = self.get_class_words(class_label)
        text_prompt_1 = f"A {class_word} with a {label_Width} width, exhibiting a {label_Color} color and {label_Pigmentation} pigmentation. It has a {label_Surface} surface, with {label_Irregular_color} irregular color and {label_Irregular_height} irregular height."
        text_prompt_2 = f"This is an image of {class_word} with a {label_Width} width, exhibiting a {label_Color} color and {label_Pigmentation} pigmentation. It has a {label_Surface} surface, with {label_Irregular_color} irregular color and {label_Irregular_height} irregular height."
        text_prompt_3 = f"{class_word} with a {label_Width} width, exhibiting a {label_Color} color and {label_Pigmentation} pigmentation. It has a {label_Surface} surface, with {label_Irregular_color} irregular color and {label_Irregular_height} irregular height presented in image"
        text_prompt_4 = f"a photo of {class_word} with a {label_Width} width, exhibiting a {label_Color} color and {label_Pigmentation} pigmentation. It has a {label_Surface} surface, with {label_Irregular_color} irregular color and {label_Irregular_height} irregular height."
        text_prompt_itemization = f"A {class_word} photo, Width: {label_Width} width, Color: {label_Color} Color, Pigmentation: {label_Pigmentation} Pigmentation, Surface: {label_Surface} Surface, Irregular color: {label_Irregular_color} Irregular Color, Irregular height: {label_Irregular_height} Irregular Height."
        text_prompt_token_1 = self.tokenizer(text_prompt_1)[0]
        text_prompt_token_2 = self.tokenizer(text_prompt_2)[0]
        text_prompt_token_3 = self.tokenizer(text_prompt_3)[0]
        text_prompt_token_4 = self.tokenizer(text_prompt_4)[0]
        text_prompt_token_itemization = self.tokenizer(text_prompt_itemization)[0]
        text_prompt_token = [text_prompt_token_1, text_prompt_token_2, text_prompt_token_3, text_prompt_token_4, text_prompt_token_itemization]
        return image, label_tensor, additional_tensor, text_prompt_token, class_word

