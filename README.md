# Medical Image Classification Project

A comprehensive framework for training and evaluating deep learning models on medical scar image classification tasks, with support for CLIP-based models (BiomedCLIP, OpenCLIP) and traditional CNN architectures (ResNet50).

## Project Overview

This project provides tools for:
- Training and testing medical image classification models
- Visualizing model predictions and feature distributions
- Comparing different model architectures (CLIP variants and ResNet50)
- Analyzing class-specific features and prediction confidence

## Project Structure

```
.
├── src/                          # Source code modules
│   ├── open_clip/               # OpenCLIP implementation
│   ├── open_clip_train/         # OpenCLIP training utilities
│   └── others/                  # Additional utilities
├── datasets/                     # Dataset loaders and preprocessing
├── scripts/                      # Utility scripts
├── *_baseline.py                # Baseline model implementations
├── visualize_*.py               # Visualization tools
└── *.sh                         # Training and testing scripts
```

## Models

### 1. BiomedCLIP
Medical domain-specific CLIP model fine-tuned on biomedical data.
- **File**: `biomedclip_baseline.py`
- **Model**: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Features**: Vision-language pre-training for medical images

### 2. OpenCLIP
General-purpose CLIP models with various backbones.
- **File**: `clip_baseline.py`
- **Backbone**: ViT-B-32 (configurable)
- **Pretrained weights**: laion400m_e32

### 3. ResNet50
Traditional CNN baseline for comparison.
- **File**: `resnet50_baseline.py`
- **Architecture**: ResNet50 with custom classification head
- **Features**: Transfer learning from ImageNet

## Quick Start

### Prerequisites

```bash
# Install required packages
pip install torch torchvision transformers open-clip-torch
pip install pandas numpy pillow scikit-learn matplotlib tqdm
pip install wandb  # For experiment tracking (optional)
```

### Training

#### Train BiomedCLIP
```bash
python biomedclip_baseline.py
```

Configuration options in the script:
- `batch_size`: Batch size for training (default: 4)
- `learning_rate`: Learning rate (default: 5e-6)
- `num_epochs`: Number of training epochs (default: 100)
- `lock_text_encoder`: Freeze text encoder weights (True/False)

#### Train OpenCLIP
```bash
# With pretrained weights
bash scar_openclip_pretrain.sh

# With vision encoder frozen
bash scar_openclip_train_vision_freeze.sh

# Custom training
cd src
python -m others.main_other \
    --batch-size 16 \
    --workers 4 \
    --model ViT-B-32 \
    --pretrained laion400m_e32 \
    --train-data /path/to/train.csv \
    --val-data /path/to/val.csv
```

#### Train ResNet50
```bash
python resnet50_baseline.py
```

### Testing

```bash
# Test with PathMNIST dataset
bash pathmnist_test.sh

# Or run directly
cd src
python -m others.main_other \
    --batch-size 1 \
    --val-data /path/to/test.csv \
    --model ViT-B-32 \
    --pretrained laion400m_e32 \
    --save-embed
```

## Visualization Tools

### 1. Class Feature Distribution
Visualizes the distribution of features for each class.

```bash
python visualize_class_feature_distribution.py \
    --model-path /path/to/model.pth \
    --csv-path /path/to/data.csv \
    --img-dir /path/to/images \
    --output-dir ./results
```

### 2. Maximum Probability Heatmap
Generates heatmaps showing prediction confidence across samples.

```bash
python visualize_max_prob_heatmap.py \
    --model-path /path/to/model.pth \
    --csv-path /path/to/data.csv \
    --img-dir /path/to/images \
    --output-dir ./results
```

### 3. Tag-Class Distribution
Analyzes the distribution of predicted tags across different classes.

```bash
python visualize_tag_class_distribution.py \
    --model-path /path/to/model.pth \
    --csv-path /path/to/data.csv \
    --img-dir /path/to/images \
    --output-dir ./results
```

### Run All Visualizations
Use the convenience script to run all visualization tools at once:

```bash
# Edit run_visualize_tags.sh to set paths:
# - OUTPUT_DIR: Where to save results
# - MODEL_PATH: Path to trained model
# - CSV_PATH: Path to data CSV
# - IMG_DIR: Path to image directory

bash run_visualize_tags.sh
```

## Configuration

### Model Configuration
Each baseline script contains a `Config` class with key parameters:

```python
class Config:
    batch_size = 4
    num_workers = 4
    learning_rate = 5e-6
    weight_decay = 1e-4
    num_epochs = 100
    patience = 5  # Early stopping
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 224
```

### Training Scripts
Shell scripts in the root directory can be customized:
- `pathmnist_train.sh`: PathMNIST training configuration
- `pathmnist_test.sh`: PathMNIST testing configuration
- `scar_Biomedclip_pretrain.sh`: BiomedCLIP pretraining
- `scar_openclip_pretrain.sh`: OpenCLIP pretraining
- `scar_openclip_train_vision_freeze.sh`: OpenCLIP with frozen vision encoder

## Dataset Format

The project expects CSV files with the following structure:
- Image paths/filenames
- Labels/class indices
- Optional: Additional metadata

Images should be organized in directories accessible via the paths in the CSV.

## Experiment Tracking

The project supports Weights & Biases (wandb) for experiment tracking:

```bash
# Training with wandb
python -m others.main_other \
    --report-to wandb \
    --wandb-project-name your-project-name \
    ...
```

## Output

### Training Output
- Model checkpoints saved periodically
- Training metrics logged to console and/or wandb
- Best model saved based on validation performance

### Visualization Output
All visualization scripts save results to the specified output directory:
- PNG/PDF plots of feature distributions
- Heatmaps of prediction confidence
- Class-wise analysis charts

## Common Tasks

### Evaluate Model Performance
```python
python resnet50_baseline.py  # Contains evaluation code
# Check accuracy, precision, recall, F1-score
```

### Compare Models
Train each model and compare:
1. Training curves
2. Validation accuracy
3. Class-wise performance
4. Inference time

### Fine-tune Pretrained Model
```bash
# Load pretrained weights and continue training
python biomedclip_baseline.py  # Modify to load checkpoint
```

### Generate Embeddings
```bash
cd src
python -m others.main_other \
    --val-data /path/to/data.csv \
    --model ViT-B-32 \
    --save-embed
```

## Troubleshooting

### Memory Issues
- Reduce `batch_size` in Config
- Use gradient accumulation
- Enable mixed precision training (`--precision amp`)

### Slow Training
- Increase `num_workers` for data loading
- Use smaller image size
- Enable GPU acceleration

### Visualization Errors
- Ensure Korean font support if using Korean labels (uncomment font installation in `run_visualize_tags.sh`)
- Check file paths are correct
- Verify model checkpoint format

## Development

### Adding New Models
1. Create new baseline file (e.g., `mymodel_baseline.py`)
2. Implement dataset class and training loop
3. Create corresponding shell script for training

### Custom Datasets
1. Modify dataset classes in baseline scripts
2. Update CSV loading logic
3. Adjust preprocessing/augmentation

### New Visualizations
Follow the pattern in existing `visualize_*.py` scripts:
1. Load model and data
2. Generate predictions/features
3. Create matplotlib visualizations
4. Save to output directory

## License

Please check the licenses of the respective model implementations:
- BiomedCLIP: Microsoft Research
- OpenCLIP: LAION
- ResNet: torchvision

## References

- BiomedCLIP: [microsoft/BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- OpenCLIP: [open-clip](https://github.com/mlfoundations/open_clip)
- ResNet: [Deep Residual Learning](https://arxiv.org/abs/1512.03385)
