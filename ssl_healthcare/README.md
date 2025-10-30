# Self-Supervised Learning for Healthcare Image Data

A complete end-to-end implementation of Self-Supervised Learning (SSL) using SimCLR for healthcare image classification with PyTorch.

## Overview

This project demonstrates how to leverage self-supervised learning to learn meaningful representations from unlabeled medical images (X-rays, MRI, CT scans) and fine-tune for disease classification with limited labeled data.

### Why Self-Supervised Learning for Healthcare?

Healthcare imaging datasets often have:
- **Large amounts of unlabeled data** (expensive to annotate)
- **Limited labeled samples** (requires expert radiologists)
- **Class imbalance** (rare diseases)

SSL helps by:
1. Learning robust anatomical features from unlabeled images
2. Improving classification accuracy with fewer labeled samples
3. Reducing dependency on large annotated datasets

## Project Structure

```
ssl_healthcare/
├── data_loader.py          # Data loading and augmentation
├── model_ssl.py            # SimCLR model and NTXent loss
├── train_ssl.py            # Self-supervised pretraining
├── fine_tune.py            # Supervised fine-tuning
├── evaluate.py             # Metrics and visualizations
├── utils.py                # Helper functions
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## Installation

```bash
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## Quick Start

### 1. Create Sample Dataset

Generate synthetic medical images for testing:

```bash
python data_loader.py
```

This creates 500 synthetic images in `./data/medical_images/` with labels.

### 2. Self-Supervised Pretraining (Phase 1)

Train SimCLR on unlabeled images:

```bash
python train_ssl.py \
    --data_dir ./data/medical_images \
    --output_dir ./checkpoints \
    --backbone resnet18 \
    --batch_size 64 \
    --num_epochs 100 \
    --lr 0.0003 \
    --temperature 0.5
```

**Key Parameters:**
- `--data_dir`: Path to unlabeled images
- `--backbone`: ResNet architecture (`resnet18` or `resnet50`)
- `--num_epochs`: Number of pretraining epochs (100-200 recommended)
- `--temperature`: Temperature for contrastive loss (0.5 works well)

### 3. Fine-Tuning with Labels (Phase 2)

Fine-tune the pretrained encoder on labeled data:

```bash
python fine_tune.py \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --output_dir ./checkpoints \
    --num_classes 3 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.001 \
    --freeze_encoder \
    --unfreeze_after_epoch 25
```

**Training Strategy:**
- Initially freeze encoder, train only classifier head
- Unfreeze encoder after 25 epochs for end-to-end fine-tuning
- Uses smaller learning rate for better convergence

**Compare with baseline** (training from scratch):

```bash
python fine_tune.py \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --output_dir ./checkpoints \
    --num_classes 3 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.001
```

### 4. Evaluation and Visualization

Evaluate the fine-tuned model:

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_finetuned_model.pth \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --output_dir ./evaluation_results \
    --num_classes 3
```

**Outputs:**
- Accuracy, F1-score, ROC-AUC metrics
- Confusion matrix
- ROC curves
- t-SNE and UMAP embeddings visualization
- Classification report

## Architecture Details

### SimCLR Framework

SimCLR (Simple Framework for Contrastive Learning) learns representations by:

1. **Augmentation**: Apply two random augmentations to each image
   - Random crop and resize
   - Color jittering (brightness, contrast)
   - Horizontal flip
   - Rotation
   - Gaussian blur

2. **Encoder**: ResNet backbone extracts features

3. **Projection Head**: Maps features to embedding space

4. **Contrastive Loss (NT-Xent)**:
   - Maximizes agreement between augmented views of same image
   - Minimizes agreement between different images
   - Temperature parameter controls distribution sharpness

### Model Components

```python
SimCLR Model:
├── Encoder (ResNet18/50)
│   └── Feature dimension: 512 (ResNet18) or 2048 (ResNet50)
├── Projection Head
│   ├── Linear(512 → 512)
│   ├── BatchNorm + ReLU
│   └── Linear(512 → 128)
└── NTXent Loss

Classifier (Fine-tuning):
├── Frozen/Unfrozen Encoder
└── Classification Head
    ├── Linear(512 → 256) + ReLU + Dropout(0.3)
    ├── Linear(256 → 128) + ReLU + Dropout(0.3)
    └── Linear(128 → num_classes)
```

## Augmentation Pipeline

Healthcare images require specialized augmentations:

```python
SSL Augmentations:
- RandomResizedCrop (scale: 0.8-1.0)
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±15°)
- ColorJitter (brightness: 0.3, contrast: 0.3)
- GaussianBlur (p=0.3)
- Normalization (ImageNet stats)

Supervised Augmentations (milder):
- Resize to 224×224
- RandomHorizontalFlip (p=0.5)
- RandomRotation (±10°)
- ColorJitter (brightness: 0.2, contrast: 0.2)
- Normalization
```

## Using Your Own Dataset

### 1. Organize Your Data

```
your_data/
├── train/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels.csv
```

### 2. Create Labels File

Format: `filename,label`

```csv
image001.jpg,0
image002.jpg,1
image003.jpg,2
```

### 3. Train

```bash
# SSL Pretraining
python train_ssl.py --data_dir your_data/train --num_epochs 200

# Fine-tuning
python fine_tune.py \
    --data_dir your_data/train \
    --label_file your_data/labels.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --num_classes 3
```

## Google Colab Usage

Run in Colab with GPU acceleration:

```python
!git clone https://github.com/your_repo/ssl_healthcare.git
%cd ssl_healthcare
!pip install -r requirements.txt

# Check GPU
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Generate sample data
!python data_loader.py

# Train SSL (smaller batch size for Colab)
!python train_ssl.py \
    --data_dir ./data/medical_images \
    --batch_size 32 \
    --num_epochs 50

# Fine-tune
!python fine_tune.py \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --batch_size 16 \
    --num_epochs 30

# Evaluate
!python evaluate.py \
    --checkpoint ./checkpoints/best_finetuned_model.pth \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv
```

## Hyperparameter Tuning

### SSL Pretraining

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| batch_size | 64 | 32-256 | Larger = more negative pairs |
| lr | 0.0003 | 0.0001-0.001 | Use cosine schedule |
| temperature | 0.5 | 0.1-1.0 | Lower = harder negatives |
| projection_dim | 128 | 64-512 | 128 works well |
| num_epochs | 100 | 50-500 | More is better |

### Fine-Tuning

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| batch_size | 32 | 16-64 | Depends on GPU memory |
| lr | 0.001 | 0.0001-0.01 | Lower for frozen encoder |
| freeze_encoder | True | - | Freeze initially |
| unfreeze_after_epoch | 25 | 10-40 | Then fine-tune end-to-end |

## Expected Results

With SSL pretraining, you should see:
- **10-20% improvement** in accuracy vs training from scratch
- **Better convergence** with fewer labeled samples
- **More meaningful embeddings** (visible in t-SNE/UMAP)

Typical metrics on small labeled datasets:
- **Accuracy**: 75-90%
- **F1-Score**: 0.70-0.85
- **ROC-AUC**: 0.80-0.95

## Common Issues

### Out of Memory (OOM)

Reduce batch size:
```bash
python train_ssl.py --batch_size 16
```

### Slow Training

- Enable GPU: Check `torch.cuda.is_available()`
- Reduce image size: `--image_size 128`
- Use fewer workers: `--num_workers 2`

### Poor Performance

- Train SSL longer: `--num_epochs 200`
- Try different temperature: `--temperature 0.3`
- Ensure data augmentation is appropriate for your domain
- Check for class imbalance

## Citation

If you use this code, please cite:

```bibtex
@article{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={ICML},
  year={2020}
}
```

## License

MIT License - feel free to use for research and commercial purposes.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

**Happy Self-Supervised Learning!** 🚀🏥
