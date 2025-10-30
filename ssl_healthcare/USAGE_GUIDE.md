# SSL Healthcare - Usage Guide

Complete guide for running the Self-Supervised Learning pipeline for healthcare images.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Step-by-Step Tutorial](#step-by-step-tutorial)
3. [Command Reference](#command-reference)
4. [Real Dataset Examples](#real-dataset-examples)
5. [Google Colab Setup](#google-colab-setup)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

Run the complete pipeline with one command:

```bash
cd ssl_healthcare
python run_full_pipeline.py
```

This will:
1. ✅ Create synthetic medical images
2. ✅ Pretrain with SimCLR (SSL)
3. ✅ Fine-tune with SSL pretraining
4. ✅ Train baseline from scratch
5. ✅ Evaluate both models
6. ✅ Generate comparison plots

**Output structure:**
```
ssl_project/
├── data/medical_images/          # Generated images
├── checkpoints/
│   ├── ssl/                      # SSL pretrained models
│   ├── finetuned/                # Models with SSL
│   └── scratch/                  # Baseline models
├── evaluation_results/
│   ├── with_ssl/                 # Metrics, plots (SSL)
│   └── from_scratch/             # Metrics, plots (baseline)
└── plots/                        # Training curves, comparison
```

---

## Step-by-Step Tutorial

### Step 1: Create Dataset

**Option A: Use synthetic data (for testing)**

```bash
python data_loader.py
```

**Option B: Prepare your own data**

Organize your data:
```
my_data/
├── train/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── labels.csv
```

Create `labels.csv`:
```csv
image001.jpg,0
image002.jpg,1
image003.jpg,0
```

### Step 2: SSL Pretraining

Learn representations from **unlabeled** images:

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

**What happens:**
- Model sees two augmented views of each image
- Learns to identify same image despite transformations
- No labels needed!

**Expected time:**
- 500 images, 100 epochs, GPU: ~15-30 minutes
- CPU: 2-4 hours

**Output:**
- `best_ssl_model.pth` - Best checkpoint
- `final_ssl_model.pth` - Final checkpoint
- `ssl_model_epoch_X.pth` - Periodic checkpoints

### Step 3: Fine-Tune with Labels

Train classifier using pretrained encoder:

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

**Training strategy:**
1. Freeze encoder (epochs 1-25): Train only classifier
2. Unfreeze encoder (epochs 26-50): Fine-tune entire network

**Expected time:**
- 500 images, 50 epochs, GPU: ~10-20 minutes
- CPU: 1-2 hours

### Step 4: Train Baseline (Optional)

Compare with training from scratch:

```bash
python fine_tune.py \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --output_dir ./checkpoints_scratch \
    --num_classes 3 \
    --batch_size 32 \
    --num_epochs 50 \
    --lr 0.001
```

Note: No `--ssl_checkpoint` → trains from random initialization

### Step 5: Evaluate Model

Get metrics and visualizations:

```bash
python evaluate.py \
    --checkpoint ./checkpoints/best_finetuned_model.pth \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --output_dir ./evaluation_results \
    --num_classes 3 \
    --batch_size 32
```

**Outputs:**
- `results.txt` - Accuracy, F1, ROC-AUC
- `confusion_matrix.png` - Classification confusion matrix
- `roc_curves.png` - ROC curves per class
- `tsne_embeddings.png` - t-SNE visualization
- `umap_embeddings.png` - UMAP visualization

---

## Command Reference

### train_ssl.py

```bash
python train_ssl.py \
    --data_dir PATH               # Required: unlabeled images
    --output_dir ./checkpoints    # Output directory
    --backbone resnet18           # resnet18 or resnet50
    --projection_dim 128          # Projection head dimension
    --batch_size 64               # Batch size
    --num_epochs 100              # Training epochs
    --lr 0.0003                   # Learning rate
    --temperature 0.5             # Contrastive loss temperature
    --num_workers 4               # Data loading workers
    --image_size 224              # Input image size
```

### fine_tune.py

```bash
python fine_tune.py \
    --data_dir PATH               # Required: labeled images
    --label_file PATH             # Required: labels.csv
    --ssl_checkpoint PATH         # Optional: SSL pretrained model
    --output_dir ./checkpoints    # Output directory
    --backbone resnet18           # resnet18 or resnet50
    --num_classes 3               # Number of classes
    --batch_size 32               # Batch size
    --num_epochs 50               # Training epochs
    --lr 0.001                    # Learning rate
    --freeze_encoder              # Freeze encoder initially
    --unfreeze_after_epoch 25     # When to unfreeze
    --train_split 0.8             # Train/val split
    --num_workers 4               # Data loading workers
    --image_size 224              # Input image size
```

### evaluate.py

```bash
python evaluate.py \
    --checkpoint PATH             # Required: model checkpoint
    --data_dir PATH               # Required: test images
    --label_file PATH             # Required: labels.csv
    --output_dir ./results        # Output directory
    --num_classes 3               # Number of classes
    --batch_size 32               # Batch size
    --num_workers 4               # Data loading workers
    --image_size 224              # Input image size
```

---

## Real Dataset Examples

### Example 1: Chest X-Ray Classification

```bash
# Download NIH Chest X-ray dataset
# Organize into ./data/chest_xray/

# SSL Pretraining (all images, no labels)
python train_ssl.py \
    --data_dir ./data/chest_xray \
    --backbone resnet50 \
    --batch_size 64 \
    --num_epochs 200 \
    --lr 0.0003

# Fine-tune (with 10% labeled data)
python fine_tune.py \
    --data_dir ./data/chest_xray \
    --label_file ./data/chest_xray/labels_10percent.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --num_classes 2 \
    --num_epochs 50 \
    --freeze_encoder \
    --unfreeze_after_epoch 25
```

### Example 2: Skin Lesion Classification (HAM10000)

```bash
# SSL on all 10,000 images
python train_ssl.py \
    --data_dir ./data/ham10000 \
    --backbone resnet18 \
    --batch_size 128 \
    --num_epochs 150 \
    --temperature 0.3

# Fine-tune for 7 classes
python fine_tune.py \
    --data_dir ./data/ham10000 \
    --label_file ./data/ham10000/labels.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --num_classes 7 \
    --batch_size 64 \
    --num_epochs 50
```

### Example 3: Brain MRI Classification

```bash
# Grayscale images, smaller dataset
python train_ssl.py \
    --data_dir ./data/brain_mri \
    --backbone resnet18 \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 0.0005

python fine_tune.py \
    --data_dir ./data/brain_mri \
    --label_file ./data/brain_mri/labels.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --num_classes 4 \
    --batch_size 16 \
    --num_epochs 40
```

---

## Google Colab Setup

### Step 1: Setup Environment

```python
# Clone repository (replace with your repo)
!git clone https://github.com/your_username/ssl_healthcare.git
%cd ssl_healthcare

# Install dependencies
!pip install -q torch torchvision numpy matplotlib scikit-learn umap-learn tqdm

# Check GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

### Step 2: Run Pipeline

```python
# Option 1: Full pipeline
!python run_full_pipeline.py \
    --ssl_epochs 30 \
    --finetune_epochs 20 \
    --batch_size_ssl 32 \
    --batch_size_ft 16

# Option 2: Individual steps
!python data_loader.py
!python train_ssl.py --data_dir ./data/medical_images --batch_size 32 --num_epochs 30
!python fine_tune.py \
    --data_dir ./data/medical_images \
    --label_file ./data/medical_images/labels.csv \
    --ssl_checkpoint ./checkpoints/best_ssl_model.pth \
    --batch_size 16 \
    --num_epochs 20
```

### Step 3: View Results

```python
# Display images
from IPython.display import Image, display

display(Image('./ssl_project/plots/ssl_training_loss.png'))
display(Image('./ssl_project/evaluation_results/with_ssl/confusion_matrix.png'))
display(Image('./ssl_project/evaluation_results/with_ssl/tsne_embeddings.png'))

# Download results
from google.colab import files
!zip -r results.zip ssl_project/
files.download('results.zip')
```

---

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution:**
```bash
# Reduce batch size
python train_ssl.py --batch_size 16
python fine_tune.py --batch_size 8

# Use smaller backbone
python train_ssl.py --backbone resnet18

# Reduce image size
python train_ssl.py --image_size 128
```

### Issue: Training Too Slow

**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())  # Number of GPUs
```

**Solutions:**
- Reduce `--num_workers` to 2 or 0
- Reduce `--num_epochs`
- Use smaller dataset for testing

### Issue: Poor Performance

**SSL Phase:**
- Train longer: `--num_epochs 200`
- Try different temperature: `--temperature 0.3`
- Increase batch size: `--batch_size 128`

**Fine-tuning Phase:**
- More labeled data (at least 50-100 per class)
- Longer fine-tuning: `--num_epochs 100`
- Try different learning rates: `--lr 0.0005`

### Issue: Class Imbalance

Add class weights:
```python
# In fine_tune.py, modify criterion:
from torch.utils.data import WeightedRandomSampler

# Calculate weights
class_counts = [100, 200, 50]  # Your counts
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
weights = weights / weights.sum()

criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

### Issue: Labels Not Loading

Check format:
```bash
# View label file
head -5 labels.csv

# Should output:
# image001.jpg,0
# image002.jpg,1
# ...
```

Ensure:
- No header row
- Comma-separated
- Filenames match actual files
- Labels are integers starting from 0

---

## Performance Tips

### For Best SSL Results:

1. **More unlabeled data** (1000-10000+ images)
2. **Longer training** (100-500 epochs)
3. **Larger batch size** (64-256)
4. **Strong augmentations** (already included)
5. **Appropriate temperature** (0.3-0.7)

### For Best Fine-tuning Results:

1. **Start with frozen encoder**
2. **Unfreeze after initial convergence**
3. **Lower learning rate** when unfreezing
4. **Balanced dataset** (similar class counts)
5. **Sufficient labeled data** (50+ per class)

### Expected Improvements with SSL:

- **Small labeled set** (10-50 per class): 15-30% improvement
- **Medium labeled set** (100-500 per class): 5-15% improvement
- **Large labeled set** (1000+ per class): 2-8% improvement

---

## Next Steps

After running the pipeline:

1. ✅ Compare metrics: SSL vs from-scratch
2. ✅ Visualize embeddings: Check t-SNE/UMAP plots
3. ✅ Analyze errors: Look at confusion matrix
4. ✅ Try your own data: Replace synthetic dataset
5. ✅ Experiment: Adjust hyperparameters

**Questions?** Check README.md or open an issue on GitHub.

Happy learning! 🚀
