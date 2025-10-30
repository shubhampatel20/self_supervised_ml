# SSL Healthcare - Quick Reference Card

## ⚡ One-Line Commands

```bash
# Test installation
python test_installation.py

# Run everything (demo with synthetic data)
python run_full_pipeline.py

# Generate synthetic data only
python data_loader.py

# SSL pretraining
python train_ssl.py --data_dir ./data/medical_images --num_epochs 100

# Fine-tune with SSL
python fine_tune.py --data_dir ./data --label_file ./data/labels.csv --ssl_checkpoint ./checkpoints/best_ssl_model.pth

# Fine-tune from scratch (baseline)
python fine_tune.py --data_dir ./data --label_file ./data/labels.csv

# Evaluate model
python evaluate.py --checkpoint ./checkpoints/best_finetuned_model.pth --data_dir ./data --label_file ./data/labels.csv
```

## 📋 Common Parameter Combinations

### Small Dataset (< 1000 images)
```bash
python train_ssl.py --batch_size 32 --num_epochs 100 --backbone resnet18
python fine_tune.py --batch_size 16 --num_epochs 50 --lr 0.001
```

### Medium Dataset (1000-5000 images)
```bash
python train_ssl.py --batch_size 64 --num_epochs 150 --backbone resnet18
python fine_tune.py --batch_size 32 --num_epochs 50 --lr 0.0005
```

### Large Dataset (> 5000 images)
```bash
python train_ssl.py --batch_size 128 --num_epochs 200 --backbone resnet50
python fine_tune.py --batch_size 64 --num_epochs 50 --lr 0.0003
```

### Low Memory (4GB GPU)
```bash
python train_ssl.py --batch_size 16 --backbone resnet18 --image_size 128
python fine_tune.py --batch_size 8 --image_size 128
```

### High Memory (16GB+ GPU)
```bash
python train_ssl.py --batch_size 256 --backbone resnet50
python fine_tune.py --batch_size 128
```

## 🎯 Key Parameters Explained

### train_ssl.py

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--batch_size` | 64 | 16-256 | Larger = more negatives, more memory |
| `--num_epochs` | 100 | 50-500 | More = better features |
| `--lr` | 0.0003 | 0.0001-0.001 | Higher = faster but less stable |
| `--temperature` | 0.5 | 0.1-1.0 | Lower = harder negatives |
| `--backbone` | resnet18 | resnet18/50 | Larger = more capacity |

### fine_tune.py

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `--batch_size` | 32 | 8-64 | Memory vs speed tradeoff |
| `--num_epochs` | 50 | 20-100 | Sufficient for convergence |
| `--lr` | 0.001 | 0.0001-0.01 | Start higher, will decrease |
| `--freeze_encoder` | False | - | Freeze initially recommended |
| `--unfreeze_after_epoch` | 25 | 10-40 | Unfreeze at halfway point |

## 🔍 Debugging Commands

```bash
# Check PyTorch and GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check data loading
python -c "from data_loader import create_sample_dataset; create_sample_dataset(num_images=10)"

# Test model creation
python -c "from model_ssl import create_simclr_model; model = create_simclr_model(); print('Model OK')"

# Quick training test (1 epoch)
python train_ssl.py --data_dir ./data/medical_images --num_epochs 1 --batch_size 8

# Check GPU memory
nvidia-smi
```

## 📊 Expected Training Times

**500 images, ResNet18:**
| Setup | SSL (100 epochs) | Fine-tune (50 epochs) | Total |
|-------|-----------------|----------------------|-------|
| RTX 3080 | 15-20 min | 8-10 min | ~30 min |
| RTX 2060 | 25-35 min | 12-15 min | ~50 min |
| CPU | 2-3 hours | 45-60 min | ~4 hours |

**5000 images, ResNet50:**
| Setup | SSL (200 epochs) | Fine-tune (50 epochs) | Total |
|-------|-----------------|----------------------|-------|
| RTX 3080 | 2-3 hours | 25-30 min | ~3.5 hours |
| RTX 2060 | 4-6 hours | 45-60 min | ~7 hours |

## 🎨 Output Files Reference

```
ssl_project/
├── checkpoints/
│   ├── ssl/
│   │   ├── best_ssl_model.pth          ← Load this for fine-tuning
│   │   ├── final_ssl_model.pth
│   │   └── ssl_model_epoch_X.pth
│   ├── finetuned/
│   │   ├── best_finetuned_model.pth    ← Load this for evaluation
│   │   └── final_finetuned_model.pth
│   └── scratch/
│       └── best_finetuned_model.pth    ← Baseline comparison
│
└── evaluation_results/
    ├── with_ssl/
    │   ├── results.txt                  ← Read metrics here
    │   ├── confusion_matrix.png
    │   ├── roc_curves.png
    │   ├── tsne_embeddings.png
    │   └── umap_embeddings.png
    └── plots/
        ├── ssl_training_loss.png
        ├── training_history.png
        └── model_comparison.png         ← SSL vs Baseline
```

## 💡 Performance Tips

### For Better SSL Pretraining:
1. ✅ Use more unlabeled data (1000-10000+)
2. ✅ Train longer (150-300 epochs)
3. ✅ Larger batch size if possible (64-256)
4. ✅ Experiment with temperature (0.3-0.7)

### For Better Fine-Tuning:
1. ✅ Start with frozen encoder
2. ✅ Unfreeze after ~50% of epochs
3. ✅ Use lower LR when unfreezing (lr/10)
4. ✅ Balance your dataset (similar class counts)
5. ✅ More labeled data per class (50+ minimum)

### Common Mistakes to Avoid:
- ❌ Too small batch size for SSL (< 16)
- ❌ Training SSL for too few epochs (< 50)
- ❌ Not freezing encoder initially
- ❌ Too high learning rate when unfreezing
- ❌ Imbalanced classes without weighting

## 🔧 Modify Augmentations

Edit `data_loader.py`:

```python
# For more aggressive augmentation
transforms.RandomRotation(degrees=30)      # More rotation
transforms.ColorJitter(brightness=0.4)     # More jitter

# For less aggressive augmentation
transforms.RandomRotation(degrees=5)       # Less rotation
transforms.ColorJitter(brightness=0.1)     # Less jitter

# Add new augmentations
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
transforms.RandomPerspective(distortion_scale=0.2)
```

## 📈 Monitoring Training

Watch for these signs:

### SSL Training (Good):
- ✅ Loss decreases steadily
- ✅ Loss: starts ~3-5, ends ~1-2
- ✅ No sudden spikes

### SSL Training (Bad):
- ❌ Loss stays flat (increase LR or temperature)
- ❌ Loss increases (decrease LR)
- ❌ NaN loss (decrease LR significantly)

### Fine-Tuning (Good):
- ✅ Train/val accuracy both increase
- ✅ Gap between train/val stays small (<10%)
- ✅ Validation accuracy > 70% (small dataset)

### Fine-Tuning (Bad):
- ❌ Large train/val gap (>20%) → overfitting, reduce capacity
- ❌ Both accuracies low → train longer or increase capacity
- ❌ Validation plateaus early → try unfreezing encoder

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| OOM Error | Reduce `--batch_size` to 16 or 8 |
| "No module named X" | Run `pip install -r requirements.txt` |
| Slow training | Check GPU available: `torch.cuda.is_available()` |
| Poor accuracy | Train SSL longer: `--num_epochs 200` |
| NaN loss | Reduce learning rate: `--lr 0.0001` |
| Can't find files | Use absolute paths or check working directory |
| No improvement | Check data quality and augmentation strength |

## 📚 Where to Find More Info

| Topic | File | Section |
|-------|------|---------|
| Architecture details | README.md | Architecture Details |
| Step-by-step tutorial | USAGE_GUIDE.md | Step-by-Step Tutorial |
| Real dataset examples | USAGE_GUIDE.md | Real Dataset Examples |
| Hyperparameter tuning | README.md | Hyperparameter Tuning |
| Google Colab setup | USAGE_GUIDE.md | Google Colab Setup |
| Project overview | PROJECT_SUMMARY.md | Entire file |

## 🎯 Workflow Cheat Sheet

```
1. Setup:
   cd ssl_healthcare
   pip install -r requirements.txt
   python test_installation.py

2. Prepare data:
   # Option A: Use synthetic
   python data_loader.py

   # Option B: Use your own
   # Place images in folder, create labels.csv

3. SSL pretrain:
   python train_ssl.py --data_dir DATA_DIR --num_epochs 100

4. Fine-tune:
   python fine_tune.py \
     --data_dir DATA_DIR \
     --label_file LABELS.csv \
     --ssl_checkpoint checkpoints/best_ssl_model.pth

5. Evaluate:
   python evaluate.py \
     --checkpoint checkpoints/best_finetuned_model.pth \
     --data_dir DATA_DIR \
     --label_file LABELS.csv

6. Compare:
   # View plots in evaluation_results/
   # Check model_comparison.png
```

## 🚀 Advanced Usage

### Use Multiple GPUs
```python
# Modify train_ssl.py or fine_tune.py
model = nn.DataParallel(model)
```

### Save Embeddings
```python
from evaluate import extract_features_and_labels
features, labels, _, _ = extract_features_and_labels(model, dataloader, device)
np.save('embeddings.npy', features)
```

### Resume Training
```python
# In train_ssl.py or fine_tune.py, add:
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

**Print this card for quick reference!** 📋
