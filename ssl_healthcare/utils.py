import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_training_plots(loss_history, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, linewidth=2, color='#2E86AB')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('SSL Training Loss Over Time', fontsize=16, pad=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training plot saved to {output_path}")


def save_finetuning_plots(train_history, val_history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(train_history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(val_history['loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, pad=15)
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_history['acc'], label='Train Accuracy', linewidth=2, color='#2E86AB')
    axes[1].plot(val_history['acc'], label='Val Accuracy', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, pad=15)
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'training_history.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Fine-tuning plots saved to {output_path}")


def print_model_summary(model, input_size=(1, 3, 224, 224)):
    print("\nModel Summary:")
    print("=" * 70)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    try:
        x = torch.randn(input_size)
        if hasattr(model, 'encoder'):
            features = model.encoder(x)
            print(f"\nEncoder output shape: {features.shape}")

        if hasattr(model, 'classifier'):
            output = model(x)
            print(f"Classifier output shape: {output.shape}")
        elif hasattr(model, 'projection_head'):
            h, z = model(x)
            print(f"Feature shape: {h.shape}")
            print(f"Projection shape: {z.shape}")
        else:
            output = model(x)
            print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Could not compute output shapes: {e}")

    print("=" * 70)


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    return device


def create_directory_structure(base_dir='./ssl_project'):
    dirs = [
        'data/medical_images',
        'checkpoints/ssl',
        'checkpoints/finetuned',
        'evaluation_results',
        'plots'
    ]

    for dir_path in dirs:
        full_path = os.path.join(base_dir, dir_path)
        os.makedirs(full_path, exist_ok=True)

    print(f"Created directory structure in {base_dir}")
    return base_dir


def load_checkpoint(checkpoint_path, model, optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', None)

    print(f"Loaded checkpoint from epoch {epoch}")
    if loss is not None:
        print(f"Checkpoint loss: {loss:.4f}")

    return model, optimizer, epoch


def compare_models(ssl_results, scratch_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metrics = ['accuracy', 'f1_score', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.35

    ssl_values = [ssl_results[m] for m in metrics]
    scratch_values = [scratch_results[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, ssl_values, width, label='With SSL Pretraining', color='#2E86AB')
    bars2 = ax.bar(x + width/2, scratch_values, width, label='From Scratch', color='#A23B72')

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison: SSL Pretraining vs From Scratch', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1 Score', 'ROC-AUC'])
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


if __name__ == '__main__':
    print("Testing utility functions...")

    set_seed(42)
    print("Random seed set to 42")

    device = get_device()

    base_dir = create_directory_structure('./test_ssl_project')
    print(f"\nDirectory structure created at {base_dir}")

    early_stopping = EarlyStopping(patience=5, mode='min')
    losses = [1.0, 0.8, 0.7, 0.75, 0.74, 0.73, 0.72, 0.71]
    for i, loss in enumerate(losses):
        if early_stopping(loss):
            print(f"\nEarly stopping triggered at epoch {i+1}")
            break
    else:
        print("\nEarly stopping not triggered")

    print("\nUtility functions test complete!")
