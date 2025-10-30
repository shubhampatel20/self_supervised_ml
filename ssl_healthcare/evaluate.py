import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import umap
import argparse
from tqdm import tqdm

from model_ssl import create_simclr_model, create_classifier
from data_loader import create_supervised_dataloader


def extract_features_and_labels(model, dataloader, device):
    model.eval()
    all_features = []
    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)

            if hasattr(model, 'encoder'):
                features = model.encoder(images)
            else:
                features = model(images)

            if hasattr(model, 'classifier'):
                logits = model.classifier(features)
                probs = torch.softmax(logits, dim=1)
                _, preds = logits.max(1)
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    if all_predictions:
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        return all_features, all_labels, all_predictions, all_probabilities

    return all_features, all_labels, None, None


def compute_metrics(y_true, y_pred, y_prob, num_classes):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    if num_classes == 2:
        auc = roc_auc_score(y_true, y_prob[:, 1])
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        auc = roc_auc_score(y_true_bin, y_prob, average='weighted', multi_class='ovr')

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

    return metrics


def plot_confusion_matrix(cm, output_path, class_names=None):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.colorbar()

    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(cm))]

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12)

    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def plot_roc_curves(y_true, y_prob, num_classes, output_path, class_names=None):
    plt.figure(figsize=(10, 8))

    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})', linewidth=2)
    else:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC = {auc:.3f})', linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=16, pad=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curves saved to {output_path}")


def plot_embeddings_tsne(features, labels, output_path, num_classes):
    print("Computing t-SNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    embeddings_2d = tsne.fit_transform(features)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i in range(num_classes):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[colors[i]], label=f'Class {i}',
                    alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization of Learned Embeddings', fontsize=16, pad=20)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t-SNE visualization saved to {output_path}")


def plot_embeddings_umap(features, labels, output_path, num_classes):
    print("Computing UMAP embeddings...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = reducer.fit_transform(features)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    for i in range(num_classes):
        mask = labels == i
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    c=[colors[i]], label=f'Class {i}',
                    alpha=0.6, s=50, edgecolors='k', linewidth=0.5)

    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('UMAP Visualization of Learned Embeddings', fontsize=16, pad=20)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"UMAP visualization saved to {output_path}")


def evaluate_model(
    checkpoint_path,
    data_dir,
    label_file,
    output_dir='./evaluation_results',
    num_classes=3,
    batch_size=32,
    num_workers=4,
    image_size=224,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    from torchvision import models
    encoder = models.resnet18(pretrained=False)
    model = create_classifier(encoder, num_classes=num_classes, freeze_encoder=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Loading test data...")
    test_loader = create_supervised_dataloader(
        data_dir=data_dir,
        label_file=label_file,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
        train=False
    )

    print("Extracting features and predictions...")
    features, labels, predictions, probabilities = extract_features_and_labels(
        model, test_loader, device
    )

    print("\nComputing metrics...")
    metrics = compute_metrics(labels, predictions, probabilities, num_classes)

    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score (weighted): {metrics['f1_score']:.4f}")
    print(f"ROC-AUC (weighted): {metrics['roc_auc']:.4f}")
    print("\nClassification Report:")
    print(metrics['classification_report'])

    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        os.path.join(output_dir, 'confusion_matrix.png')
    )

    plot_roc_curves(
        labels, probabilities, num_classes,
        os.path.join(output_dir, 'roc_curves.png')
    )

    plot_embeddings_tsne(
        features, labels,
        os.path.join(output_dir, 'tsne_embeddings.png'),
        num_classes
    )

    plot_embeddings_umap(
        features, labels,
        os.path.join(output_dir, 'umap_embeddings.png'),
        num_classes
    )

    results_txt = os.path.join(output_dir, 'results.txt')
    with open(results_txt, 'w') as f:
        f.write(f"Model Evaluation Results\n")
        f.write(f"========================\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Data Directory: {data_dir}\n")
        f.write(f"Label File: {label_file}\n\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score (weighted): {metrics['f1_score']:.4f}\n")
        f.write(f"ROC-AUC (weighted): {metrics['roc_auc']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(metrics['classification_report'])
        f.write("\n\nConfusion Matrix:\n")
        f.write(str(metrics['confusion_matrix']))

    print(f"\nEvaluation complete! Results saved to {output_dir}")
    print(f"Summary saved to {results_txt}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test image directory')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Directory to save evaluation results')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')

    args = parser.parse_args()

    evaluate_model(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        label_file=args.label_file,
        output_dir=args.output_dir,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
