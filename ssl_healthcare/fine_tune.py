import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path

from model_ssl import create_simclr_model, create_classifier
from data_loader import create_supervised_dataloader


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc=f"Train Epoch {epoch}")

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total
        progress_bar.set_postfix({'loss': loss.item(), 'acc': f'{acc:.2f}%'})

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


def fine_tune(
    data_dir,
    label_file,
    ssl_checkpoint=None,
    output_dir='./checkpoints',
    backbone='resnet18',
    num_classes=3,
    batch_size=32,
    num_epochs=50,
    lr=0.001,
    freeze_encoder=True,
    unfreeze_after_epoch=25,
    train_split=0.8,
    num_workers=4,
    image_size=224,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading supervised data...")
    full_dataset = create_supervised_dataloader(
        data_dir=data_dir,
        label_file=label_file,
        batch_size=batch_size,
        num_workers=0,
        image_size=image_size,
        train=True
    ).dataset

    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    if ssl_checkpoint and os.path.exists(ssl_checkpoint):
        print(f"Loading pretrained SSL model from {ssl_checkpoint}...")
        checkpoint = torch.load(ssl_checkpoint, map_location=device)

        ssl_model = create_simclr_model(
            backbone=checkpoint.get('backbone', backbone),
            projection_dim=128,
            pretrained=False
        )
        ssl_model.load_state_dict(checkpoint['model_state_dict'])
        encoder = ssl_model.encoder

        print("Using SSL-pretrained encoder")
    else:
        print("Training from scratch (no SSL pretraining)")
        from torchvision import models
        if backbone == 'resnet18':
            encoder = models.resnet18(pretrained=False)
        elif backbone == 'resnet50':
            encoder = models.resnet50(pretrained=False)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    print(f"Creating classifier (freeze_encoder={freeze_encoder})...")
    model = create_classifier(
        encoder=encoder,
        num_classes=num_classes,
        freeze_encoder=freeze_encoder
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    print(f"\nStarting fine-tuning for {num_epochs} epochs...")

    best_val_acc = 0.0
    train_history = {'loss': [], 'acc': []}
    val_history = {'loss': [], 'acc': []}

    for epoch in range(1, num_epochs + 1):
        if freeze_encoder and epoch == unfreeze_after_epoch:
            print(f"\nUnfreezing encoder at epoch {epoch}...")
            model.unfreeze_encoder()
            optimizer = optim.Adam(model.parameters(), lr=lr/10, weight_decay=1e-4)

        train_loss, train_acc = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        val_loss, val_acc = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['acc'].append(val_acc)

        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(output_dir, 'best_finetuned_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'num_classes': num_classes
            }, checkpoint_path)
            print(f"Saved best model with val_acc: {val_acc:.2f}%")

    final_checkpoint = os.path.join(output_dir, 'final_finetuned_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'num_classes': num_classes,
        'train_history': train_history,
        'val_history': val_history
    }, final_checkpoint)

    print(f"\nFine-tuning completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final model saved to {final_checkpoint}")

    return model, train_history, val_history


def main():
    parser = argparse.ArgumentParser(description='Fine-tune classifier on medical images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to labeled medical image directory')
    parser.add_argument('--label_file', type=str, required=True,
                        help='Path to labels CSV file')
    parser.add_argument('--ssl_checkpoint', type=str, default=None,
                        help='Path to SSL pretrained checkpoint (optional)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder initially')
    parser.add_argument('--unfreeze_after_epoch', type=int, default=25,
                        help='Epoch to unfreeze encoder (if frozen)')
    parser.add_argument('--train_split', type=float, default=0.8,
                        help='Train/val split ratio')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training')

    args = parser.parse_args()

    fine_tune(
        data_dir=args.data_dir,
        label_file=args.label_file,
        ssl_checkpoint=args.ssl_checkpoint,
        output_dir=args.output_dir,
        backbone=args.backbone,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        freeze_encoder=args.freeze_encoder,
        unfreeze_after_epoch=args.unfreeze_after_epoch,
        train_split=args.train_split,
        num_workers=args.num_workers,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
