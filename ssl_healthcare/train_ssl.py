import os
import torch
import torch.optim as optim
from tqdm import tqdm
import argparse
from pathlib import Path

from model_ssl import create_simclr_model, NTXentLoss
from data_loader import create_ssl_dataloader


def train_ssl_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, (views) in enumerate(progress_bar):
        view1, view2 = views
        view1, view2 = view1.to(device), view2.to(device)

        batch_size = view1.size(0)
        combined = torch.cat([view1, view2], dim=0)

        optimizer.zero_grad()

        _, z = model(combined)
        z1, z2 = torch.split(z, batch_size, dim=0)

        loss = criterion(z1, z2)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_ssl(
    data_dir,
    output_dir='./checkpoints',
    backbone='resnet18',
    projection_dim=128,
    batch_size=64,
    num_epochs=100,
    lr=0.0003,
    temperature=0.5,
    num_workers=4,
    image_size=224,
    device=None
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    dataloader = create_ssl_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size
    )

    print(f"Creating SimCLR model with {backbone} backbone...")
    model = create_simclr_model(
        backbone=backbone,
        projection_dim=projection_dim,
        pretrained=False
    )
    model = model.to(device)

    criterion = NTXentLoss(temperature=temperature)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=0
    )

    print(f"\nStarting SSL pretraining for {num_epochs} epochs...")
    print(f"Total batches per epoch: {len(dataloader)}")

    best_loss = float('inf')
    loss_history = []

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_ssl_epoch(
            model=model,
            dataloader=dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        loss_history.append(avg_loss)
        print(f"Epoch {epoch}/{num_epochs} - Avg Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(output_dir, 'best_ssl_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'backbone': backbone,
                'feature_dim': model.feature_dim
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        if epoch % 10 == 0 or epoch == num_epochs:
            checkpoint_path = os.path.join(output_dir, f'ssl_model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'backbone': backbone,
                'feature_dim': model.feature_dim
            }, checkpoint_path)

    final_checkpoint = os.path.join(output_dir, 'final_ssl_model.pth')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'backbone': backbone,
        'feature_dim': model.feature_dim,
        'loss_history': loss_history
    }, final_checkpoint)

    print(f"\nTraining completed! Final model saved to {final_checkpoint}")
    print(f"Best loss: {best_loss:.4f}")

    return model, loss_history


def main():
    parser = argparse.ArgumentParser(description='Train SimCLR on medical images')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to unlabeled medical image directory')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet50'],
                        help='Backbone architecture')
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='Projection head output dimension')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for NTXent loss')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training')

    args = parser.parse_args()

    train_ssl(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        backbone=args.backbone,
        projection_dim=args.projection_dim,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        temperature=args.temperature,
        num_workers=args.num_workers,
        image_size=args.image_size
    )


if __name__ == '__main__':
    main()
