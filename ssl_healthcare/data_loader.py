import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path


class ContrastiveTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class MedicalImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, label_file=None, return_labels=False):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_labels = return_labels

        self.image_paths = []
        self.labels = []

        if self.root_dir.is_dir():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                self.image_paths.extend(list(self.root_dir.rglob(ext)))

        if label_file and os.path.exists(label_file):
            label_dict = {}
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) == 2:
                        label_dict[parts[0]] = int(parts[1])

            filtered_paths = []
            for img_path in self.image_paths:
                img_name = img_path.name
                if img_name in label_dict:
                    self.labels.append(label_dict[img_name])
                    filtered_paths.append(img_path)

            if filtered_paths:
                self.image_paths = filtered_paths
            else:
                self.labels = [0] * len(self.image_paths)
        else:
            self.labels = [0] * len(self.image_paths)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.return_labels:
            label = self.labels[idx]
            return image, label

        return image


def get_ssl_augmentation(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_supervised_augmentation(image_size=224, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def create_ssl_dataloader(data_dir, batch_size=64, num_workers=4, image_size=224):
    base_transform = get_ssl_augmentation(image_size)
    contrastive_transform = ContrastiveTransform(base_transform)

    dataset = MedicalImageDataset(
        root_dir=data_dir,
        transform=contrastive_transform,
        return_labels=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def create_supervised_dataloader(data_dir, label_file, batch_size=32,
                                  num_workers=4, image_size=224, train=True):
    transform = get_supervised_augmentation(image_size, train=train)

    dataset = MedicalImageDataset(
        root_dir=data_dir,
        transform=transform,
        label_file=label_file,
        return_labels=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True
    )

    return dataloader


def create_sample_dataset(output_dir='./data/medical_images', num_images=500):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Creating synthetic medical image dataset in {output_dir}...")

    for i in range(num_images):
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        if i % 2 == 0:
            center_y, center_x = 128, 128
            radius = np.random.randint(30, 60)
            y, x = np.ogrid[:256, :256]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[mask] = img[mask] * 0.5 + 128
        else:
            for _ in range(3):
                y1, x1 = np.random.randint(50, 200, 2)
                y2, x2 = y1 + np.random.randint(20, 50), x1 + np.random.randint(20, 50)
                img[y1:y2, x1:x2] = img[y1:y2, x1:x2] * 0.7 + 80

        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(output_dir, f'medical_{i:04d}.png'))

    label_file = os.path.join(output_dir, 'labels.csv')
    with open(label_file, 'w') as f:
        for i in range(num_images):
            label = i % 3
            f.write(f'medical_{i:04d}.png,{label}\n')

    print(f"Created {num_images} synthetic images and labels.csv")
    return output_dir, label_file


if __name__ == '__main__':
    data_dir, label_file = create_sample_dataset()

    print("\nTesting SSL DataLoader:")
    ssl_loader = create_ssl_dataloader(data_dir, batch_size=8)
    for batch in ssl_loader:
        view1, view2 = batch
        print(f"View 1 shape: {view1.shape}, View 2 shape: {view2.shape}")
        break

    print("\nTesting Supervised DataLoader:")
    sup_loader = create_supervised_dataloader(data_dir, label_file, batch_size=8)
    for images, labels in sup_loader:
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        print(f"Label distribution: {labels.unique(return_counts=True)}")
        break
