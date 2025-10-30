import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.projection(x)


class SimCLR(nn.Module):
    def __init__(self, backbone='resnet18', projection_dim=128, pretrained=False):
        super(SimCLR, self).__init__()

        if backbone == 'resnet18':
            self.encoder = models.resnet18(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        elif backbone == 'resnet50':
            self.encoder = models.resnet50(pretrained=pretrained)
            feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.projection_head = ProjectionHead(
            input_dim=feature_dim,
            hidden_dim=feature_dim,
            output_dim=projection_dim
        )

        self.feature_dim = feature_dim

    def forward(self, x):
        h = self.encoder(x)
        z = self.projection_head(h)
        return h, z

    def get_features(self, x):
        with torch.no_grad():
            h = self.encoder(x)
        return h


class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]

        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=2
        )

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float()
        mask = mask.to(representations.device)

        numerator = torch.exp(positives / self.temperature)
        denominator = (mask * torch.exp(similarity_matrix / self.temperature)).sum(dim=1)

        loss = -torch.log(numerator / denominator)
        return loss.mean()


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes=3, feature_dim=512, freeze_encoder=True):
        super(Classifier, self).__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.set_grad_enabled(not self._is_encoder_frozen()):
            features = self.encoder(x)
        logits = self.classifier(features)
        return logits

    def _is_encoder_frozen(self):
        return not next(self.encoder.parameters()).requires_grad

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True


def create_simclr_model(backbone='resnet18', projection_dim=128, pretrained=False):
    model = SimCLR(
        backbone=backbone,
        projection_dim=projection_dim,
        pretrained=pretrained
    )
    return model


def create_classifier(encoder, num_classes=3, freeze_encoder=True):
    feature_dim = encoder.fc.in_features if hasattr(encoder, 'fc') else 512

    if hasattr(encoder, 'fc'):
        encoder.fc = nn.Identity()

    classifier = Classifier(
        encoder=encoder,
        num_classes=num_classes,
        feature_dim=feature_dim,
        freeze_encoder=freeze_encoder
    )
    return classifier


if __name__ == '__main__':
    print("Testing SimCLR Model:")
    model = create_simclr_model(backbone='resnet18', projection_dim=128)
    x = torch.randn(4, 3, 224, 224)
    h, z = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Feature shape: {h.shape}")
    print(f"Projection shape: {z.shape}")

    print("\nTesting NTXent Loss:")
    criterion = NTXentLoss(temperature=0.5)
    z1 = torch.randn(8, 128)
    z2 = torch.randn(8, 128)
    loss = criterion(z1, z2)
    print(f"Loss: {loss.item():.4f}")

    print("\nTesting Classifier:")
    classifier = create_classifier(model.encoder, num_classes=3, freeze_encoder=True)
    x = torch.randn(4, 3, 224, 224)
    logits = classifier(x)
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")

    total_params = sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
