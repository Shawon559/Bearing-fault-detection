import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """Gradient Reversal Layer implementation."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer wrapper."""

    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        """Update lambda value during training."""
        self.lambda_ = lambda_


class MultiScaleFeatureExtractor(nn.Module):
    """Multi-scale feature extractor with attention-based fusion."""

    def __init__(
        self,
        envelope_dim: int,
        fft_dims: list,
        stats_dim: int = 8,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Envelope spectrum encoder
        self.envelope_encoder = nn.Sequential(
            nn.Linear(envelope_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FFT encoders (one for each scale)
        self.fft_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for dim in fft_dims
        ])

        # Statistics encoder
        self.stats_encoder = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Attention fusion
        num_branches = 1 + len(fft_dims) + 1  # envelope + FFTs + stats
        self.attention = nn.Sequential(
            nn.Linear(num_branches * hidden_dim, fusion_dim),
            nn.Tanh(),
            nn.Linear(fusion_dim, num_branches),
            nn.Softmax(dim=1),
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(num_branches * hidden_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fusion_dim = fusion_dim

    def forward(self, envelope, fft_scale1, fft_scale2, fft_scale3, stats):
        # Encode each branch
        envelope_feat = self.envelope_encoder(envelope)
        fft_feat1 = self.fft_encoders[0](fft_scale1)
        fft_feat2 = self.fft_encoders[1](fft_scale2)
        fft_feat3 = self.fft_encoders[2](fft_scale3)
        stats_feat = self.stats_encoder(stats)

        # Concatenate all features
        all_features = torch.cat([
            envelope_feat,
            fft_feat1,
            fft_feat2,
            fft_feat3,
            stats_feat
        ], dim=1)

        # Compute attention weights
        attention_weights = self.attention(all_features)

        # Apply attention to each branch
        weighted_features = []
        for i, feat in enumerate([envelope_feat, fft_feat1, fft_feat2, fft_feat3, stats_feat]):
            weight = attention_weights[:, i:i+1]
            weighted_features.append(feat * weight)

        # Concatenate weighted features
        weighted_all = torch.cat(weighted_features, dim=1)

        # Fusion
        features = self.fusion(weighted_all)

        return features


class LabelClassifier(nn.Module):
    """Label classifier (predicts bearing fault class)."""

    def __init__(self, fusion_dim: int = 128, num_classes: int = 4):
        super().__init__()
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, features):
        return self.classifier(features)


class DomainDiscriminator(nn.Module):
    """Domain discriminator (predicts source vs target domain)."""

    def __init__(self, fusion_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2),  # Binary: source vs target
        )

    def forward(self, features):
        return self.discriminator(features)


class DANN(nn.Module):
    """Domain-Adversarial Neural Network (DANN) for bearing fault diagnosis."""

    def __init__(
        self,
        envelope_dim: int,
        fft_dims: list,
        stats_dim: int = 8,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.feature_extractor = MultiScaleFeatureExtractor(
            envelope_dim=envelope_dim,
            fft_dims=fft_dims,
            stats_dim=stats_dim,
            hidden_dim=hidden_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

        self.label_classifier = LabelClassifier(
            fusion_dim=fusion_dim,
            num_classes=num_classes,
        )

        self.domain_discriminator = DomainDiscriminator(
            fusion_dim=fusion_dim,
            hidden_dim=hidden_dim,
        )

        self.grl = GradientReversalLayer(lambda_=1.0)

    def forward(self, envelope, fft_scale1, fft_scale2, fft_scale3, stats, alpha=1.0):
        # Extract features
        features = self.feature_extractor(envelope, fft_scale1, fft_scale2, fft_scale3, stats)

        # Label classification
        class_logits = self.label_classifier(features)

        # Domain classification (with gradient reversal)
        self.grl.set_lambda(alpha)
        reversed_features = self.grl(features)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits

    def predict(self, envelope, fft_scale1, fft_scale2, fft_scale3, stats):
        features = self.feature_extractor(envelope, fft_scale1, fft_scale2, fft_scale3, stats)
        class_logits = self.label_classifier(features)
        return class_logits


def create_dann_model(sample_batch, hidden_dim=64, fusion_dim=128, num_classes=4, dropout=0.3):
    """Create DANN model based on sample batch dimensions."""
    # Get dimensions from sample
    envelope_dim = sample_batch['envelope'].shape[1] if len(sample_batch['envelope'].shape) > 1 else sample_batch['envelope'].shape[0]
    fft_dims = [
        sample_batch['fft_scale1'].shape[1] if len(sample_batch['fft_scale1'].shape) > 1 else sample_batch['fft_scale1'].shape[0],
        sample_batch['fft_scale2'].shape[1] if len(sample_batch['fft_scale2'].shape) > 1 else sample_batch['fft_scale2'].shape[0],
        sample_batch['fft_scale3'].shape[1] if len(sample_batch['fft_scale3'].shape) > 1 else sample_batch['fft_scale3'].shape[0],
    ]
    stats_dim = sample_batch['stats'].shape[1] if len(sample_batch['stats'].shape) > 1 else sample_batch['stats'].shape[0]

    model = DANN(
        envelope_dim=envelope_dim,
        fft_dims=fft_dims,
        stats_dim=stats_dim,
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
        num_classes=num_classes,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nDANN Model created:")
    print(f"  Envelope dim: {envelope_dim}")
    print(f"  FFT dims: {fft_dims}")
    print(f"  Stats dim: {stats_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Fusion dim: {fusion_dim}")
    print(f"  Num classes: {num_classes}")
    print(f"  Dropout: {dropout}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    return model


if __name__ == "__main__":
    print("Testing DANN model...")

    # Create dummy batch
    batch = {
        'envelope': torch.randn(16, 4096),
        'fft_scale1': torch.randn(16, 1025),
        'fft_scale2': torch.randn(16, 4097),
        'fft_scale3': torch.randn(16, 16385),
        'stats': torch.randn(16, 8),
    }

    model = create_dann_model(batch)

    # Test forward pass
    with torch.no_grad():
        class_logits, domain_logits = model(
            batch['envelope'],
            batch['fft_scale1'],
            batch['fft_scale2'],
            batch['fft_scale3'],
            batch['stats'],
            alpha=1.0
        )

    print(f"\nForward pass test:")
    print(f"  Input batch size: {batch['envelope'].shape[0]}")
    print(f"  Class logits shape: {class_logits.shape}")
    print(f"  Domain logits shape: {domain_logits.shape}")
    print(f"  OK DANN model working correctly!")
