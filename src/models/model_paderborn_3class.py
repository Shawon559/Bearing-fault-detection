"""
Multi-Scale Attention Fusion Network for 3-Class Bearing Fault Detection.

This module implements a deep learning architecture that combines multiple
signal representations (envelope spectrum, multi-scale FFT, and time-domain statistics)
with attention-based fusion for robust bearing fault classification.

Performance: 99.26% test accuracy on Paderborn dataset
"""

import torch
import torch.nn as nn


class MultiScaleAttentionFusion3Class(nn.Module):
    """
    Multi-Scale Attention Fusion Network for bearing fault diagnosis.

    Architecture combines:
    - Envelope spectrum analysis (Hilbert transform + FFT)
    - Multi-scale FFT (3 scales: 2048, 8192, 32768 points)
    - Time-domain statistical features
    - Attention-based feature fusion

    Args:
        envelope_dim: Dimension of envelope spectrum features
        fft_dims: List of dimensions for each FFT scale
        stats_dim: Dimension of statistical features (default: 8)
        hidden_dim: Hidden layer dimension (default: 64)
        fusion_dim: Fusion layer dimension (default: 128)
        num_classes: Number of output classes (default: 3)
        dropout: Dropout probability (default: 0.3)

    Output Classes:
        0: Healthy bearing
        1: Inner race fault
        2: Outer race fault
    """

    def __init__(
        self,
        envelope_dim: int,
        fft_dims: list,
        stats_dim: int = 8,
        hidden_dim: int = 64,
        fusion_dim: int = 128,
        num_classes: int = 3,  # 3 classes: Healthy, Inner, Outer
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

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, envelope, fft_scale1, fft_scale2, fft_scale3, stats):
        # Encode each branch
        envelope_feat = self.envelope_encoder(envelope)  # (batch, hidden_dim)
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
        ], dim=1)  # (batch, 5 * hidden_dim)

        # Compute attention weights
        attention_weights = self.attention(all_features)  # (batch, 5)

        # Apply attention to each branch
        weighted_features = []
        for i, feat in enumerate([envelope_feat, fft_feat1, fft_feat2, fft_feat3, stats_feat]):
            weight = attention_weights[:, i:i+1]  # (batch, 1)
            weighted_features.append(feat * weight)

        # Concatenate weighted features
        weighted_all = torch.cat(weighted_features, dim=1)  # (batch, 5 * hidden_dim)

        # Fusion
        fused = self.fusion(weighted_all)  # (batch, fusion_dim)

        # Classification
        logits = self.classifier(fused)  # (batch, num_classes)

        return logits


def create_model_3class(sample_batch, hidden_dim=64, fusion_dim=128, num_classes=3, dropout=0.3):
    # Get dimensions from sample
    envelope_dim = sample_batch['envelope'].shape[1] if len(sample_batch['envelope'].shape) > 1 else sample_batch['envelope'].shape[0]
    fft_dims = [
        sample_batch['fft_scale1'].shape[1] if len(sample_batch['fft_scale1'].shape) > 1 else sample_batch['fft_scale1'].shape[0],
        sample_batch['fft_scale2'].shape[1] if len(sample_batch['fft_scale2'].shape) > 1 else sample_batch['fft_scale2'].shape[0],
        sample_batch['fft_scale3'].shape[1] if len(sample_batch['fft_scale3'].shape) > 1 else sample_batch['fft_scale3'].shape[0],
    ]
    stats_dim = sample_batch['stats'].shape[1] if len(sample_batch['stats'].shape) > 1 else sample_batch['stats'].shape[0]

    model = MultiScaleAttentionFusion3Class(
        envelope_dim=envelope_dim,
        fft_dims=fft_dims,
        stats_dim=stats_dim,
        hidden_dim=hidden_dim,
        fusion_dim=fusion_dim,
        num_classes=num_classes,
        dropout=dropout,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel created:")
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
