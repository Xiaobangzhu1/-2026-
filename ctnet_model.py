import torch
import torch.nn as nn


class ConvPatchEmbedding(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_size: int = 64,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        kernel_size: int = 64,
        pool_size_1: int = 8,
        pool_size_2: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        temporal_filters = max(8, temporal_filters)
        self.temporal = nn.Sequential(
            nn.Conv2d(
                1,
                temporal_filters,
                kernel_size=(1, kernel_size),
                padding=(0, kernel_size // 2),
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                temporal_filters * depth_multiplier,
                kernel_size=(channels, 1),
                groups=temporal_filters,
                bias=False,
            ),
            nn.BatchNorm2d(temporal_filters * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size_1)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Sequential(
            nn.Conv2d(temporal_filters * depth_multiplier, emb_size, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(emb_size),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, pool_size_2)),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.projection(x)
        x = x.squeeze(2).transpose(1, 2)
        return x


class CTNetClassifier(nn.Module):
    def __init__(
        self,
        chans: int,
        num_classes: int,
        emb_size: int = 64,
        depth: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
        temporal_filters: int = 16,
        depth_multiplier: int = 2,
        kernel_size: int = 64,
        pool_size_1: int = 8,
        pool_size_2: int = 8,
        mlp_ratio: int = 2,
        grad_clip: float = 1.0,
    ):
        super().__init__()
        self.grad_clip = grad_clip
        self.embedding = ConvPatchEmbedding(
            channels=chans,
            emb_size=emb_size,
            temporal_filters=temporal_filters,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            pool_size_1=pool_size_1,
            pool_size_2=pool_size_2,
            dropout=dropout,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size,
            nhead=num_heads,
            dim_feedforward=emb_size * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.position_dropout = nn.Dropout(dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(emb_size)
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.position_dropout(x)
        x = self.encoder(x)
        x = self.norm(x.mean(dim=1))
        return self.classifier(x)

    def clip_gradients(self):
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip)
