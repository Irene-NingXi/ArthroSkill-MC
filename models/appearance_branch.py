"""Spatiotemporal Video Swin-T with explicit checkpoint loading."""
import torch
from torch import nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

class AppearanceBranch(nn.Module):
    def __init__(self, output_dim=768, pretrained=True, dropout=0.1, checkpoint=None):
        super().__init__()
        self.backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1 if pretrained and not checkpoint else None)
        self.feature_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state.get("backbone_state_dict", state), strict=True)
        self.proj = nn.Sequential(nn.Linear(self.feature_dim, output_dim), nn.LayerNorm(output_dim), nn.Dropout(dropout))

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != 3:
            raise ValueError("Expected video tensor [B, 3, T, H, W]")
        return self.proj(self.backbone(x))
