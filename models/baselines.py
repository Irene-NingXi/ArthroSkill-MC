"""General-purpose video baselines reported in the paper's expanded comparison."""
import torch
import torch.nn as nn
from .appearance_branch import AppearanceBranch

class VideoSwinOnly(nn.Module):
    """Video Swin-T appearance-only classifier/regressor."""
    def __init__(self, config):
        super().__init__(); dim=int(config['model']['appearance'].get('output_dim',768)); h=int(config['model']['heads']['classification'].get('hidden_dim',512)); d=float(config['model']['heads']['classification'].get('dropout',.3))
        self.backbone=AppearanceBranch(output_dim=dim,pretrained=config['model']['appearance'].get('pretrained',True)); self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,h),nn.ReLU(),nn.Dropout(d)); self.cls=nn.Linear(h,config['data']['num_classes']); self.reg=nn.Linear(h,1)
    def forward(self, appearance, motion=None, detection_conf=None):
        z=self.head(self.backbone(appearance)); return {'cls_logits':self.cls(z),'total_score':self.reg(z)}

class ResNet3DBaseline(nn.Module):
    """3D-ResNet-18 baseline; torchvision weights are optional."""
    def __init__(self, config, pretrained=False):
        super().__init__()
        from torchvision.models.video import r3d_18, R3D_18_Weights
        self.backbone=r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
        dim=self.backbone.fc.in_features
        self.backbone.fc=nn.Identity()
        self.cls=nn.Linear(dim,config['data']['num_classes']); self.reg=nn.Linear(dim,1)
    def forward(self, appearance, motion=None, detection_conf=None):
        z=self.backbone(appearance); return {'cls_logits':self.cls(z),'total_score':self.reg(z)}
