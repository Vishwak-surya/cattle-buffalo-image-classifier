from typing import Tuple
import torch.nn as nn
import torchvision.models as tvm


def get_backbone(name: str = "resnet50", pretrained: bool = True) -> Tuple[nn.Module, int]:
    if name.lower() == "resnet50":
        weights = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        model = tvm.resnet50(weights=weights)
        feat_dim = model.fc.in_features
        backbone = nn.Sequential(*list(model.children())[:-1])  # global avg pool output: (N, feat_dim, 1, 1)
        return backbone, feat_dim
    raise ValueError(f"Unsupported backbone: {name}")
