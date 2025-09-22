from typing import Dict, List, Optional
import torch.nn as nn
from .backbones import get_backbone
from .heads import ClassificationHead, RegressionHead


class ATCModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        num_classes: int = 2,
        regression_traits: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.class_head = (
            ClassificationHead(feat_dim, num_classes) if num_classes and num_classes > 0 else None
        )
        self.reg_head = (
            RegressionHead(feat_dim, len(regression_traits)) if regression_traits else None
        )

    def forward(self, x) -> Dict[str, Optional[object]]:
        feats = self.backbone(x)  # (N, C, 1, 1)
        outputs: Dict[str, Optional[object]] = {}
        if self.class_head is not None:
            outputs["class_logits"] = self.class_head(feats)
        if self.reg_head is not None:
            outputs["traits"] = self.reg_head(feats)
        return outputs
