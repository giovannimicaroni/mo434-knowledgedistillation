from __future__ import annotations

import torch.nn as nn
from torchvision.models import convnext_small, ConvNeXt_Small_Weights

from models.base_teacher import BaseTeacherModel

class ConvNeXtTeacher(BaseTeacherModel):
    def __init__(
        self,
        weights_path: str | None = None,
        pretrained: bool = True,
        num_classes: int = 2,
    ):
        self.pretrained = pretrained
        self.num_classes = num_classes
        super().__init__(weights_path=weights_path)

    def _build_model(self) -> nn.Module:
        if self.pretrained:
            model = convnext_small(weights=ConvNeXt_Small_Weights.DEFAULT)
        else:
            model = convnext_small(weights=None)

        if self.num_classes != 1000:
            
            in_features = model.classifier[2].in_features
            model.classifier[2] = nn.Linear(in_features, self.num_classes)

        return model
    
    def forward(self, x):

        return self.model(x)

    def extract_features(self, x):

        return self.model.features(x)

    def forward_classifier(self, x):

        if x.dim() == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
            
        
        if x.size(-1) > 1 or x.size(-2) > 1:
            x = self.model.avgpool(x)
            
        return self.model.classifier(x)