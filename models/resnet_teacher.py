import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from models.base_teacher import BaseTeacherModel


class ResNetTeacher(BaseTeacherModel):
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
            model = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            model = resnet34(weights=None)

        if self.num_classes != 1000:
            in_features = model.fc.in_features
            print(in_features)
            model.fc = nn.Linear(in_features, self.num_classes) # type: ignore

        model.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        )
        
        model.classifier = model.fc

        return model