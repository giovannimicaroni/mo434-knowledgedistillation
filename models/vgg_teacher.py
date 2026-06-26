import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights

from models.base_teacher import BaseTeacherModel


class VGGTeacher(BaseTeacherModel):
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
            model = vgg11(weights=VGG11_Weights.DEFAULT)
        else:
            model = vgg11(weights=None)

        if self.num_classes != 1000:
            in_features = model.classifier[0].in_features
            print(in_features)
            model.classifier = nn.Linear(in_features, self.num_classes) # type: ignore

        return model
