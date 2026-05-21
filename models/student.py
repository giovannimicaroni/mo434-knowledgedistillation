import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.relu(self.depthwise(x))
        x = self.relu(self.bn(self.pointwise(x)))
        return x


class MobileStudentModel(nn.Module):
    """Lightweight student feature extractor using depthwise separable convolutions.

    Outputs (B, 512, 7, 7) spatial feature maps compatible with the teacher's forward_classifier.
    """

    def __init__(self):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
        )
        self.features = nn.Sequential(
            DepthwiseSeparableBlock(32, 64), nn.MaxPool2d(2),   # 112 -> 56
            DepthwiseSeparableBlock(64, 128), nn.MaxPool2d(2),  # 56 -> 28
            DepthwiseSeparableBlock(128, 256), nn.MaxPool2d(2), # 28 -> 14
            DepthwiseSeparableBlock(256, 512), nn.MaxPool2d(2), # 14 -> 7
            DepthwiseSeparableBlock(512, 512),                  # 7x7, keep channels
        )

    def forward(self, x):
        return self.features(self.initial(x))  # (B, 512, 7, 7)
