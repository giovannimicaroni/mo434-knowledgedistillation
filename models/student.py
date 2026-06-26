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


class BaseStudentModel(nn.Module):
    """
    A generic parent class that handles forward passes, layer generation,
    and parameter tracking so you don't have to rewrite them.
    """
    def __init__(self, initial_channels: int, config: list):
        super().__init__()
        
        # 1. Standard Initial Block
        self.initial = nn.Sequential(
            nn.Conv2d(3, initial_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
        )
        
        # 2. Dynamically build the features pipeline based on the config
        self.features = self._make_layers(initial_channels, config)

    def _make_layers(self, in_channels, config):
        layers = []
        current_channels = in_channels
        
        # Parse the config: each element is (out_channels, should_maxpool)
        for out_channels, pool_after in config:
            layers.append(DepthwiseSeparableBlock(current_channels, out_channels))
            if pool_after:
                layers.append(nn.MaxPool2d(2))
            current_channels = out_channels
            
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.features(self.initial(x))

    def get_student_parameters(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        initial_params = sum(p.numel() for p in self.initial.parameters()) if self.initial else 0
        features_params = sum(p.numel() for p in self.features.parameters()) if self.features else 0

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "initial_block": initial_params,
            "features_block": features_params
        }
    
class MobileStudentModel(BaseStudentModel):
    def __init__(self):
        # Configuration mapping out your original architecture: (channels, max_pool?)
        config = [
            (64, True),   # 112 -> 56
            (128, True),  # 56 -> 28
            (256, True),  # 28 -> 14
            (256, False), # Keeps 14x14
            (512, True),  # 14 -> 7
            (768, False)  # Keeps 7x7
        ]
        super().__init__(initial_channels=32, config=config)
    
class SlimStudentModel(BaseStudentModel):
    def __init__(self):
        config = [
            (32, True),
            (64, True),
            (128, True),
            (256, True),
            (768, False)  # Aggressive downsampling straight to 7x7
        ]
        super().__init__(initial_channels=32, config=config)

class MediumSlimModel(BaseStudentModel):
    def __init__(self):
        config = [
            (64, True),   # 112 -> 56
            (128, True),  # 56 -> 28
            (256, True),  # 28 -> 14
            (512, True),  # 14 -> 7
            (768, False)  # Keeps 7x7, expands capacity at the bottleneck
        ]
        # Starts with 32 initial channels
        super().__init__(initial_channels=32, config=config)

class MediumLargeModel(BaseStudentModel):
    def __init__(self):
        config = [
            (64, True),   # 112 -> 56
            (128, True),  # 56 -> 28
            (192, True),  # 28 -> 14 (Slimmed down from 256)
            (256, False), # 14 -> 14 (Slimmed down from 256)
            (512, True),  # 14 -> 7
            (768, False)  # Keeps 7x7
        ]
        super().__init__(initial_channels=32, config=config)