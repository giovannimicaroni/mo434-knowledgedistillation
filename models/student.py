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
    def __init__(self, initial_channels: int, config: list, out_channels: int = 512):
        super().__init__()

        # 1. Standard Initial Block
        self.initial = nn.Sequential(
            nn.Conv2d(3, initial_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112
        )

        # 2. Dynamically build the features pipeline based on the config.
        #    Each config must contain exactly 4 pool_after=True entries so that,
        #    together with the initial block's pool, the 224x224 input is reduced
        #    to a 7x7 spatial map.
        self.features = self._make_layers(initial_channels, config)

        # 3. Smoothed projection adapter
        #    Gradually transitions channel dimensions and stabilizes features
        #    before matching the teacher's output dimension.
        backbone_out = config[-1][0]
        mid_channels = (backbone_out + out_channels) // 2
        
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_out, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        )

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
        return self.proj(self.features(self.initial(x)))

    def get_student_parameters(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        initial_params = sum(p.numel() for p in self.initial.parameters()) if self.initial else 0
        features_params = sum(p.numel() for p in self.features.parameters()) if self.features else 0
        proj_params = sum(p.numel() for p in self.proj.parameters()) if self.proj else 0

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": total_params - trainable_params,
            "initial_block": initial_params,
            "features_block": features_params,
            "proj_block": proj_params,
        }

    def num_blocks(self) -> int:
        """Number of DepthwiseSeparableBlocks in the backbone (a depth proxy)."""
        return sum(1 for m in self.features if isinstance(m, DepthwiseSeparableBlock))
    
# The five students below form the comparison set. They share the BaseStudentModel
# machinery (depthwise-separable blocks + a final 1x1 projection to `out_channels`,
# the teacher's feature dim) and deliberately span a range of depth / width / total
# params. Every `config` has exactly 4 pool_after=True entries so the 224x224 input
# is reduced to a 7x7 map.

class MobileStudentModel(BaseStudentModel):
    """Original architecture (medium): ~6 blocks, balanced width."""
    def __init__(self, out_channels: int = 512):
        config = [
            (64, True),   # 112 -> 56
            (128, True),  # 56 -> 28
            (256, True),  # 28 -> 14
            (256, False), # Keeps 14x14
            (512, True),  # 14 -> 7
            (768, False), # Keeps 7x7
        ]
        super().__init__(initial_channels=32, config=config, out_channels=out_channels)


class NanoStudentModel(BaseStudentModel):
    """Smallest: narrow channels, 4 blocks — minimal parameter count."""
    def __init__(self, out_channels: int = 512):
        config = [
            (24, True),   # 112 -> 56
            (48, True),   # 56 -> 28
            (64, True),   # 28 -> 14
            (96, True),   # 14 -> 7
        ]
        super().__init__(initial_channels=16, config=config, out_channels=out_channels)


class WideStudentModel(BaseStudentModel):
    """Widest: large channel counts, 4 blocks — high params via width."""
    def __init__(self, out_channels: int = 512):
        config = [
            (96, True),   # 112 -> 56
            (192, True),  # 56 -> 28
            (384, True),  # 28 -> 14
            (512, True),  # 14 -> 7
        ]
        super().__init__(initial_channels=48, config=config, out_channels=out_channels)


class DeepStudentModel(BaseStudentModel):
    """Deepest: 7 blocks with interleaved non-pooling blocks — high params via depth."""
    def __init__(self, out_channels: int = 512):
        config = [
            (48, True),   # 112 -> 56
            (64, False),  # Keeps 56x56
            (96, True),   # 56 -> 28
            (128, False), # Keeps 28x28
            (192, True),  # 28 -> 14
            (256, False), # Keeps 14x14
            (384, True),  # 14 -> 7
        ]
        super().__init__(initial_channels=32, config=config, out_channels=out_channels)


class BottleneckStudentModel(BaseStudentModel):
    """Narrows then re-expands at the deepest stage — 5 blocks."""
    def __init__(self, out_channels: int = 512):
        config = [
            (64, True),   # 112 -> 56
            (128, True),  # 56 -> 28
            (256, True),  # 28 -> 14
            (128, False), # Bottleneck: narrow at 14x14
            (256, True),  # 14 -> 7
        ]
        super().__init__(initial_channels=32, config=config, out_channels=out_channels)