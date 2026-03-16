import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, features_batch: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(features_batch))

class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, features_batch: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(features_batch)
        out = self.block(features_batch)
        return self.pool(self.relu(out + residual))

class BrailleDotNet(nn.Module):
    def __init__(self, num_classes: int = 6, in_channels: int = 1, input_shape: tuple[int, int] = (64, 48)):
        super().__init__()
        self.num_classes = num_classes
        
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.features = nn.Sequential(
            ResidualConvBlock(16, 32),   
            ResidualConvBlock(32, 64),   
            ResidualConvBlock(64, 128)   
        )
        
        self.dropout = nn.Dropout(0.2)
        
        with torch.no_grad():
            dummy_tensor = torch.zeros(1, in_channels, *input_shape)
            self.flattened_size = self.features(self.stem(dummy_tensor)).numel()
            
        self.classifier = nn.Linear(self.flattened_size, num_classes)

    def forward(self, images_batch: torch.Tensor) -> torch.Tensor:
        features_batch = self.stem(images_batch)
        features_batch = self.features(features_batch)
        features_flat = torch.flatten(features_batch, 1)
        features_reg = self.dropout(features_flat)
        logits_batch = self.classifier(features_reg)
        return logits_batch