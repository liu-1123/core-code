import torch
import torch.nn as nn
from torchvision import models


# 1. 注意力机制模块
class LightweightChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_val], dim=1)))


# 2. 核心模型架构
class FaultDiagnosisModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = models.resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        # 添加注意力模块
        self.leca_layers = nn.ModuleList([
            LightweightChannelAttention(64),
            LightweightChannelAttention(128),
            LightweightChannelAttention(256),
            LightweightChannelAttention(512)
        ])
        self.spatial_layers = nn.ModuleList([SpatialAttention() for _ in range(4)])

        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        for i, (layer, leca, spatial) in enumerate(zip(
                [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4],
                self.leca_layers,
                self.spatial_layers
        )):
            x = layer(x)
            x = leca(x)
            x = spatial(x)

        x = self.backbone.avgpool(x)
        return self.backbone.fc(x.flatten(1))


# 3. 训练流程（伪代码）
def train_model():
    # 初始化
    model = FaultDiagnosisModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

