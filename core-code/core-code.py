import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

# 轻量级通道注意力模块（LECA）
class ChannelAttention(nn.Module):
    """轻量级通道注意力模块，仅使用平均池化，激活函数为Hard-Sigmoid"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False),
            nn.Hardswish()  # PyTorch中Hardswish近似Hard-Sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(self.avg_pool(x)).sigmoid()  # 应用Sigmoid归一化

# 空间注意力模块 
class PositionAttention(nn.Module):
    """空间注意力模块，通过特征图的通道间关系生成空间注意力图"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        spatial = self.sigmoid(self.conv(spatial))
        return x * spatial

# 修改后的残差块，在每个残差块后添加注意力机制
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        
        # 添加注意力模块
        self.ca = ChannelAttention(planes)
        self.pa = PositionAttention()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # 先通道注意力，再空间注意力
        out = self.ca(out)
        out = self.pa(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# 自定义ResNet18，使用修改后的残差块
class ResNet18WithAttention(nn.Module):
    def __init__(self, num_classes=10, pretrained=False, in_channels=1):
        super(ResNet18WithAttention, self).__init__()
        # 单通道输入时不加载预训练权重（避免通道不匹配）
        if in_channels == 1 and pretrained:
            pretrained = False
        resnet = models.resnet18(pretrained=pretrained)
        
        # 修改输入通道
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained and in_channels == 3:
            self.conv1.weight.data = resnet.conv1.weight.data.clone()  
        
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = resnet.avgpool
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # 初始化权重
        if in_channels != 3:
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )
        layers = [BasicBlock(inplanes, planes, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 故障诊断模型主体
class IndustrialFaultDetector(nn.Module):
    def __init__(self, num_classes: int = 10, pretrained: bool = False, in_channels: int = 1):
        super().__init__()
        self.backbone = ResNet18WithAttention(num_classes, pretrained, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# 模型训练与评估
def initialize_training(
    model: nn.Module, 
    learning_rate: float = 0.001,  
    weight_decay: float = 0.00001,  
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple:
    model = model.to(device)
    # 冻结前几层（conv1, bn1, layer1, layer2）
    if hasattr(model, 'backbone'):
        for name, param in model.backbone.named_parameters():
            if 'conv1' in name or 'bn1' in name or 'layer1' in name or 'layer2' in name:
                param.requires_grad = False
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate, 
        weight_decay=weight_decay  # 1e-5
    )
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, device

def train_epoch(
    model: nn.Module, 
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """执行一个训练轮次，返回平均训练损失"""
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 混合精度训练
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * inputs.size(0)
    
    return running_loss / len(train_loader.dataset)

def validate(
    model: nn.Module, 
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple[float, float]:
    """在验证集上评估模型，返回平均损失和准确率"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(val_loader.dataset), correct / total

# 完整训练流程
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    save_path: str = 'best_model.pth',
    use_amp: bool = True
) -> None:
    """完整的模型训练流程，包括早停和模型保存"""
    model, optimizer, criterion, device = initialize_training(
        model, learning_rate, weight_decay, device
    )
    
    # 学习率调度器 
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=7, gamma=0.1  # 每7个epoch衰减0.1倍
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # 训练阶段
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        
        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()  # 更新学习率
        
        print(f'Epoch {epoch+1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Val Acc: {val_acc:.4f}')
        
        # 模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
            }, save_path)
            print(f'Model saved with accuracy: {val_acc:.4f}')

# 使用示例
if __name__ == "__main__":
    # 模型初始化 
    model = IndustrialFaultDetector(num_classes=5, pretrained=True, in_channels=1)
    
    # 这里需要添加数据加载器初始化代码
    # train_loader = ...
    # val_loader = ...
    
    # 训练模型
    # train_model(model, train_loader, val_loader, epochs=30)
    pass    
