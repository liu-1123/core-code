import torch
import torch.nn as nn
from torchvision import models
from typing import Optional

# 注意力机制实现
class ChannelAttention(nn.Module):
    """轻量级通道注意力模块，通过平均池化和最大池化提取特征"""
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用1x1卷积替代全连接层
        self.shared_fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_fc(self.avg_pool(x))
        max_out = self.shared_fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class PositionAttention(nn.Module):
    """空间注意力模块，通过特征图的通道间关系生成空间注意力图"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 沿通道维度进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_out, max_out], dim=1)
        
        # 生成空间注意力图并应用
        spatial = self.sigmoid(self.conv(spatial))
        return x * spatial

# 故障诊断模型主体
class IndustrialFaultDetector(nn.Module):
    """基于ResNet18的工业故障诊断模型，集成了注意力机制增强特征提取"""
    
    def __init__(self, num_classes: int = 10, pretrained: bool = False):
        super().__init__()
        # 使用预训练ResNet18作为骨干网络
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # 调整输入通道数（如果需要非RGB输入）
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 定义注意力模块
        self.attention_blocks = nn.ModuleList([
            nn.Sequential(
                ChannelAttention(64),
                PositionAttention()
            ),
            nn.Sequential(
                ChannelAttention(128),
                PositionAttention()
            ),
            nn.Sequential(
                ChannelAttention(256),
                PositionAttention()
            ),
            nn.Sequential(
                ChannelAttention(512),
                PositionAttention()
            )
        ])
        
        # 修改分类头
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 特征提取阶段
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        # 特征增强阶段（应用注意力机制）
        for i, layer in enumerate([
            self.backbone.layer1, 
            self.backbone.layer2,
            self.backbone.layer3, 
            self.backbone.layer4
        ]):
            x = layer(x)
            x = self.attention_blocks[i](x)
        
        # 分类阶段
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x)

# 模型训练与评估
def initialize_training(
    model: nn.Module, 
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> tuple:
    """初始化训练所需的组件，返回模型、优化器、损失函数和设备"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
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
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    save_path: str = 'best_model.pth',
    use_amp: bool = True
) -> None:
    """完整的模型训练流程，包括早停和模型保存"""
    model, optimizer, criterion, device = initialize_training(
        model, learning_rate, weight_decay, device
    )
    
    # 学习率调度器和混合精度支持
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
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
        scheduler.step(val_loss)
        
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
    model = IndustrialFaultDetector(num_classes=5)
    
    # 这里需要添加数据加载器初始化代码
    # train_loader = ...
    # val_loader = ...
    
    # 训练模型
    # train_model(model, train_loader, val_loader, epochs=30)
    pass
