# components.py: 常用的模型组件
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """基础卷积块：Conv + BN + 激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, activation="relu"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU() if activation == "relu" else nn.Identity()

    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class ResidualBlock(nn.Module):
    """残差块（Residual Block）"""
    def __init__(self, channels, activation="relu"):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, activation=activation)
        self.conv2 = ConvBlock(channels, channels, activation=None)  # 最后一层不加激活
        self.identity = nn.Identity()  # 残差连接

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.identity(x)

class ChannelAttention(nn.Module):
    """通道注意力（SE Block）"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = ConvBlock(2, 1, kernel_size=kernel_size, padding=kernel_size//2, activation=None)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

class CBAMBlock(nn.Module):
    """组合通道+空间注意力（CBAM）"""
    def __init__(self, channels):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x
    
class InvertedResidual(nn.Module):
    """倒残差模块（用于EfficientNet等）"""
    def __init__(self, in_channels, out_channels, expansion_ratio=6):
        super().__init__()
        hidden_dim = in_channels * expansion_ratio
        self.conv1 = ConvBlock(in_channels, hidden_dim, kernel_size=1)
        self.conv2 = ConvBlock(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False)
        self.identity = nn.Identity()

    def forward(self, x):
        return self.conv3(self.conv2(self.conv1(x))) + self.identity(x)
    
