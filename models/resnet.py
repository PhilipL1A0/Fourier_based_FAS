# config.py
import torch
import torch.nn as nn
from .components import ConvBlock, ResidualBlock, CBAMBlock

class ResNet18(nn.Module):
    """
    自定义实现的 ResNet18 模型
    """
    def __init__(self, input_channels=1, num_classes=2, num_blocks=[2, 2, 2, 2], 
                 use_attention=True, dropout_rate=0):
        """
        初始化 ResNet 模型

        Args:
            input_channels (int): 输入通道数
            num_classes (int): 分类任务的类别数
            num_blocks (list): 每个阶段的残差块数量
            use_attention (bool): 是否启用注意力模块
            dropout_rate (float): Dropout比率
        """
        super().__init__()
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        # 初始卷积层
        self.conv1 = ConvBlock(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差块阶段
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """构建一个阶段的残差块"""
        layers = []
        in_channels = out_channels // 2 if stride == 2 else out_channels

        # 第一个残差块
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        if self.use_attention:
            layers.append(CBAMBlock(out_channels))

        # 后续残差块
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            if self.use_attention:
                layers.append(CBAMBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """前向传播"""
        x = self.conv1(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.dropout_rate > 0:
            x = self.dropout(x)
        x = self.fc(x)
        return x

    def extract_features(self, x):
        """提取特征（用于可视化或其他任务）"""
        feats = []
        
        x = self.conv1(x)
        x = self.pool(x)

        x = self.layer1(x)
        feats.append(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)

        return feats