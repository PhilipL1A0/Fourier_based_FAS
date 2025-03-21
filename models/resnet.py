# resnet18模型定义
import torch
import torch.nn as nn
from components import ConvBlock, ResidualBlock, CBAMBlock

class ResNet18(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, use_attention=True):
        super().__init__()
        self.use_attention = use_attention
        
        # 频域输入层（单通道或双通道）
        self.conv1 = ConvBlock(
            in_channels=input_channels,  # 频域数据通道数（如幅度谱：1，幅度+相位：2）
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # 构建各个阶段的残差块（支持CBAM注意力）
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """构建一个阶段的残差块（可选CBAM）"""
        layers = []
        in_channels = out_channels // 2 if stride == 2 else out_channels
        
        # 第一个块需要调整通道和步长
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        if self.use_attention:
            layers.append(CBAMBlock(out_channels))
         
        # 后续块保持通道和步长
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
            if self.use_cbam:
                layers.append(CBAMBlock(out_channels))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    # 提取特征
    def extract_features(self, x):
        feats = []
        x = self.conv1(x)
        
        x = self.layer1(x)
        feats.append(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)
        
        return feats