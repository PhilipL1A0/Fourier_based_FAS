# config.py
import torch
import torch.nn as nn
import torchvision.models as models
from .components import ConvBlock, ResidualBlock, CBAMBlock

class ResNet18(nn.Module):
    """
    通用 ResNet 模型，支持动态配置残差块数量、注意力模块和预训练模型
    """
    def __init__(self, input_channels=1, num_classes=2, num_blocks=[2, 2, 2, 2], 
                 use_attention=True, dropout_rate=0, pretrained=False, 
                 freeze_backbone=False, freeze_layers=None):
        """
        初始化 ResNet 模型

        Args:
            input_channels (int): 输入通道数（如频域数据的通道数）。
            num_classes (int): 分类任务的类别数。
            num_blocks (list): 每个阶段的残差块数量（仅在非预训练模式下使用）。
            use_attention (bool): 是否启用 CBAM 注意力模块。
            dropout_rate (float): Dropout 概率。
            pretrained (bool): 是否使用预训练模型。
            freeze_backbone (bool): 是否冻结主干网络。
            freeze_layers (list): 要冻结的层名称列表。
        """
        super().__init__()
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.pretrained = pretrained
        
        if freeze_layers is None:
            freeze_layers = []
        
        if pretrained:
            # 使用预训练 ResNet 模型
            base_model = models.resnet18(weights='IMAGENET1K_V1')
            
            # 如果输入通道数不是3（ImageNet预训练使用RGB三通道），则修改第一层卷积
            if input_channels != 3:
                # 保存原始的第一层卷积权重
                original_conv = base_model.conv1.weight.data
                
                # 创建新的卷积层，输入通道数为 input_channels
                base_model.conv1 = nn.Conv2d(
                    input_channels, 
                    64, 
                    kernel_size=7, 
                    stride=2, 
                    padding=3, 
                    bias=False
                )
                
                # 初始化新卷积层的权重
                if input_channels == 1:
                    # 如果输入是单通道，可以使用原始权重的平均值
                    base_model.conv1.weight.data = original_conv.mean(dim=1, keepdim=True)
                else:
                    # 否则初始化为相同通道的重复
                    for i in range(input_channels):
                        if i < 3:
                            base_model.conv1.weight.data[:, i:i+1, :, :] = original_conv[:, i:i+1, :, :]
                        else:
                            # 超过3个通道的部分使用随机初始化
                            nn.init.kaiming_normal_(base_model.conv1.weight.data[:, i:i+1, :, :])
            
            # 复用预训练模型的各个部分
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.pool = base_model.maxpool
            self.layer1 = base_model.layer1
            self.layer2 = base_model.layer2
            self.layer3 = base_model.layer3
            self.layer4 = base_model.layer4
            self.avgpool = base_model.avgpool
            
            # 冻结指定层
            if freeze_backbone:
                for param in self.parameters():
                    param.requires_grad = False
            else:
                for layer_name in freeze_layers:
                    if hasattr(self, layer_name):
                        for param in getattr(self, layer_name).parameters():
                            param.requires_grad = False
            
        else:
            # 使用自定义 ResNet 实现
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
        
        # 注意力模块（对于预训练模型，单独添加）
        if self.use_attention and self.pretrained:
            self.cbam1 = CBAMBlock(64)
            self.cbam2 = CBAMBlock(128)
            self.cbam3 = CBAMBlock(256)
            self.cbam4 = CBAMBlock(512)
        
        # 全连接层
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        """
        构建一个阶段的残差块（支持 CBAM 注意力模块）
        注意：此方法仅在非预训练模式下使用

        Args:
            out_channels (int): 输出通道数。
            num_blocks (int): 残差块的数量。
            stride (int): 第一个残差块的步长。

        Returns:
            nn.Sequential: 一个阶段的残差块序列。
        """
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
        """
        前向传播

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 分类结果。
        """
        if self.pretrained:
            # 对于预训练模型
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.layer1(x)
            if self.use_attention:
                x = self.cbam1(x)
                
            x = self.layer2(x)
            if self.use_attention:
                x = self.cbam2(x)
                
            x = self.layer3(x)
            if self.use_attention:
                x = self.cbam3(x)
                
            x = self.layer4(x)
            if self.use_attention:
                x = self.cbam4(x)
        else:
            # 对于自定义模型
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
        """
        提取特征（用于可视化或其他任务）

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            list[torch.Tensor]: 每个阶段的特征图。
        """
        feats = []
        
        if self.pretrained:
            # 对于预训练模型
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.pool(x)

            x = self.layer1(x)
            if self.use_attention:
                x = self.cbam1(x)
            feats.append(x)
                
            x = self.layer2(x)
            if self.use_attention:
                x = self.cbam2(x)
            feats.append(x)
                
            x = self.layer3(x)
            if self.use_attention:
                x = self.cbam3(x)
            feats.append(x)
                
            x = self.layer4(x)
            if self.use_attention:
                x = self.cbam4(x)
            feats.append(x)
        else:
            # 对于自定义模型
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