import torch
import torch.nn as nn
import torchvision.models as models
from .components import CBAMBlock

class PretrainedResNet18(nn.Module):
    """
    基于预训练权重的 ResNet18 模型
    """
    def __init__(self, input_channels=3, num_classes=2, use_attention=True, 
                 dropout_rate=0, freeze_backbone=False, freeze_layers=None):
        """
        初始化预训练 ResNet 模型

        Args:
            input_channels (int): 输入通道数
            num_classes (int): 分类任务的类别数
            use_attention (bool): 是否使用CBAM注意力机制
            dropout_rate (float): Dropout比率
            freeze_backbone (bool): 是否冻结主干网络
            freeze_layers (list): 要冻结的层名称列表
        """
        super().__init__()
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        
        if freeze_layers is None:
            freeze_layers = []
        
        # 加载预训练模型
        base_model = models.resnet18(weights='IMAGENET1K_V1')
        
        # 处理输入通道数不是3的情况
        if input_channels != 3:
            # 保存原始权重
            original_conv = base_model.conv1.weight.data
            
            # 创建新的卷积层
            base_model.conv1 = nn.Conv2d(
                input_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
            
            # 初始化新卷积层权重
            if input_channels == 1:
                # 单通道输入：使用RGB通道的平均值
                base_model.conv1.weight.data = original_conv.mean(dim=1, keepdim=True)
            else:
                # 多通道输入：前三个通道复用，其余随机初始化
                for i in range(input_channels):
                    if i < 3:
                        base_model.conv1.weight.data[:, i:i+1, :, :] = original_conv[:, i:i+1, :, :]
                    else:
                        nn.init.kaiming_normal_(base_model.conv1.weight.data[:, i:i+1, :, :])
        
        # 复用预训练模型的各个组件
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
        
        # 添加注意力模块
        if use_attention:
            self.cbam1 = CBAMBlock(64)
            self.cbam2 = CBAMBlock(128)
            self.cbam3 = CBAMBlock(256)
            self.cbam4 = CBAMBlock(512)
        
        # 添加分类器
        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        
        # 修改全连接层以匹配类别数
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """前向传播"""
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

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        if self.dropout_rate > 0:
            x = self.dropout(x)
            
        x = self.fc(x)
        return x
    
    def extract_features(self, x):
        """提取特征，用于可视化或其他任务"""
        feats = []
        
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

        return feats