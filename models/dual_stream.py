import torch
import torch.nn as nn
import torchvision.models as models
from .resnet import ResNet18
from .components import CBAMBlock


class DualStreamNetwork(nn.Module):
    """
    双流神经网络 - 结合空域RGB图像和频域数据
    """
    def __init__(self, 
                spatial_channels=3,  # RGB空域图像通道数
                freq_channels=1,     # 频域图像通道数
                num_classes=2,       # 分类数量
                fusion_type='concat', # 融合类型：'concat', 'add', 'attention'
                spatial_backbone='resnet18',  # 空域主干网络
                freq_backbone='resnet18',     # 频域主干网络
                pretrained=True,      # 是否使用预训练
                use_attention=True,   # 是否使用注意力机制
                dropout_rate=0.5,     # Dropout率
                freeze_spatial=False, # 是否冻结空域分支
                freeze_freq=False,    # 是否冻结频域分支
                fusion_dropout=0.5):  # 融合层Dropout率
        super().__init__()
        
        self.fusion_type = fusion_type
        self.use_attention = use_attention
        
        # 1. 初始化空域分支
        self.spatial_branch = ResNet18(
            input_channels=spatial_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
        # 2. 初始化频域分支
        self.freq_branch = ResNet18(
            input_channels=freq_channels,
            num_classes=num_classes,
            pretrained=pretrained,
            use_attention=use_attention,
            dropout_rate=dropout_rate
        )
        
        # 删除原始分支的FC层，保留特征提取器
        # 获取特征维度
        self.feat_dim = 512  # ResNet18的最后一层特征维度
        
        # 删除原始分支的分类器层
        self.spatial_branch.fc = nn.Identity()
        self.freq_branch.fc = nn.Identity()
        
        # 冻结分支 (如果需要)
        if freeze_spatial:
            for param in self.spatial_branch.parameters():
                param.requires_grad = False
        
        if freeze_freq:
            for param in self.freq_branch.parameters():
                param.requires_grad = False
        
        # 3. 初始化融合模块
        if fusion_type == 'concat':
            # 简单拼接
            self.fusion_dim = self.feat_dim * 2
            self.fusion = nn.Identity()
        elif fusion_type == 'add':
            # 特征相加
            self.fusion_dim = self.feat_dim
            self.fusion = lambda x, y: x + y
        elif fusion_type == 'attention':
            # 注意力融合
            self.fusion_dim = self.feat_dim
            self.spatial_attention = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feat_dim // 4, 1),
                nn.Sigmoid()
            )
            self.freq_attention = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim // 4),
                nn.ReLU(),
                nn.Linear(self.feat_dim // 4, 1),
                nn.Sigmoid()
            )
            self.fusion = self._attention_fusion
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        # 4. 初始化分类器
        classifier_layers = []
        
        # 添加额外的FC层进行融合后的特征处理
        classifier_layers.append(nn.Linear(self.fusion_dim, self.fusion_dim // 2))
        classifier_layers.append(nn.ReLU())
        
        if fusion_dropout > 0:
            classifier_layers.append(nn.Dropout(fusion_dropout))
            
        classifier_layers.append(nn.Linear(self.fusion_dim // 2, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
    
    def _attention_fusion(self, spatial_feat, freq_feat):
        """
        注意力机制融合空域和频域特征
        """
        # 计算注意力权重
        spatial_weight = self.spatial_attention(spatial_feat)
        freq_weight = self.freq_attention(freq_feat)
        
        # 归一化权重
        sum_weights = spatial_weight + freq_weight
        spatial_weight = spatial_weight / sum_weights
        freq_weight = freq_weight / sum_weights
        
        # 加权融合
        return spatial_weight * spatial_feat + freq_weight * freq_feat
    
    def extract_features(self, spatial_input, freq_input):
        """
        提取空域和频域特征
        """
        # 提取空域特征
        spatial_features = self.spatial_branch.extract_features(spatial_input)
        
        # 提取频域特征
        freq_features = self.freq_branch.extract_features(freq_input)
        
        return spatial_features, freq_features
        
    def forward(self, spatial_input, freq_input=None, combined_input=None):
        """
        前向传播
        
        Args:
            spatial_input: 空域RGB图像输入
            freq_input: 频域数据输入
            combined_input: 已组合的输入 (可选，用于兼容已组合的输入格式)
            
        Returns:
            logits: 分类预测结果
        """
        # 处理不同输入情况
        if combined_input is not None:
            # 如果是已组合的输入，需要分离通道
            if combined_input.shape[1] == 4:  # 假设是RGB(3通道)+频域(1通道)
                spatial_input = combined_input[:, :3, :, :]
                freq_input = combined_input[:, 3:, :, :]
            else:
                raise ValueError(f"组合输入的通道数错误: {combined_input.shape[1]}")
        
        # 确保频域输入存在
        if freq_input is None:
            raise ValueError("频域输入不能为空")
            
        # 1. 提取空域特征
        spatial_feat = self.spatial_branch.forward(spatial_input)
        
        # 2. 提取频域特征
        freq_feat = self.freq_branch.forward(freq_input)
        
        # 3. 特征融合
        if self.fusion_type == 'concat':
            fused_feat = torch.cat([spatial_feat, freq_feat], dim=1)
        elif self.fusion_type == 'add':
            fused_feat = spatial_feat + freq_feat
        elif self.fusion_type == 'attention':
            fused_feat = self._attention_fusion(spatial_feat, freq_feat)
        
        # 4. 分类
        logits = self.classifier(fused_feat)
        
        return logits