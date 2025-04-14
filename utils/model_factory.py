"""
模型工厂模块 - 负责创建各种模型实例
"""
from models import *

def create_model(config):
    """
    根据配置创建模型实例
    
    Args:
        config: 配置对象
        
    Returns:
        model: 模型实例
    """
    if config.model_type == "ResNet18":
        return create_resnet18(config)
    elif config.model_type == "PretrainedResNet18":
        return create_pretrained_resnet18(config)
    elif config.model_type == "DualStreamNetwork":
        return create_dual_stream_network(config)
    else:
        raise ValueError(f"不支持的模型类型: {config.model_type}")

def create_resnet18(config):
    """创建普通ResNet18模型"""
    return ResNet18(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        dropout_rate=config.dropout,
        use_attention=config.use_attention
    )

def create_pretrained_resnet18(config):
    """创建预训练ResNet18模型"""
    return PretrainedResNet18(
        input_channels=config.input_channels,
        num_classes=config.num_classes,
        use_attention=config.use_attention,
        dropout_rate=config.dropout,
        freeze_backbone=config.freeze_backbone,
        freeze_layers=config.freeze_layers
    )

def create_dual_stream_network(config):
    """创建双流网络模型"""
    # 创建空域分支网络
    spatial_network = None
    if hasattr(config, 'spatial_model_type'):
        if config.spatial_model_type == "PretrainedResNet18":
            spatial_network = PretrainedResNet18(
                input_channels=config.spatial_channels,
                num_classes=config.num_classes,
                use_attention=config.use_attention,
                dropout_rate=config.dropout,
                freeze_backbone=config.freeze_spatial,
                freeze_layers=config.freeze_layers if hasattr(config, 'freeze_layers') else []
            )
        elif config.spatial_model_type == "ResNet18":
            spatial_network = ResNet18(
                num_classes=config.num_classes,
                input_channels=config.spatial_channels,
                dropout_rate=config.dropout,
                use_attention=config.use_attention
            )
        else:
            raise ValueError(f"不支持的空域模型类型: {config.spatial_model_type}")
    else:
        # 默认使用预训练的ResNet18
        spatial_network = PretrainedResNet18(
            input_channels=config.spatial_channels,
            num_classes=config.num_classes,
            use_attention=config.use_attention,
            dropout_rate=config.dropout
        )
    
    # 创建频域分支网络
    freq_network = None
    if hasattr(config, 'freq_model_type'):
        if config.freq_model_type == "PretrainedResNet18":
            freq_network = PretrainedResNet18(
                input_channels=config.freq_channels,
                num_classes=config.num_classes,
                use_attention=config.use_attention,
                dropout_rate=config.dropout,
                freeze_backbone=config.freeze_freq,
                freeze_layers=config.freeze_layers if hasattr(config, 'freeze_layers') else []
            )
        elif config.freq_model_type == "ResNet18":
            freq_network = ResNet18(
                num_classes=config.num_classes,
                input_channels=config.freq_channels,
                dropout_rate=config.dropout,
                use_attention=config.use_attention
            )
        else:
            raise ValueError(f"不支持的频域模型类型: {config.freq_model_type}")
    else:
        # 默认使用普通ResNet18
        freq_network = ResNet18(
            num_classes=config.num_classes,
            input_channels=config.freq_channels,
            dropout_rate=config.dropout,
            use_attention=config.use_attention
        )
    
    # 创建双流网络，传入已构建的分支网络
    return DualStreamNetwork(
        spatial_network=spatial_network,
        freq_network=freq_network,
        num_classes=config.num_classes,
        fusion_type=config.fusion_type,
        fusion_dropout=config.fusion_dropout
    )
