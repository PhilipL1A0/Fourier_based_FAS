# config.py
from tkinter.tix import Tree


class Config:
    def __init__(self):
        #===== 模型结构配置 =====
        # 基本模型类型设置
        self.model_type = "DualStreamNetwork"  # "ResNet18", "PretrainedResNet18", "DualStreamNetwork"
        
        # ResNet配置
        self.backbone = "resnet18"
        self.block_type = "residual"
        self.num_blocks = [3, 4, 6, 3]
        self.num_classes = 2
        self.dropout = 0.5
        
        # 注意力机制配置
        self.use_attention = True
        self.attention_type = "cbam"
        
        # 预训练设置
        self.pretrained = True if self.model_type == "PretrainedResNet18" else False
        self.freeze_backbone = True
        self.freeze_layers = ["conv1", "bn1", "pool", "layer1", "layer2"]
        
        # 双流网络配置
        if self.model_type == "DualStreamNetwork":
            self.spatial_model_type = "PretrainedResNet18"  # 空域分支模型类型
            self.pretrained = True if self.spatial_model_type == "PretrainedResNet18" else False
            self.freq_model_type = "ResNet18"  # 频域分支模型类型
            
            self.spatial_channels = 3  # RGB图像通道数
            self.freq_channels = 1     # 频域图像通道数
            
            # 融合设置
            self.fusion_type = "mlp"   # 'concat', 'add', 'attention', 'mlp', 'cbam'
            self.fusion_dropout = 0.5  # 融合层Dropout率

            self.freeze_spatial = False  # 是否冻结空域分支
            self.freeze_freq = False     # 是否冻结频域分支

        #===== 训练参数配置 =====
        self.batch_size = 32
        self.epochs = 300
        self.lr = 1e-5
        self.backbone_lr_ratio = 0.5  # 主干网络学习率缩放比例
        self.weight_decay = 1e-4
        self.num_workers = 4
        self.loss_func = "cross_entropy"  # "cross_entropy", "focal_loss"

        #===== 数据路径配置 =====
        self.base_dir = "/media/main/lzf/FBFAS"
        self.data_dir = f"{self.base_dir}/data"
        self.output_dir = f"{self.base_dir}/outputs"
        self.compress_data = True

        #===== 训练策略开关 =====
        self.use_early_stopping = True
        self.use_multi_gpu = True
        self.use_amp = True
        self.use_warmup = True
        self.use_lr_cos = True
        self.use_augment = True
        self.use_gradient_clipping = True
        self.use_tensorboard = True
        self.test_model = True

        #===== 训练策略参数 =====
        self.patience = 30
        self.devices = [0, 1]
        self.amp_opt_level = "O1"
        self.warmup_epochs = 5
        self.lr_min = 1e-6
        self.lr_cycles = 1
        self.noise_std = 0.01
        self.clip_value = 1.0

        #===== 数据集配置 =====
        self.dataset = "CASIA"  # "CASIA", "idiap", "MSU", "OULU", "all"
        self.test_dataset = "CASIA"  # "CASIA", "idiap", "MSU", "OULU", "all"
        self.data_mode = "both"  # "spatial", "frequency", "both"
        self.spatial_type = "rgb"  # "rgb", "gray"
        self.use_multi_channel = True
        self.detect_face = True
        self.input_channels = self._calculate_input_channels()
        self.balance_data = True

        #===== 模型命名 =====
        if_aug = "Aug" if self.use_augment else "NoAug"
        if_dual = "Dual" if self.model_type == "DualStreamNetwork" else "Single"
        if_pretrained = "_Pre" if self.pretrained else ""
        self.model_name = f"{if_dual}{if_pretrained}_{self.dataset}_{self.data_mode}_{if_aug}"
            
        self.test_model = f"{self.model_name}_on_{self.test_dataset}"
        
    def _calculate_input_channels(self):
        """根据数据模式计算输入通道数"""
        if self.data_mode == "spatial":
            return 3
        elif self.data_mode == "frequency":
            return 1
        elif self.data_mode == "both":
            return 4