# config.py
import json
from xmlrpc.client import TRANSPORT_ERROR
class Config:
    def __init__(self):
        # 训练基础配置
        self.backbone = "resnet"
        self.block_type = "residual"
        self.use_attention = True
        self.attention_type = "cbam"
        self.num_blocks = [2, 2, 2, 2]
        self.num_classes = 2
        self.input_channels = 1
        self.loss_func = "cross_entropy"
        self.dropout = 0.5

        # 数据集配置
        self.dataset = "CASIA" # "CASIA", "idiap", "MSU", "OULU", "all"
        self.data_mode = "frequency"     # "spatial", "frequency", "both"
        self.spatial_type = "rgb"    # "rgb", "gray"
        self.use_multi_channel = True
        self.input_channels = self._calculate_input_channels()

        # 训练参数配置
        self.batch_size = 32
        self.epochs = 300
        self.lr = 5e-4
        self.weight_decay = 1e-4
        self.num_workers = 4

        # 数据路径配置
        self.base_dir = "/media/user/data4/lzf/FBFAS"
        self.data_dir = f"{self.base_dir}/data"
        self.output_dir = f"{self.base_dir}/outputs"
        self.compress_data = True

        # 训练策略开关
        self.use_early_stopping = True
        self.use_multi_gpu = True
        self.use_amp = True
        self.use_warmup = True
        self.use_lr_cos = True
        self.use_augment = False

        # 训练策略参数
        self.patience = 10
        self.devices = [0, 1]
        self.amp_opt_level = "O1"
        self.warmup_epochs = 5
        self.lr_min = 1e-6
        self.lr_cycles = 1
        self.noise_std = 0.01

        # 其他可选功能
        self.use_gradient_clipping = True
        self.clip_value = 1.0
        self.use_tensorboard = True
        self.test_model = True

        # 模型名称
        self.model_name = f"{self.dataset}_freq_{self.lr}_{self.loss_func}_NoAug"
        # self.model_name = "CASIA_L2_cross_entropy_NoAug"
        
    def _calculate_input_channels(self):
        if self.data_mode == "spatial":
            return 3
        elif self.data_mode == "frequency":
            return 1
        elif self.data_mode == "both":
            return 4