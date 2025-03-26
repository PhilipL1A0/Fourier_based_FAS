# config.py
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

        # 训练参数配置
        self.batch_size = 32
        self.epochs = 100
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.num_workers = 4

        # 数据路径配置
        self.data_dir = "/media/main/lzf/FBFAS/data/dataset"
        self.output_dir = "/media/main/lzf/FBFAS/outputs"

        # 训练策略开关
        self.use_early_stopping = True
        self.use_multi_gpu = True
        self.use_amp = True
        self.use_warmup = True
        self.use_lr_cos = True
        self.use_augmentation = True

        # 训练策略参数
        self.patience = 10
        self.devices = [0, 1]
        self.amp_opt_level = "O1"
        self.warmup_epochs = 5
        self.lr_min = 1e-6
        self.lr_cycles = 1

        # 其他可选功能
        self.use_gradient_clipping = True
        self.clip_value = 1.0
        self.use_tensorboard = True
        self.test_model = True

        # 模型名称
        self.model_name = f"{self.backbone}_L2_{self.epochs}_{self.batch_size}_{self.lr}_{self.loss_func}_NoAug"