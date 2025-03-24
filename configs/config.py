# config.py

class Config:
    # 训练基础配置
    backbone = "resnet"          # 模型主干
    block_type = "residual"      # 基础块类型（residual）
    use_attention = True         # 启用注意力模块
    attention_type = "cbam"      # 注意力类型
    num_blocks = [2, 2, 2, 2]    # 残差块配置
    num_classes = 2              # 二分类任务
    input_channels = 1           # 频域数据通道数（假设是单通道，如幅度谱）
    attention_type = "cbam"

    # 训练参数配置
    batch_size = 32
    epochs = 100
    lr = 1e-3
    weight_decay = 1e-4
    num_workers = 4

    # 数据路径配置
    model_name = f"{backbone}_{epochs}_{batch_size}_{lr}_{epochs}.pth"
    data_dir = "/media/main/lzf/FBFAS/data/dataset"
    output_dir = "/media/main/lzf/FBFAS/outputs"

    # 训练策略开关
    use_early_stopping = True  # 早停
    use_multi_gpu = True       # 多GPU
    use_amp = True             # 混合精度
    use_warmup = True          # Warmup学习率
    use_lr_cos = True          # 余弦退火学习率

    # 训练策略参数
    patience = 10         # 早停等待次数
    devices = [0, 1]      # 多GPU设备
    amp_opt_level = "O1"  # 混合精度级别（如O1/O2）
    warmup_epochs = 5     # Warmup步数
    total_steps = 1000    # 总训练步数（用于Warmup+Cosine）
    lr_min = 1e-5         # 最小学习率
    lr_cycles = 1         # 余弦周期数

    # 其他可选功能（可扩展）
    use_gradient_clipping = True  # 梯度裁剪
    clip_value = 1.0              # 裁剪值
    use_tensorboard = True        # Tensorboard可视化