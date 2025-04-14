import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import _LRScheduler
import math

def setup_device(config, model):
    """设备配置（多GPU/单GPU/CPU）"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.use_multi_gpu and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=config.devices)
    model.to(device)
    return model, device


def setup_optimizer(model, config):
    """初始化优化器（AdamW）"""
    return optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )


class WarmupCosineScheduler(_LRScheduler):
    """带预热的余弦学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0, last_epoch=-1):
        """
        初始化调度器
        
        Args:
            optimizer: 优化器
            warmup_epochs: 预热轮次数
            max_epochs: 总训练轮次
            min_lr: 最小学习率
            last_epoch: 上一轮次
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """计算当前学习率"""
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段 - 线性增加学习率
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # 确保不超过1
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [self.min_lr + (base_lr - self.min_lr) * cosine_factor 
                    for base_lr in self.base_lrs]

def setup_scheduler(optimizer, config):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        config: 配置对象
        
    Returns:
        学习率调度器
    """
    if config.use_warmup and config.use_lr_cos:
        # 预热 + 余弦退火
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.warmup_epochs,
            max_epochs=config.epochs,
            min_lr=config.lr_min
        )
    elif config.use_lr_cos:
        # 仅余弦退火
        return CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
            eta_min=config.lr_min
        )
    else:
        # 常数学习率（不做调整）
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)


def setup_logger(output_dir, model_name):
    """
    设置日志记录器，将日志保存到文件并输出到控制台。

    Args:
        output_dir (str): 日志文件保存的目录。
        model_name (str): 模型名称，用于生成日志文件名。

    Returns:
        logging.Logger: 配置好的日志记录器。
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'log', model_name + ".log")

    # 创建日志记录器
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.INFO)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def setup_early_stopping(config, model_save_path):
    """早停机制初始化"""
    class EarlyStopping:
        def __init__(self, patience=10, verbose=True, path="checkpoint.pth"):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')
            self.path = path

        def __call__(self, val_loss, model):
            score = -val_loss
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + 1e-7:
                self.counter += 1
                if self.verbose:
                    print(f"早停计数: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(f'验证损失减少 ({self.val_loss_min:.6f} -> {val_loss:.6f})，保存模型...')
            save_model(model, self.path)
            self.val_loss_min = val_loss

    return EarlyStopping(
        patience=config.patience,
        path=os.path.join(model_save_path)
    )


def setup_amp(config):
    """混合精度初始化"""
    if config.use_amp:
        return GradScaler()
    return None


def save_model(model, save_path, optimizer=None, epoch=None, config=None):
    """
    保存模型权重和训练状态
    
    Args:
        model (nn.Module): 模型实例
        save_path (str): 保存路径
        optimizer (Optimizer, optional): 优化器
        epoch (int, optional): 当前训练轮次
        config (Config, optional): 配置对象
    """
    save_path = save_path + ".pth"
    
    # 保存模型
    state = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_classes': model.num_classes if hasattr(model, 'num_classes') else None,
            'input_channels': model.input_channels if hasattr(model, 'input_channels') else None,
            'use_attention': model.use_attention if hasattr(model, 'use_attention') else None,
            'dropout_rate': model.dropout_rate if hasattr(model, 'dropout_rate') else None,
            'pretrained': model.pretrained if hasattr(model, 'pretrained') else None,
        }
    }
    
    # 添加优化器状态和当前轮次（如果提供）
    if optimizer is not None:
        state['optimizer_state_dict'] = optimizer.state_dict()
    if epoch is not None:
        state['epoch'] = epoch
    if config is not None:
        state['config'] = vars(config)
    
    # 保存到文件
    torch.save(state, save_path)


def count_parameters(model):
    """
    统计模型的参数量（总参数量和可训练参数量）。

    Args:
        model (torch.nn.Module): 要统计的模型。

    Returns:
        dict: 包含总参数量和可训练参数量的字典。
    """
    if isinstance(model, DataParallel):  # 如果是多 GPU 模型，访问内部的模型
        model = model.module

    total_params = sum(p.numel() for p in model.parameters())  # 总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # 可训练参数量

    return {
        "total_params": total_params,
        "trainable_params": trainable_params
    }


def get_class_weights(train_data):
    """计算类别权重"""
    class_count = torch.tensor([len(train_data) - sum(train_data.labels), sum(train_data.labels)]).float()
    class_weights = class_count / class_count.sum()
    return class_weights


class FocalLoss(nn.Module):
    """
    Focal Loss 实现，用于处理类别不平衡问题。
    Args:
        alpha (float): 平衡因子，用于调整正负样本的权重。
        gamma (float): 聚焦因子，用于调整难分类样本的权重。
        reduction (str): 损失的聚合方式，支持 'mean'、'sum' 和 'none'。
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 计算预测概率
        pt = torch.exp(-ce_loss)
        # 计算 Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss