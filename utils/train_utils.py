import os
import csv
from numpy import save
import torch
import logging
import torch.optim as optim
from torch.nn.parallel import DataParallel
from torch.optim.lr_scheduler import LambdaLR
from torch.amp.grad_scaler import GradScaler
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


def setup_scheduler(optimizer, config):
    """学习率调度器（Warmup+Cosine）"""
    def warmup_cosine_lr(current_step):
        if current_step < config.warmup_epochs:
            return (current_step / config.warmup_epochs) * (config.lr - config.lr_min) + config.lr_min
        else:
            progress = (current_step - config.warmup_epochs) / (config.epochs - config.warmup_epochs)
            return config.lr_min + 0.5 * (config.lr - config.lr_min) * (1 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)


def setup_logger(output_dir, model_name):
    """
    设置日志记录器，将日志保存到文件并输出到控制台，同时初始化 CSV 文件记录训练信息。

    Args:
        output_dir (str): 日志文件保存的目录。
        model_name (str): 模型名称，用于生成日志文件名。

    Returns:
        logging.Logger: 配置好的日志记录器。
        str: CSV 文件路径。
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'log', model_name + ".log")
    csv_path = os.path.join(output_dir, 'csv', model_name + ".csv")

    # 创建日志记录器
    logger = logging.getLogger("TrainingLogger")
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

    # 初始化 CSV 文件
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Train Loss", "Train Acc", "Val Loss", "Val Acc"])

    return logger, csv_path


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
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            save_model(model, self.path)
            self.val_loss_min = val_loss

    return EarlyStopping(
        patience=config.patience,
        path=os.path.join(model_save_path+ ".pth")
    )


def setup_amp(config):
    """混合精度初始化"""
    if config.use_amp:
        return GradScaler()
    return None


def save_model(model, path):
    """保存模型（兼容多GPU）"""
    path = path + ".pth"
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)