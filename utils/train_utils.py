import torch
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


def setup_early_stopping(config):
    """早停机制初始化"""
    class EarlyStopping:
        def __init__(self, patience=10, verbose=True, path="checkpoint.pt"):
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
            torch.save(model.state_dict(), self.path)
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
            self.val_loss_min = val_loss

    return EarlyStopping(
        patience=config.patience,
        path=f"{config.output_dir}/best_model.pth"
    )


def setup_amp(config):
    """混合精度初始化"""
    if config.use_amp:
        return GradScaler('cuda')
    return None


def save_model(model, path):
    """保存模型（兼容多GPU）"""
    if isinstance(model, DataParallel):
        torch.save(model.module.state_dict(), path)
    else:
        torch.save(model.state_dict(), path)


def load_model(model, path):
    """加载模型权重（兼容多GPU）"""
    checkpoint = torch.load(path)
    if isinstance(model, DataParallel):
        model.module.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    return model