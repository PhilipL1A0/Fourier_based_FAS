import os
import sys
import csv
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from models import ResNet18
from utils import (
    FourierDataset, 
    plot_training_history, compute_metrics, plot_confusion_matrix,
    setup_device, setup_optimizer, setup_scheduler, setup_early_stopping,
    setup_amp, save_model, setup_logger
)


def train():
    # 初始化配置
    config = Config()

    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    model_save_path = os.path.join(config.output_dir, 'model', config.model_name)

    # 设置日志记录器和 CSV 文件
    logger, csv_path = setup_logger(config.output_dir, config.model_name)

    # 保存模型配置到日志
    logger.info("Model Configuration:")
    logger.info(f"Backbone: {config.backbone}, Attention: {config.attention_type}")
    logger.info(f"Input Channels: {config.input_channels}, Classes: {config.num_classes}")
    logger.info(f"Learning Rate: {config.lr}, Batch Size: {config.batch_size}, Optimizer: AdamW")
    logger.info(f"Weight Decay: {config.weight_decay}, Loss Function: CrossEntropyLoss")

    # 准备数据集
    train_data = FourierDataset(split_name='train', data_dir=config.data_dir, add_noise=config.use_augmentation)
    val_data = FourierDataset(split_name='val', data_dir=config.data_dir)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 初始化模型
    model = ResNet18(num_classes=config.num_classes, input_channels=config.input_channels)
    model, device = setup_device(config, model)

    # 初始化优化器、调度器和早停机制
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    early_stopping = setup_early_stopping(config, model_save_path)

    # 初始化混合精度
    scaler = setup_amp(config)

    # 损失函数  
    class_count = torch.tensor([len(train_data) - sum(train_data.labels), sum(train_data.labels)]).float()
    class_weights = class_count / class_count.sum()
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # 训练过程
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # 训练阶段
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", ncols=100):
            inputs, labels = batch['freq'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                if config.use_gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}", ncols=100):
                inputs, labels = batch['freq'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存到 CSV 文件
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch + 1, train_loss, train_acc, val_loss, val_acc])

        # 学习率调度
        scheduler.step()

        # 早停机制
        if config.use_early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break

    # 保存最终模型
    save_model(model, model_save_path)
    logger.info(f"Model saved to {model_save_path}")

    # 绘制训练曲线
    plot_training_history(history, save_path=os.path.join(config.output_dir, "img", "curv", f"{config.model_name}.png"))
    logger.info("Training history plot saved.")

    # 评估模型
    targets, preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['freq'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    
    metrics = compute_metrics(targets, preds)
    logger.info("Validation Metrics:")
    logger.info(metrics)

    # 绘制混淆矩阵
    plot_confusion_matrix(targets, preds, class_names=["spoofing", "living"], save_path=os.path.join(config.output_dir, "img", "cm", f"train_{config.model_name}.png"))
    logger.info("Confusion matrix plot saved.")
    logger.info("Training completed.")
    logger.handlers.clear()

if __name__ == "__main__":
    train() 