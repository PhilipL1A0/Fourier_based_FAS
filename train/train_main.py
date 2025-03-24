import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from models import ResNet18
from utils import (
    FourierDataset, 
    plot_training_history, compute_metrics,
    setup_device, setup_optimizer, setup_scheduler, setup_early_stopping,
    setup_amp, save_model
)


def train():
    # 初始化配置
    config = Config()

    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)

    # 准备数据集
    train_data = FourierDataset(split_name='train', data_dir=config.data_dir)
    val_data = FourierDataset(split_name='val', data_dir=config.data_dir)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 初始化模型
    model = ResNet18(num_classes=config.num_classes, input_channels=config.input_channels)
    model, device = setup_device(config, model)

    # 初始化优化器、调度器和早停机制
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    early_stopping = setup_early_stopping(config)

    # 初始化混合精度
    scaler = setup_amp(config)

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 训练过程
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    for epoch in range(config.epochs):
        print(f"Epoch {epoch + 1}/{config.epochs}")

        # 训练阶段
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_progress = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", ncols=100)
        for batch in train_progress:
            inputs, labels = batch['freq'].to(device), batch['label'].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(enabled=config.use_amp):
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

            # 更新进度条信息
            train_progress.set_postfix(loss=loss.item(), acc=correct / total)

        train_loss /= total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        val_progress = tqdm(val_loader, desc=f"Validating Epoch {epoch + 1}", ncols=100)
        with torch.no_grad():
            for batch in val_progress:
                inputs, labels = batch['freq'].to(device), batch['label'].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # 更新进度条信息
                val_progress.set_postfix(loss=loss.item(), acc=correct / total)

        val_loss /= total
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 学习率调度
        scheduler.step()

        # 早停机制
        if config.use_early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

    # 保存最终模型
    save_model(model, os.path.join(config.output_dir, 'model', config.model_name))

    # 绘制训练曲线
    plot_training_history(history, save_path=os.path.join(config.output_dir, "img", config.model_name, ".png"))

    # 评估模型
    print("Training completed. Evaluating model...")
    targets, preds = [], []
    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch['freq'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())

    metrics = compute_metrics(targets, preds)
    print("Evaluation Metrics:", metrics)


if __name__ == "__main__":
    train()