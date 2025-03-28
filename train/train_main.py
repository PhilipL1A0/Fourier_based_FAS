import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *
from test_main import test
from configs import Config
from models import ResNet18


def train():
    # 初始化配置
    config = Config()

    # 确保输出目录存在
    os.makedirs(config.output_dir, exist_ok=True)
    model_save_path = os.path.join(config.output_dir, 'model', config.model_name)
    config_save_path = os.path.join(config.output_dir, "cfg", f"{config.model_name}.json")
    
    # 设置记录文件
    save_config(config, config_save_path)
    logger = setup_logger(config.output_dir, config.model_name)

    # 保存模型配置到日志
    logger.info("Model Configuration:")
    logger.info(f"Backbone: {config.backbone}, Attention: {config.attention_type}")
    logger.info(f"Input Channels: {config.input_channels}, Dropout: {config.dropout}")
    logger.info(f"Learning Rate: {config.lr}, Batch Size: {config.batch_size}, Optimizer: AdamW")
    logger.info(f"Weight Decay: {config.weight_decay}, Loss Function: {config.loss_func}")
    
    # 数据集信息日志
    if config.dataset == "all":
        logger.info("Dataset: ALL (混合训练模式)")
    else:
        logger.info(f"Dataset: {config.dataset}")
    
    # 准备数据集
    train_loader = load_dataset(config, 'train')
    # 验证集
    val_loader = load_dataset(config, 'val')
    
    # 数据集加载完成的日志
    if config.dataset == "all":
        logger.info(f"成功加载混合数据集: 训练集大小 {len(train_loader.dataset)} 样本")
    else:
        logger.info(f"成功加载数据集 {config.dataset}: 训练集大小 {len(train_loader.dataset)} 样本")

    # 初始化模型
    model = ResNet18(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        dropout_rate=config.dropout
    )
    model, device = setup_device(config, model)

    # 统计参数量
    param_stats = count_parameters(model)
    logger.info(f"Total parameters: {param_stats['total_params']:,}")
    logger.info(f"Trainable parameters: {param_stats['trainable_params']:,}")

    # 初始化优化器、调度器和早停机制
    optimizer = setup_optimizer(model, config)
    scheduler = setup_scheduler(optimizer, config)
    early_stopping = setup_early_stopping(config, model_save_path)

    # 初始化混合精度
    scaler = setup_amp(config)

    # 损失函数 - 针对混合数据集计算类权重
    if config.loss_func == "cross_entropy":
        criterion = nn.CrossEntropyLoss(weight=get_class_weights(train_loader.dataset)).to(device)
    elif config.loss_func == "focal_loss":
        criterion = FocalLoss(alpha=0.3, gamma=2.0).to(device)
    else:
        logger.info(f"未知损失函数: {config.loss_func}，使用默认CrossEntropy")
        criterion = nn.CrossEntropyLoss().to(device)

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
    plot_confusion_matrix(targets, preds, class_names=["spoofing", "living"], save_path=os.path.join(config.output_dir, "img", "train_cm", f"{config.model_name}.png"))
    logger.info("Confusion matrix plot saved.")
    logger.info("Training completed.")
    logger.handlers.clear()

    # 测试模型
    if config.test_model:
        test()

if __name__ == "__main__":
    train()