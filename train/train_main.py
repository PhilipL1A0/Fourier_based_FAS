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
    logger.info("="*20)
    logger.info(f"开始训练模型: {config.model_name}")
    logger.info("="*20)
    
    # 打印关键配置信息
    logger.info("模型配置:")
    logger.info(f"- 模型类型: {config.model_type}")
    if config.model_type == "DualStreamNetwork":
        logger.info(f"- 融合方式: {config.fusion_type}")
    
    logger.info(f"- 主干网络: {config.backbone}, 注意力机制: {config.attention_type}")
    logger.info(f"- 输入通道数: {config.input_channels}, Dropout率: {config.dropout}")
    
    # 数据集信息日志
    logger.info("数据集信息:")
    if config.dataset == "all":
        logger.info("- 使用混合数据集进行训练")
    else:
        logger.info(f"- 训练数据集: {config.dataset}")
    logger.info(f"- 数据模式: {config.data_mode}")
    
    # 训练参数信息
    logger.info("训练参数:")
    logger.info(f"- 学习率: {config.lr}, 批大小: {config.batch_size}")
    logger.info(f"- 权重衰减: {config.weight_decay}, 损失函数: {config.loss_func}")
    logger.info(f"- 最大训练轮次: {config.epochs}")
    
    # 准备数据集
    logger.info("加载数据集...")
    train_loader = load_dataset(config, 'train')
    val_loader = load_dataset(config, 'val')
    
    # 数据集加载完成的日志
    if config.dataset == "all":
        logger.info(f"已加载混合数据集: 训练集 {len(train_loader.dataset)} 样本, 验证集 {len(val_loader.dataset)} 样本")
    else:
        logger.info(f"已加载数据集 {config.dataset}: 训练集 {len(train_loader.dataset)} 样本, 验证集 {len(val_loader.dataset)} 样本")

    # 初始化模型
    logger.info("初始化模型...")
    model = create_model(config)
    model, device = setup_device(config, model)

    # 统计参数量
    param_stats = count_parameters(model)
    logger.info(f"模型参数: 总计 {param_stats['total_params']:,}, 可训练 {param_stats['trainable_params']:,}")

    # 设置优化器
    logger.info("配置优化器...")
    if config.pretrained:
        # 区分参数组
        backbone_params = []
        classifier_params = []
        
        # 收集预训练主干和分类器的参数
        for name, param in model.named_parameters():
            if 'fc' in name or 'classifier' in name:  # 分类器参数
                if param.requires_grad:
                    classifier_params.append(param)
            else:  # 主干网络参数
                if param.requires_grad:
                    backbone_params.append(param)
        
        # 使用不同学习率
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': config.lr * config.backbone_lr_ratio},  # 主干网络使用较小学习率
            {'params': classifier_params, 'lr': config.lr}  # 分类器使用完整学习率
        ], weight_decay=config.weight_decay)
        
        logger.info(f"使用差异化学习率: 主干网络 {config.lr * config.backbone_lr_ratio:.6f}, 分类器 {config.lr:.6f}")
    else:
        # 对于非预训练模型，使用统一学习率
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        
        logger.info(f"使用统一学习率: {config.lr:.6f}")

    # 设置学习率调度器
    scheduler = setup_scheduler(optimizer, config)
    if config.use_warmup and config.use_lr_cos:
        logger.info(f"使用预热({config.warmup_epochs}轮)+余弦退火学习率策略")
    else:
        logger.info("使用固定学习率策略")

    # 设置早停和混合精度
    early_stopping = setup_early_stopping(config, model_save_path)
    scaler = setup_amp(config)

    # 损失函数
    if config.loss_func == "cross_entropy":
        class_weights = get_class_weights(train_loader.dataset)
        logger.info(f"使用带权重的交叉熵损失，类别权重: {class_weights}")
        criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    elif config.loss_func == "focal_loss":
        # 对于严重不平衡的数据集，Focal Loss更合适
        class_weights = get_class_weights(train_loader.dataset)
        criterion = FocalLoss(alpha=class_weights, gamma=2.0).to(device)
        logger.info(f"使用Focal Loss，alpha: {class_weights}, gamma: 2.0")
    else:
        logger.warning(f"未知损失函数: {config.loss_func}，使用默认CrossEntropy")
        criterion = nn.CrossEntropyLoss().to(device)

    # 训练过程
    logger.info("="*20)
    logger.info("开始训练")
    logger.info("="*20)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(config.epochs):
        logger.info(f"Epoch {epoch + 1}/{config.epochs}")

        # 训练阶段
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        
        for batch in tqdm(train_loader, desc=f"训练", ncols=100):
            # 处理双流网络的输入数据
            if config.model_type == "DualStreamNetwork":
                # 双流网络需要同时输入空域和频域数据
                spatial_input = batch['spatial'].to(device)
                freq_input = batch['freq'].to(device)
                
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = model(spatial_input, freq_input)
                    labels = batch['label'].to(device)
                    loss = criterion(outputs, labels)
                    
                # 记录数据大小
                batch_size = spatial_input.size(0)
            else:
                # 原始单流网络的数据处理逻辑
                if config.use_multi_channel and config.data_mode == "both":
                    inputs = batch['combined'].to(device)
                elif config.data_mode == "spatial":
                    inputs = batch['spatial'].to(device)
                elif config.data_mode == "frequency":
                    inputs = batch['freq'].to(device)
                else:
                    # 默认使用频域数据
                    inputs = batch['freq'].to(device)
                    
                labels = batch['label'].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', enabled=config.use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # 记录数据大小
                batch_size = inputs.size(0)

            # 反向传播和优化步骤
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

            train_loss += loss.item() * batch_size
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss /= total
        train_acc = correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"验证", ncols=100):
                # 处理双流网络的输入数据
                if config.model_type == "DualStreamNetwork":  # 检查是否为双流网络
                    spatial_input = batch['spatial'].to(device)
                    freq_input = batch['freq'].to(device)
                    outputs = model(spatial_input, freq_input)
                    labels = batch['label'].to(device)
                    loss = criterion(outputs, labels)
                    
                    # 记录数据大小
                    batch_size = spatial_input.size(0)
                else:
                    # 原始单流网络的数据处理逻辑
                    if config.use_multi_channel and config.data_mode == "both":
                        inputs = batch['combined'].to(device)
                    elif config.data_mode == "spatial":
                        inputs = batch['spatial'].to(device)
                    elif config.data_mode == "frequency":
                        inputs = batch['freq'].to(device)
                    else:
                        # 默认使用频域数据
                        inputs = batch['freq'].to(device)
                        
                    labels = batch['label'].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 记录数据大小
                    batch_size = inputs.size(0)

                val_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        val_loss /= total
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 打印轮次结果
        logger.info(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.4f}")
        logger.info(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.4f}")
        
        # 学习率调度 - 简化调度更新
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        if abs(current_lr - new_lr) > 1e-8:
            if config.pretrained and len(optimizer.param_groups) > 1:
                logger.info(f"学习率调整: 主干 {optimizer.param_groups[0]['lr']:.6f}, 分类器 {optimizer.param_groups[1]['lr']:.6f}")
            else:
                logger.info(f"学习率调整: {new_lr:.6f}")

        # 早停机制
        if config.use_early_stopping:
            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                logger.info("触发早停机制，停止训练。")
                break

    # 保存最终模型
    save_model(model, model_save_path)
    logger.info(f"模型已保存至: {model_save_path}.pth")

    # 绘制训练曲线
    plot_path = os.path.join(config.output_dir, "img", "curv", f"{config.model_name}.png")
    plot_training_history(history, save_path=plot_path)
    logger.info(f"训练曲线已保存至: {plot_path}")

    # 评估模型
    logger.info("在验证集上进行最终评估...")
    targets, preds = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="最终评估", ncols=100):
            # 处理双流网络的输入数据
            if config.model_type == "DualStreamNetwork":  # 检查是否为双流网络
                spatial_input = batch['spatial'].to(device)
                freq_input = batch['freq'].to(device)
                outputs = model(spatial_input, freq_input)
            else:
                # 原始单流网络的数据处理逻辑
                if config.use_multi_channel and config.data_mode == "both":
                    inputs = batch['combined'].to(device)
                elif config.data_mode == "spatial":
                    inputs = batch['spatial'].to(device)
                elif config.data_mode == "frequency":
                    inputs = batch['freq'].to(device)
                else:
                    # 默认使用频域数据
                    inputs = batch['freq'].to(device)
                    
                outputs = model(inputs)
                
            labels = batch['label'].to(device)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    
    metrics = compute_metrics(targets, preds)
    logger.info("验证集评估指标:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"- {metric_name}: {metric_value:.4f}")

    logger.info("="*20)
    logger.info("训练完成!")
    logger.info("="*20)
    logger.handlers.clear()

    # 测试模型
    if config.test_model:
        test()

if __name__ == "__main__":
    train()