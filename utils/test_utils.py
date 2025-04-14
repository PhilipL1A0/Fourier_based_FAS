import torch
from tqdm import tqdm

def load_trained_model(model, model_path, device):
    """
    加载训练好的模型权重
    
    Args:
        model (nn.Module): 模型实例
        model_path (str): 模型权重文件路径
        device (torch.device): 计算设备
        
    Returns:
        nn.Module: 加载权重后的模型
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    # 处理可能的 DataParallel 包装
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 如果保存的是完整检查点，提取模型状态字典
        checkpoint = checkpoint['model_state_dict']
    
    # 如果有"module."前缀(DataParallel)，处理兼容性
    if all(k.startswith('module.') for k in checkpoint.keys()):
        # 创建新的状态字典，移除'module.'前缀
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k  # 移除 'module.' 前缀
            new_state_dict[name] = v
        checkpoint = new_state_dict
    
    # 加载参数字典，处理可能缺失的键
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as e:
        print(f"警告: 模型加载时出现非严格匹配，将使用非严格模式加载")
        # 尝试非严格加载
        model.load_state_dict(checkpoint, strict=False)
    
    return model


def test_model(model, test_loader, device, config=None):
    """测试模型并返回目标值和预测值"""
    model.eval()
    targets, preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试中", ncols=100):
            # 确保batch中包含所需的数据键
            has_spatial = 'spatial' in batch
            has_freq = 'freq' in batch
            has_combined = 'combined' in batch
            
            # 处理双流网络输入
            if config.model_type == "DualStreamNetwork":
                # 根据可用数据提供输入
                spatial_input = batch['spatial'].to(device) if has_spatial else None
                freq_input = batch['freq'].to(device) if has_freq else None
                combined_input = batch['combined'].to(device) if has_combined else None
                
                # 确保至少有一种输入
                if not (spatial_input is not None or freq_input is not None or combined_input is not None):
                    raise ValueError("数据批次中缺少所需的输入: 至少需要'spatial'、'freq'或'combined'中的一种")
                
                # 使用灵活的前向传播
                outputs = model(spatial_input, freq_input, combined_input)
            # 处理单流网络输入
            elif config and config.use_multi_channel and config.data_mode == "both":
                inputs = batch['combined'].to(device)
                outputs = model(inputs)
            elif config and config.data_mode == "spatial":
                inputs = batch['spatial'].to(device)
                outputs = model(inputs)
            elif config and config.data_mode == "frequency" or not config:
                inputs = batch['freq'].to(device)
                outputs = model(inputs)
            else:
                # 默认使用频域数据
                inputs = batch['freq'].to(device)
                outputs = model(inputs)
            
            labels = batch['label'].to(device)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    
    return targets, preds