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
        print(f"WARNING: 模型加载时出现非严格匹配: {e}")
        # 尝试非严格加载
        model.load_state_dict(checkpoint, strict=False)
        print("已使用非严格模式加载模型权重")
    
    return model


def test_model(model, test_loader, device, config=None):
    """测试模型并返回目标值和预测值"""
    model.eval()
    targets, preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=100):
            # 根据配置选择输入数据
            if config and config.use_multi_channel and config.data_mode == "both":
                inputs = batch['combined'].to(device)
            elif config and config.data_mode == "spatial":
                inputs = batch['spatial'].to(device)
            elif config and config.data_mode == "frequency" or not config:
                inputs = batch['freq'].to(device)
            else:
                # 默认使用频域数据
                inputs = batch['freq'].to(device)
            
            labels = batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    
    return targets, preds