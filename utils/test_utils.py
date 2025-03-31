import torch
from tqdm import tqdm

def load_trained_model(model, model_path, device):
    """
    加载训练好的模型权重。

    Args:
        model (torch.nn.Module): 模型实例。
        model_path (str): 模型权重文件路径。
        device (torch.device): 设备（CPU 或 GPU）。

    Returns:
        torch.nn.Module: 加载权重后的模型。
    """
    checkpoint = torch.load(model_path)
    if not any(key.startswith('module.') for key in checkpoint.keys()):
        checkpoint = {'module.' + key: value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # 设置为评估模式
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