import torch
from tqdm import tqdm
from torch.nn import DataParallel

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
    if any(key.startswith('module.') for key in checkpoint.keys()):
        checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()  # 设置为评估模式
    return model


def test_model(model, test_loader, device):
    """
    测试模型性能。

    Args:
        model (torch.nn.Module): 已加载权重的模型。
        test_loader (torch.utils.data.DataLoader): 测试集数据加载器。
        device (torch.device): 设备（CPU 或 GPU）。

    Returns:
        list, list: 真实标签列表，预测标签列表。
    """
    targets, preds = [], []
    test_progress = tqdm(test_loader, desc="Testing", ncols=100)
    with torch.no_grad():
        for batch in test_progress:
            inputs, labels = batch['freq'].to(device), batch['label'].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
    return targets, preds