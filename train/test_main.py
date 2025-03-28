import os
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from models import ResNet18
from utils import *

def test():
    # 初始化配置
    config = Config()
    model_path = os.path.join(config.output_dir, 'model', f"{config.model_name}.pth")
    logger = setup_logger(config.output_dir, config.model_name)

    # 准备测试集
    logger.info(f"Dataset: {config.dataset}")
    test_loader = load_dataset(config, 'test')
    logger.info(f"成功加载测试集: 测试集大小 {len(test_loader.dataset)} 样本")

    # 初始化模型
    model = ResNet18(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        dropout_rate=config.dropout
        )
    model, device = setup_device(config, model)

    # 加载训练好的模型
    model = load_trained_model(model, model_path, device)

    # 测试模型
    targets, preds = test_model(model, test_loader, device)

    # 评估模型
    metrics = compute_metrics(targets, preds)
    logger.info("Test Metrics:")
    logger.info(metrics)

    plot_confusion_matrix(targets, preds, class_names=["spoofing","living"], save_path=os.path.join(config.output_dir, "img", "test_cm", f"{config.model_name}.png"))
    logger.info("Confusion matrix plot saved.")
    logger.info("Test finished.")
    
    
if __name__ == "__main__":
    test()