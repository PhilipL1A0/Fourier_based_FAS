import os
import sys
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from models import ResNet18
from utils import (setup_device, FourierDataset, 
                   load_trained_model, test_model,
                   compute_metrics, plot_confusion_matrix
                   )

def test(model_path=None):
    # 初始化配置
    config = Config()

    # 准备测试集
    test_data = FourierDataset(split_name='test', data_dir=config.data_dir)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # 初始化模型
    model = ResNet18(num_classes=config.num_classes, input_channels=config.input_channels)
    model, device = setup_device(config, model)

    # 加载训练好的模型
    model = load_trained_model(model, model_path, device)

    # 测试模型
    targets, preds = test_model(model, test_loader, device)

    # 评估模型
    metrics = compute_metrics(targets, preds)
    print("Test Metrics:", metrics)

    plot_confusion_matrix(targets, preds, class_names=["spoofing","living"], save_path=os.path.join(config.output_dir, "img", "cm", "test_"+config.model_name+".png"))
    


if __name__ == "__main__":
    model_path = "/media/main/lzf/FBFAS/outputs/best_model.pth"
    test(model_path)