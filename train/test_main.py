import os
import sys
import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from models import ResNet18
from utils import *

def test():
    # 初始化配置
    config = Config()
    test_model = config.test_model
    model_path = os.path.join(config.output_dir, 'model', f"{config.model_name}.pth")
    logger = setup_logger(config.output_dir, config.model_name)

    # 准备测试集
    config.dataset = config.test_dataset
    logger.info(f"Dataset: {config.dataset}")
    test_loader = load_dataset(config, 'test')
    logger.info(f"成功加载测试集: 测试集大小 {len(test_loader.dataset)} 样本")

    # 初始化模型 - 注意需要与训练时使用的参数一致
    model = ResNet18(
        num_classes=config.num_classes,
        input_channels=config.input_channels,
        dropout_rate=config.dropout,
        pretrained=config.pretrained,  # 添加预训练参数
        use_attention=config.use_attention  # 确保注意力机制配置一致
    )
    model, device = setup_device(config, model)

    # 加载训练好的模型
    model = load_trained_model(model, model_path, device)
    logger.info(f"成功加载训练好的模型: {model_path}")

    # 测试模型
    model.eval()
    targets, preds, probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=100):
            # 根据配置选择输入数据
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
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            probs.extend(probabilities[:, 1].cpu().numpy())  # 取正类的概率值

    # 评估模型
    metrics = compute_metrics(targets, preds)
    logger.info("Test Metrics:")
    logger.info(metrics)

    # plot_confusion_matrix(targets, preds, class_names=["spoofing","living"], save_path=os.path.join(config.output_dir, "img", "test_cm", f"{config.model_name}.png"))
    
    eval_results = advanced_evaluation_plots(
        targets, probs,
        save_dir=os.path.join(config.output_dir, "img"),
        filename_prefix=test_model,
    )
    logger.info(f"Advanced evaluation metrics: AUC={eval_results['auc']:.4f}, AP={eval_results['ap']:.4f}")
    
    logger.info("Test finished.")
    logger.handlers.clear()
    
        
if __name__ == "__main__":
    test()