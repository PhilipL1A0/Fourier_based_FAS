import os
import sys
import torch
from tqdm import tqdm
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from utils import *

def test():
    # 初始化配置
    config = Config()
    test_model = config.test_model
    model_path = os.path.join(config.output_dir, 'model', f"{config.model_name}.pth")
    logger = setup_logger(config.output_dir, config.model_name + "_test")

    # 打印测试配置
    logger.info("="*20)
    logger.info(f"开始测试模型: {config.model_name}")
    logger.info("="*20)
    
    logger.info(f"测试数据集: {config.test_dataset}")
    logger.info(f"模型路径: {model_path}")
    
    # 准备测试集
    config.dataset = config.test_dataset
    test_loader = load_dataset(config, 'test')
    logger.info(f"测试集样本数: {len(test_loader.dataset)}")

    # 初始化模型
    logger.info("加载模型...")
    model = create_model(config)
    model, device = setup_device(config, model)

    # 加载训练好的模型
    try:
        model = load_trained_model(model, model_path, device)
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        return

    # 测试模型
    logger.info("开始测试...")
    model.eval()
    targets, preds, probs = [], [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="测试进度", ncols=100):
            # 处理双流网络的输入数据
            if config.model_type == "DualStreamNetwork":
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
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            targets.extend(labels.cpu().numpy())
            preds.extend(predicted.cpu().numpy())
            probs.extend(probabilities[:, 1].cpu().numpy())  # 取正类的概率值

    # 评估模型
    metrics = compute_metrics(targets, preds)
    logger.info("测试集评估指标:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"- {metric_name}: {metric_value:.4f}")

    # 高级评估指标计算
    eval_results = advanced_evaluation_plots(
        targets, probs,
        save_dir=os.path.join(config.output_dir, "img"),
        filename_prefix=test_model,
    )
    logger.info(f"高级评估指标: AUC={eval_results['auc']:.4f}, AP={eval_results['ap']:.4f}")
    
    logger.info(""+"="*20)
    logger.info("测试完成!")
    logger.info("="*20)
    logger.handlers.clear()
    
        
if __name__ == "__main__":
    test()