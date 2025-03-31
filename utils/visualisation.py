import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc

def plot_training_history(history, save_path=None):
    """
    Visualize the training process.

    Args:
        history (dict): A dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'.
        save_path (str, optional): Path to save the plot. If None, the plot will be shown.
    """
    epochs = range(1, len(history['train_loss']) + 1)

    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def compute_metrics(targets, preds, display_cm=False):
    """
    计算评估指标并可选显示混淆矩阵。

    Args:
        targets (list): 真实标签。
        preds (list): 预测标签。
        display_cm (bool): 是否在终端打印混淆矩阵。

    Returns:
        dict: 包含评估指标的字典。
    """
    cm = confusion_matrix(targets, preds)
    tp, fp, fn, tn = cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision': tp / (tp + fp + 1e-8),
        'recall': tp / (tp + fn + 1e-8),
        'F1': f1_score(targets, preds),
        'FAR': fp / (fp + tn + 1e-8),
        'FRR': fn / (tp + fn + 1e-8),
        'HTER': (fp / (fp + tn) + fn / (tp + fn)) / 2 if (fp + tn and tp + fn) else 0
    }

    if display_cm:
        print("Confusion Matrix:")
        print(cm)

    return metrics


def plot_confusion_matrix(targets, preds, class_names, save_path=None):
    """
    绘制混淆矩阵的热力图。

    Args:
        targets (list): 真实标签。
        preds (list): 预测标签。
        class_names (list): 类别名称列表。
        save_path (str, optional): 保存图片的路径。如果为 None，则直接显示图片。
    """
    cm = confusion_matrix(targets, preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在每个格子中显示数值
    thresh = cm_normalized.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, f"{cm[i, j]}\n{cm_normalized[i, j]:.2f}",
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()


def plot_roc_curve(targets, probs, save_path=None):
    """
    绘制ROC曲线并计算AUC值。

    Args:
        targets (list): 真实标签，二分类(0,1)。
        probs (list): 预测为正类(1)的概率值。
        save_path (str, optional): 保存图片的路径。如果为 None，则直接显示图片。

    Returns:
        float: AUC值
    """
    # 确保输入类型正确
    targets = np.array(targets)
    probs = np.array(probs)
    
    # 计算ROC曲线的点
    fpr, tpr, thresholds = roc_curve(targets, probs)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC_Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC曲线已保存至 {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(targets, probs, save_path=None):
    """
    绘制精确率-召回率曲线。

    Args:
        targets (list): 真实标签，二分类(0,1)。
        probs (list): 预测为正类(1)的概率值。
        save_path (str, optional): 保存图片的路径。如果为 None，则直接显示图片。
        
    Returns:
        float: 平均精确率(AP)值
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    # 计算精确率-召回率曲线
    precision, recall, _ = precision_recall_curve(targets, probs)
    ap_score = average_precision_score(targets, probs)
    
    # 绘制精确率-召回率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR_Curve (AP = {ap_score:.4f})')
    plt.fill_between(recall, precision, alpha=0.2, color='blue')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR Curve')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if save_path:
        plt.savefig(save_path)
        print(f"PR曲线已保存至 {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return ap_score


def advanced_evaluation_plots(targets, probs, save_dir=None, filename_prefix=''):
    """
    生成一系列高级评估图表（ROC曲线、PR曲线）。

    Args:
        targets (list): 真实标签列表。
        probs (list): 预测概率列表。
        save_dir (str): 保存图表的目录。
        filename_prefix (str): 文件名前缀。

    Returns:
        dict: 包含各种评估指标的字典。
    """
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    results = {}
    
    # ROC曲线和AUC
    roc_path = os.path.join(save_dir, "roc", f"{filename_prefix}.png") if save_dir else None
    roc_auc = plot_roc_curve(targets, probs, save_path=roc_path)
    results['auc'] = roc_auc
    
    # 精确率-召回率曲线和AP
    pr_path = os.path.join(save_dir, "pr", f"{filename_prefix}.png") if save_dir else None
    ap_score = plot_precision_recall_curve(targets, probs, save_path=pr_path)
    results['ap'] = ap_score
    
    return results