import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

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