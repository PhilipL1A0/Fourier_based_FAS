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

def compute_metrics(targets, preds):
    cm = confusion_matrix(targets, preds)
    tp, fp, fn, tn = cm[1,1], cm[0,1], cm[1,0], cm[0,0]
    metrics = {
        'accuracy': accuracy_score(targets, preds),
        'precision': tp / (tp + fp + 1e-8),
        'recall': tp / (tp + fn + 1e-8),
        'F1': f1_score(targets, preds),
        'FAR': fp / (fp + tn + 1e-8),
        'FRR': fn / (tp + fn + 1e-8),
        'HTER': (fp/(fp+tn) + fn/(tp+fn))/2 if (fp+tn and tp+fn) else 0
    }
    return metrics