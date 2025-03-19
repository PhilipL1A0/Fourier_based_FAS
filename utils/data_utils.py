import os
import json
import torch
import numpy as np
import torchvision.transforms as ts
from PIL import Image
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


class FourierDataset(Dataset):
    def __init__(self, 
                 split_name,  # 数据集划分名称（'train', 'val', 'test'）
                 data_dir="preprocessed",  # 预处理后的数据根目录
                 stats_file="configs/dataset_stats.json",  # 统计信息文件路径
                 split_file="configs/splits.json"):  # 数据划分文件路径
        self.split_name = split_name
        self.data_dir = data_dir
        self.stats = json.load(open(stats_file))
        
        # 加载数据划分路径
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.image_paths = splits[split_name][0]
        self.labels = splits[split_name][1]
        
        # 设置数据增强
        self.transform = get_transform(split_name, 
                                      self.stats['spatial']['mean'],
                                      self.stats['spatial']['std'])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        spatial_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取空域图像
        img = Image.open(spatial_path).convert("RGB")
        img = self.transform(img)
        
        # 读取预生成的频域特征
        freq_path = os.path.join(
            self.data_dir, 
            "frequency", 
            f"{os.path.basename(spatial_path).split('.')[0]}.npy"
        )
        freq_data = np.load(freq_path)
        freq_tensor = torch.from_numpy(freq_data).float().unsqueeze(0)  # 添加通道维度
        
        return {
            'spatial': img,
            'freq': freq_tensor,
            'label': torch.tensor(label, dtype=torch.long)
        }

def get_transform(category, mean, std):
    if category == 'train':
        return ts.Compose([
            ts.Resize((128, 128)),
            ts.RandomResizedCrop((96, 96), scale=(0.8, 1.0)),
            ts.RandomHorizontalFlip(p=0.5),
            ts.RandomVerticalFlip(),
            ts.RandomRotation(30),
            ts.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
            ts.ToTensor(),
            ts.Normalize(mean, std)
        ])
    else:
        return ts.Compose([
            ts.Resize((96, 96)),
            ts.ToTensor(),
            ts.Normalize(mean, std)
        ])

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