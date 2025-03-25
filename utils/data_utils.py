import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as ts
from torch.utils.data import Dataset


class FourierDataset(Dataset):
    def __init__(self, 
                 split_name,  # 数据集划分名称（'train', 'val', 'test'）
                 data_dir="dataset",  # 预处理后的数据根目录
                 config_dir="configs",  # 配置文件目录
                 add_noise=False,  # 是否添加噪声
                 noise_std=0.01):  # 数据划分文件路径
        self.split_name = split_name
        self.data_dir = data_dir
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.stats = json.load(open(config_dir + "/dataset_stats.json", "r"))

        # 加载数据划分路径
        with open(config_dir + "/splits.json", 'r') as f:
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

        # 为频域特征添加随机噪声
        if self.add_noise and self.split_name == 'train':  # 仅对训练集添加噪声
            noise = np.random.normal(0, self.noise_std, freq_data.shape)
            freq_data = freq_data + noise

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