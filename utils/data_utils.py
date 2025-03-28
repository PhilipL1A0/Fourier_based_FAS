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
                 dataset_name=None,    # 数据集名称，如"CASIA", "idiap"等
                 add_noise=False,  # 是否添加噪声
                 noise_std=0.01):  # 噪声标准差
        self.split_name = split_name
        self.data_dir = data_dir
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.dataset_name = dataset_name
        
        # 加载统计信息
        with open(config_dir + "/dataset_stats.json", "r") as f:
            all_stats = json.load(f)
        
        # 确保选择了正确的数据集统计信息
        if dataset_name is not None and dataset_name in all_stats:
            self.stats = all_stats[dataset_name]
        else:
            # 默认使用第一个数据集的统计信息（或可以调整为其他策略）
            self.stats = list(all_stats.values())[0]
            print(f"Warning: Using default dataset stats. Specified dataset '{dataset_name}' not found.")

        # 加载数据划分路径
        with open(config_dir + "/splits.json", 'r') as f:
            all_splits = json.load(f)
        
        # 确保选择了正确的数据集划分
        if dataset_name is not None and dataset_name in all_splits:
            splits = all_splits[dataset_name]
        else:
            # 默认使用第一个数据集的划分（或可以调整为其他策略）
            splits = list(all_splits.values())[0]
            print(f"Warning: Using default dataset splits. Specified dataset '{dataset_name}' not found.")
            
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
        
        # 构建频域特征路径 - 考虑数据集名称组织
        dataset_name = self.dataset_name if self.dataset_name else ""
        base_name = os.path.basename(spatial_path).split('.')[0]
        
        # 读取预生成的频域特征
        freq_path = os.path.join(
            self.data_dir, 
            "frequency",
            dataset_name,  # 按数据集名称组织
            f"{base_name}.npy"
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
    
# 在训练脚本中加载特定数据集
def create_dataloader(dataset, batch_size=32, is_training=True):
    """创建数据加载器"""
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=4,
        pin_memory=True,
        drop_last=is_training
    )

def load_dataset(config, category):
    """根据配置加载数据集"""
    dataset_paths = config.get_dataset_paths()
    dataset_stats = config.get_dataset_stats()
    
    add_noise = config.augment_freq_noise if category == 'train' else False
    if_training = True if category == 'train' else False
    # 如果使用单个数据集
    if config.dataset != "all":
        dataset_name = config.dataset
        
        # 创建数据集
        dataset = FourierDataset(
            split_name=category,
            data_dir=config.data_dir,
            config_dir="configs",
            dataset_name=dataset_name,
            add_noise=add_noise,
            # noise_std=config.noise_std
        )

        # 创建数据加载器
        loader = create_dataloader(
            dataset, 
            batch_size=config.batch_size,
            is_training=if_training
        )
        
        return loader
    else:
        # 处理所有数据集的情况
        all_datasets = []
        
        # 为每个数据集创建数据集对象
        for dataset_name in dataset_paths.keys():
            dataset = FourierDataset(
                split_name=category,
                data_dir=config.data_dir,
                config_dir="configs",
                dataset_name=dataset_name,
                add_noise=add_noise,
                # noise_std=config.noise_std
            )
            
            all_datasets.append(dataset)
        
            # 直接合并所有数据集
            combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
            
            loader = create_dataloader(
                combined_dataset, 
                batch_size=config.batch_size,
                is_training=if_training
            )

            return loader