import os
import json
from cv2 import resize
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as ts
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class FASDataset(Dataset):
    """
    人脸反欺骗数据集：支持空域、频域和多通道模式
    """
    def __init__(self, split_name, config, data_dir="dataset", config_dir="configs"):
        # 基础配置
        self.split_name = split_name
        self.data_dir = data_dir
        self.dataset_name = config.dataset
        self.data_mode = config.data_mode
        self.compress_data = config.compress_data
        
        # 多通道配置 (RGB + 频域)
        self.use_multi_channel = getattr(config, 'use_multi_channel', False)
        
        # 数据增强配置
        self.use_augment = getattr(config, 'use_augment', True) and split_name == 'train'
        self.noise_std = getattr(config, 'noise_std', 0.01)
        
        # 加载数据集统计信息
        with open(f"{config_dir}/dataset_stats.json", "r") as f:
            all_stats = json.load(f)
            self.stats = all_stats.get(self.dataset_name, list(all_stats.values())[0])
        
        # 加载数据集划分
        with open(f"{config_dir}/splits.json", "r") as f:
            all_splits = json.load(f)
            splits = all_splits.get(self.dataset_name, list(all_splits.values())[0])
            self.image_paths = splits[split_name][0]
            self.labels = splits[split_name][1]
        
        # 创建变换
        self._create_transforms()
    
    def _create_transforms(self):
        """创建数据变换，区分增强和非增强情况"""
        mean, std = self.stats['spatial']['mean'], self.stats['spatial']['std']
        
        # 空域和频域使用不同的变换
        if self.use_augment:
            self.spatial_transform = ts.Compose([
                ts.Resize((112, 112)),
                ts.RandomResizedCrop((96, 96), scale=(0.9, 1.0)),  # 减小裁剪范围
                ts.RandomHorizontalFlip(),  # 保留水平翻转（人脸左右是对称的）
                ts.RandomRotation(10)  # 减小旋转角度
            ])
            
            self.freq_transform = ts.Resize((96, 96))
            
            # 仅用于彩色图像的颜色变换
            self.color_transform = ts.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03  # 减小参数
            )
        else:
            # 非增强情况下，简单调整大小
            self.spatial_transform = self.freq_transform = ts.Resize((96, 96))
            self.color_transform = None
        
        # 最终变换
        self.final_transform = ts.Compose([
            ts.ToTensor(),
            ts.Normalize(mean, std)
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 准备结果字典和加载标签
        path = self.image_paths[idx]
        label = self.labels[idx]
        result = {'label': torch.tensor(label, dtype=torch.long)}
        
        try:
            # spatial
            if self.data_mode in ["spatial", "both"]:
                
                img = Image.open(path).convert("RGB")
                img = self.spatial_transform(img)
                if self.color_transform and self.use_augment:
                    img = self.color_transform(img)
                img = self.final_transform(img)
                
                # 保存RGB图像
                result['spatial'] = img
            
            # frequency
            if self.data_mode in ["frequency", "both"]:
                # 获取频域数据路径
                base_name = os.path.basename(path).split('.')[0]
                dataset_dir = self.dataset_name if self.dataset_name else ""
                ext = 'npz' if self.compress_data else 'npy'
                freq_path = os.path.join(self.data_dir, "frequency", dataset_dir, f"{base_name}.{ext}")
                
                # 加载频域数据
                if self.compress_data:
                    freq_data = np.load(freq_path)['data']
                else:
                    freq_data = np.load(freq_path)
                
                if self.use_augment:
                    noise = np.random.normal(0, self.noise_std, freq_data.shape)
                    freq_data = freq_data + noise
                
                # 转化为PIL并应用变换
                freq_pil = self._numpy_to_pil(freq_data)
                freq_pil = self.freq_transform(freq_pil)
                freq_tensor = ts.ToTensor()(freq_pil)
                result['freq'] = freq_tensor
            
            # 3. 创建多通道输入 (如果需要)
            if self.use_multi_channel and self.data_mode == "both":
                if 'spatial' in result and 'freq' in result:
                    result['combined'] = torch.cat([result['spatial'], result['freq']], dim=0)
        
        except Exception as e:
            # 异常处理：创建零张量
            if self.data_mode in ["spatial", "both"] and 'spatial' not in result:
                result['spatial'] = torch.zeros((3, 96, 96), dtype=torch.float)
            
            if self.data_mode in ["frequency", "both"] and 'freq' not in result:
                result['freq'] = torch.zeros((1, 96, 96), dtype=torch.float)
            
            if self.use_multi_channel and self.data_mode == "both" and 'combined' not in result:
                if 'spatial' in result and 'freq' in result:
                    result['combined'] = torch.cat([result['spatial'], result['freq']], dim=0)
                else:
                    result['combined'] = torch.zeros((4, 96, 96), dtype=torch.float)
            
            print(f"Error processing {path}: {str(e)}")
        
        return result
    
    def _numpy_to_pil(self, array):
        """将numpy数组转换为PIL图像，适用于频域数据"""
        if array.min() < 0 or array.max() > 1:
            # 归一化到[0,1]
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        
        # 缩放到[0,255]并转换为uint8
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array)


def create_dataloader(dataset, batch_size=32, is_training=True, num_workers=4):
    """创建数据加载器"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_training
    )


def load_dataset(config, category):
    """根据配置加载数据集"""
    is_training = category == 'train'
    
    # 单个数据集情况
    if config.dataset != "all":
        dataset = FASDataset(
            split_name=category,
            config=config,
            data_dir=config.data_dir,
            config_dir="configs"
        )
        
        return create_dataloader(
            dataset, 
            batch_size=config.batch_size, 
            is_training=is_training,
            num_workers=config.num_workers
        )
    
    # 多数据集情况 (all)
    else:
        all_datasets = []
        
        # 获取所有可用数据集
        with open("configs/splits.json", 'r') as f:
            all_splits = json.load(f)
        
        # 为每个数据集创建实例
        for dataset_name in all_splits.keys():
            # 创建临时配置副本
            temp_config = copy_config(config)
            temp_config.dataset = dataset_name
            
            # 创建数据集
            dataset = FASDataset(
                split_name=category,
                config=temp_config,
                data_dir=config.data_dir,
                config_dir="configs"
            )
            
            all_datasets.append(dataset)
        
        # 合并所有数据集
        combined_dataset = ConcatDataset(all_datasets)
        
        return create_dataloader(
            combined_dataset, 
            batch_size=config.batch_size, 
            is_training=is_training,
            num_workers=config.num_workers
        )


def copy_config(config):
    """创建配置对象的副本，避免修改原始对象"""
    import copy
    return copy.deepcopy(config)