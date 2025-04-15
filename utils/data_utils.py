import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as ts
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class FASDataset(Dataset):
    """
    人脸反欺骗数据集：支持空域、频域和多通道模式
    确保空域和频域数据的几何变换完全一致
    """
    def __init__(self, split_name, config, data_dir="dataset", config_dir="configs"):
        # 基础配置
        self.split_name = split_name
        self.data_dir = data_dir
        self.dataset_name = config.dataset
        self.data_mode = config.data_mode
        self.compress_data = config.compress_data
        
        # 多通道配置 (RGB + 频域)
        self.use_multi_channel = config.use_multi_channel
        
        # 数据增强配置
        self.is_training = split_name == "train"
        self.use_augment = config.use_augment and self.is_training
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
        self.spatial_mean = self.stats['spatial']['mean']
        self.spatial_std = self.stats['spatial']['std']
        
        # 为频域数据计算统计信息（如果没有，使用默认值）
        if 'frequency' in self.stats:
            self.freq_mean = self.stats['frequency']['mean']
            self.freq_std = self.stats['frequency']['std']
        else:
            self.freq_mean, self.freq_std = 0.5, 0.5
        
        # 非增强情况下的基础变换
        self.resize_transform = ts.Resize((96, 96))
        
        # 颜色变换（仅用于空间图像）
        if self.use_augment:
            self.color_transform = ts.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03
            )
        else:
            self.color_transform = None
        
        # 标准化变换
        self.spatial_normalize = ts.Normalize(self.spatial_mean, self.spatial_std)
        self.freq_normalize = ts.Normalize(self.freq_mean, self.freq_std)
    
    def _compute_frequency_features(self, img_tensor):
        """
        计算输入图像的频域特征
        
        Args:
            img_tensor: 形状为 [C, H, W] 的图像张量
            
        Returns:
            频域特征张量，形状为 [1, H, W]
        """
        # 将 RGB 转为灰度 (对频域特征使用灰度处理更常见)
        if img_tensor.shape[0] == 3:
            # 使用 RGB 到灰度的标准转换公式
            gray = 0.299 * img_tensor[0] + 0.587 * img_tensor[1] + 0.114 * img_tensor[2]
            img = gray.unsqueeze(0)  # 添加通道维度
        else:
            img = img_tensor
        
        # 将图像转换为 numpy 进行 FFT 处理
        img_np = img.numpy()[0]  # 移除通道维度
        
        # 执行二维傅里叶变换
        f_transform = np.fft.fft2(img_np)
        f_shift = np.fft.fftshift(f_transform)
        
        # 计算幅度谱（取绝对值），并应用对数变换增强可视化
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # 归一化到 [0, 1] 范围
        magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
        
        # 转回 torch tensor
        freq_tensor = torch.from_numpy(magnitude_spectrum).float().unsqueeze(0)  # 添加通道维度
        return freq_tensor
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 准备结果字典和加载标签
        path = self.image_paths[idx]
        label = self.labels[idx]
        result = {'label': torch.tensor(label, dtype=torch.long)}
        
        try:
            # 1. 加载原始图像
            img = Image.open(path).convert("RGB")
            
            # 2. 应用几何变换 (Resize以及可能的数据增强)
            img = self.resize_transform(img)  # 先调整大小到统一尺寸
            
            # 存储变换参数以保持一致性
            i, j, h, w = 0, 0, img.height, img.width  # 默认使用完整图像
            hflip = False
            angle = 0
            
            # 应用数据增强 (如果启用)
            if self.use_augment:
                # 随机裁剪 (存储参数)
                i, j, h, w = ts.RandomResizedCrop.get_params(
                    img, scale=(0.9, 1.0), ratio=(1.0, 1.0))
                
                # 随机水平翻转 (存储参数)
                hflip = torch.rand(1).item() < 0.5
                
                # 随机旋转角度 (存储参数)
                angle = torch.randint(-10, 10, (1,)).item()
            
            # 3. 应用几何变换到图像 (裁剪、翻转、旋转)
            img = TF.resized_crop(img, i, j, h, w, (96, 96))
            if hflip:
                img = TF.hflip(img)
            if angle != 0:
                img = TF.rotate(img, angle)
            
            # 4. 转换为张量 (保持几何变换后、颜色变换前的版本用于频域计算)
            img_tensor = TF.to_tensor(img)
            
            # 5. 根据数据模式处理不同的数据流
            if self.data_mode in ["spatial", "both"]:
                # 对于空间流，应用颜色变换(如果启用)
                spatial_img = img
                if self.color_transform and self.use_augment:
                    spatial_img = self.color_transform(spatial_img)
                
                # 转为张量并标准化
                spatial_tensor = TF.to_tensor(spatial_img)
                spatial_tensor = self.spatial_normalize(spatial_tensor)
                result['spatial'] = spatial_tensor
            
            if self.data_mode in ["frequency", "both"]:
                # 6. 计算频域特征 (基于已应用几何变换的图像)
                freq_tensor = self._compute_frequency_features(img_tensor)
                
                # 应用频域特定的增强: 添加噪声
                if self.use_augment:
                    noise = torch.randn_like(freq_tensor) * self.noise_std
                    freq_tensor = freq_tensor + noise
                
                # 标准化频域数据
                freq_tensor = self.freq_normalize(freq_tensor)
                result['freq'] = freq_tensor
            
            # 7. 创建多通道输入 (如果需要)
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