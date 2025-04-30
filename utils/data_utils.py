import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as ts
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import Counter
import random
import os


class FASDataset(Dataset):
    """
    人脸反欺骗数据集：支持空域、频域和多通道模式
    确保空域和频域数据的几何变换完全一致
    """
    def __init__(self, split_name, config, data_dir="dataset", config_dir="configs", balance_data=False):
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
        
        try:
            # 加载数据集划分
            with open(f"{config_dir}/splits.json", "r") as f:
                all_splits = json.load(f)
                print(f"可用数据集: {list(all_splits.keys())}")
                
                splits = all_splits.get(self.dataset_name)
                if splits is None:
                    raise ValueError(f"在 splits.json 中未找到数据集 {self.dataset_name}")
                    
                print(f"数据集 {self.dataset_name} 可用模式: {list(splits.keys())}")
                
                mode = self.data_mode if self.data_mode in splits else 'spatial'
                print(f"使用模式: {mode}")
                
                if split_name not in splits[mode]:
                    raise ValueError(f"在 {mode} 模式下未找到分割 {split_name}")
                    
                split_data = splits[mode][split_name]
                if not isinstance(split_data, list) or len(split_data) != 2:
                    raise ValueError(f"数据格式错误: 应为 [图片列表, 标签列表]，实际为 {type(split_data)}")
                    
                self.image_paths = split_data[0]
                self.labels = split_data[1]
                
                print(f"加载数据集: {self.dataset_name}")
                print(f"分割: {split_name}")
                print(f"图片数量: {len(self.image_paths)}")
                print(f"标签数量: {len(self.labels)}")
                print(f"标签分布: {Counter(self.labels)}")
                
                # 添加数据一致性检查
                if len(self.image_paths) != len(self.labels):
                    raise ValueError(f"数据集 {self.dataset_name} 的 {split_name} 分割中，"
                                f"图片数量({len(self.image_paths)})与标签数量({len(self.labels)})不匹配")
                                
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise
        
        # 平衡数据（下采样）
        if balance_data:
            self._balance_data()
        
        # 创建变换
        self._create_transforms()
    
    def _balance_data(self):
        """
        对数据进行下采样以平衡类别分布，增加错误处理
        """
        try:
            # 转换为列表类型，确保索引操作安全
            self.image_paths = list(self.image_paths)
            self.labels = list(self.labels)
            
            # 统计每个类别的样本数
            label_counter = Counter(self.labels)
            if not label_counter:
                raise ValueError("没有有效的标签数据")
                
            # 找到最小类别的样本数
            min_count = min(label_counter.values())
            if min_count == 0:
                raise ValueError("存在样本数为0的类别")
                
            balanced_image_paths = []
            balanced_labels = []
            
            # 对每个类别进行下采样
            for label in label_counter.keys():
                # 获取当前类别的所有索引
                label_indices = [i for i, lbl in enumerate(self.labels) if lbl == label]
                
                if not label_indices:
                    print(f"警告：类别 {label} 没有样本")
                    continue
                    
                # 安全采样
                sample_size = min(min_count, len(label_indices))
                sampled_indices = random.sample(label_indices, sample_size)
                
                # 使用列表推导式前确保索引有效
                valid_paths = []
                valid_labels = []
                for idx in sampled_indices:
                    if 0 <= idx < len(self.image_paths):
                        valid_paths.append(self.image_paths[idx])
                        valid_labels.append(self.labels[idx])
                    else:
                        print(f"警告：跳过无效索引 {idx}")
                
                balanced_image_paths.extend(valid_paths)
                balanced_labels.extend(valid_labels)
            
            if not balanced_image_paths:
                raise ValueError("平衡后没有剩余有效样本")
                
            # 更新数据集
            self.image_paths = balanced_image_paths
            self.labels = balanced_labels
            
            print(f"平衡后的数据集统计: {Counter(self.labels)}")
            
        except Exception as e:
            print(f"数据平衡失败: {str(e)}")
            # 保持原始数据不变
            return
    
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
        
        # 非增强情况下的基础变换 - 固定大小为96x96
        self.img_size = 96
        self.resize_transform = ts.Resize((self.img_size, self.img_size))
        
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
    
    def _get_file_path(self, filename, mode='spatial'):
        """构造完整的文件路径"""
        return os.path.join(
            self.data_dir,
            'dataset',
            mode,
            self.dataset_name,
            self.split_name,
            filename
        )
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 准备结果字典和加载标签
        filename = self.image_paths[idx]
        label = self.labels[idx]
        result = {'label': torch.tensor(label, dtype=torch.long)}
        
        try:
            # 1. 空域数据处理
            if self.data_mode in ["spatial", "both"]:
                spatial_path = self._get_file_path(filename, 'spatial')
                img = Image.open(spatial_path).convert("RGB")
                
                # 应用几何变换 (Resize以及可能的数据增强)
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
                
                # 应用几何变换到图像 (裁剪、翻转、旋转)
                img = TF.resized_crop(img, i, j, h, w, (self.img_size, self.img_size))
                if hflip:
                    img = TF.hflip(img)
                if angle != 0:
                    img = TF.rotate(img, angle)
                
                # 转换为张量 (保持几何变换后、颜色变换前的版本用于频域计算)
                img_tensor = TF.to_tensor(img)
                
                # 对于空间流，应用颜色变换(如果启用)
                spatial_img = img
                if self.color_transform and self.use_augment:
                    spatial_img = self.color_transform(spatial_img)
                
                # 转为张量并标准化
                spatial_tensor = TF.to_tensor(spatial_img)
                spatial_tensor = self.spatial_normalize(spatial_tensor)
                result['spatial'] = spatial_tensor
            
            # 2. 频域数据处理
            if self.data_mode in ["frequency", "both"]:
                # 对于频域数据，需要改变文件扩展名
                freq_filename = os.path.splitext(filename)[0] + ('.npz' if self.compress_data else '.npy')
                freq_path = self._get_file_path(freq_filename, 'frequency')
                
                try:
                    # 加载频域数据
                    if self.compress_data:
                        freq_data = np.load(freq_path)['data']
                    else:
                        freq_data = np.load(freq_path)
                    
                    # 确保频域数据的大小与空域数据一致
                    freq_tensor = torch.from_numpy(freq_data).float()
                    
                    # 确保频域数据是单通道的
                    if len(freq_tensor.shape) == 2:  # 如果是2D张量 (H, W)
                        freq_tensor = freq_tensor.unsqueeze(0)  # 变为 (1, H, W)
                    elif len(freq_tensor.shape) > 3:  # 如果维度过多
                        freq_tensor = freq_tensor.squeeze()  # 移除多余维度
                        if len(freq_tensor.shape) == 2:
                            freq_tensor = freq_tensor.unsqueeze(0)
                    
                    # 确保频域数据的高宽与空域数据一致
                    if freq_tensor.shape[1] != self.img_size or freq_tensor.shape[2] != self.img_size:
                        freq_tensor = TF.resize(freq_tensor, (self.img_size, self.img_size))
                    
                    # 应用频域特定的增强: 添加噪声
                    if self.use_augment:
                        noise = torch.randn_like(freq_tensor) * self.noise_std
                        freq_tensor = freq_tensor + noise
                    
                    # 标准化频域数据
                    freq_tensor = self.freq_normalize(freq_tensor)
                    result['freq'] = freq_tensor
                
                except Exception as e:
                    print(f"Error loading frequency data {freq_path}: {str(e)}")
                    # 创建空白的频域数据
                    result['freq'] = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float)
            
            # 3. 创建多通道输入 (如果需要)
            if self.use_multi_channel and self.data_mode == "both":
                if 'spatial' in result and 'freq' in result:
                    # 验证两个张量的形状是否兼容
                    spatial_shape = result['spatial'].shape
                    freq_shape = result['freq'].shape
                    
                    if spatial_shape[1:] != freq_shape[1:]:
                        print(f"Warning: Shape mismatch - spatial: {spatial_shape}, freq: {freq_shape}")
                        # 调整频域数据大小以匹配空域数据
                        result['freq'] = TF.resize(result['freq'], (spatial_shape[1], spatial_shape[2]))
                    
                    # 连接两个张量
                    result['combined'] = torch.cat([result['spatial'], result['freq']], dim=0)
        
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            
            # 异常处理：创建零张量
            if self.data_mode in ["spatial", "both"] and 'spatial' not in result:
                result['spatial'] = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float)
            
            if self.data_mode in ["frequency", "both"] and 'freq' not in result:
                result['freq'] = torch.zeros((1, self.img_size, self.img_size), dtype=torch.float)
            
            if self.use_multi_channel and self.data_mode == "both" and 'combined' not in result:
                result['combined'] = torch.zeros((4, self.img_size, self.img_size), dtype=torch.float)
        
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
            config_dir="configs",
            balance_data=config.balance_data and is_training
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
                config_dir="configs",
                balance_data=config.balance_data and is_training
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