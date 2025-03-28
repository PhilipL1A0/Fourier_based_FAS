import os
import sys
import cv2
import json
import time
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config import Config
from sklearn.model_selection import train_test_split


def split_datasets(data_dirs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 修改函数返回值，添加数据集名称信息
    all_splits = {}
    
    for data_dir in data_dirs:
        dataset_name = os.path.basename(data_dir)
        all_images = []
        all_labels = []
        
        for category in ['living', 'spoofing']:
            label = 1 if category == 'living' else 0
            img_dir = os.path.join(data_dir, category)
            if not os.path.exists(img_dir):
                print(f"Warning: {img_dir} does not exist. Skipping.")
                continue
                
            for img_name in os.listdir(img_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.bmp')):
                    img_path = os.path.join(img_dir, img_name)
                    all_images.append(img_path)
                    all_labels.append(label)
        
        if not all_images:
            print(f"Warning: No images found for dataset {dataset_name}. Skipping.")
            continue
            
        X = np.array(all_images)
        y = np.array(all_labels)
        
        # 分层划分数据集
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=1 - train_ratio, stratify=y, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=test_ratio/(val_ratio + test_ratio),
            stratify=y_temp, random_state=42
        )
        
        all_splits[dataset_name] = {
            'train': (X_train.tolist(), y_train.tolist()),
            'val': (X_val.tolist(), y_val.tolist()),
            'test': (X_test.tolist(), y_test.tolist())
        }
    
    return all_splits

def compute_mean_std(paths, is_spatial=True):
    """
    计算图像数据的均值和标准差，支持空域和频域。
    
    :param paths: 图像路径列表
    :param is_spatial: 是否为空域图像
    :return: 均值和标准差
    """
    total_pixels = 0
    sum_channels = None
    sum_squares = None

    for path in tqdm(paths, desc="Computing Mean and Std", ncols=100, leave=False):
        if is_spatial:
            img = cv2.imread(path)  # 读取 BGR 图像
            if img is None:
                print(f"Warning: Unable to read image {path}. Skipping.")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
            img = img.astype(np.float64) / 255.0  # 归一化到 [0, 1]
            h, w, c = img.shape
            if sum_channels is None:
                sum_channels = np.zeros(c, dtype=np.float64)
                sum_squares = np.zeros(c, dtype=np.float64)
            sum_channels += np.sum(img, axis=(0, 1))  # 每个通道的像素值求和
            sum_squares += np.sum(img**2, axis=(0, 1))  # 每个通道的像素值平方求和
            total_pixels += h * w  # 累加总像素数
        else:
            img = np.load(path)  # 假设频域特征以.npy格式保存
            if img is None or img.size == 0:
                print(f"Warning: Unable to read frequency data {path}. Skipping.")
                continue
            img = img.astype(np.float64)  # 确保高精度
            if sum_channels is None:
                sum_channels = 0.0
                sum_squares = 0.0
            sum_channels += np.sum(img)
            sum_squares += np.sum(img**2)
            total_pixels += img.size

    if total_pixels == 0:
        raise ValueError("Total pixels is zero. Check the input image paths or data.")

    mean = sum_channels / total_pixels  # 每个通道的均值
    variance = sum_squares / total_pixels - (mean ** 2)  # 每个通道的方差
    variance = np.maximum(variance, 0)  # 防止浮点误差导致负值
    std = np.sqrt(variance)  # 每个通道的标准差

    return mean.tolist(), std.tolist()

def generate_frequency_features(dataset_name, spatial_paths, output_base_dir, 
                               compress_data=True, precision="float32"):
    """
    按数据集生成频域特征，优化存储空间
    
    参数:
        dataset_name: 数据集名称
        spatial_paths: 空域图像路径
        output_base_dir: 输出目录
        compress_data: 是否使用压缩存储
        precision: 数据精度 ("float16", "float32")
    """
    output_dir = os.path.join(output_base_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    skipped_files = 0
    total_original_size = 0
    total_optimized_size = 0
    
    for path in tqdm(spatial_paths, 
                    desc=f"Generating Frequency Features for {dataset_name}", 
                    ncols=100, 
                    leave=False):
        try:
            # 生成频域特征并保存
            img_bgr = cv2.imread(path)
            # 检查图像是否成功读取
            if img_bgr is None:
                print(f"Warning: Unable to read image at {path}. Skipping file.")
                skipped_files += 1
                continue
                
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            fft = np.fft.fft2(img_gray)
            magnitude = np.abs(fft)
            magnitude_shift = np.fft.fftshift(magnitude)
            log_magnitude = np.log(1 + magnitude_shift)
            normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)
            
            # 计算原始大小（使用float64）
            original_data = normalized.copy()
            base_name = os.path.basename(path).split('.')[0]
            temp_path = os.path.join(output_dir, f"{base_name}_temp.npy")
            np.save(temp_path, original_data)
            original_size = os.path.getsize(temp_path)
            total_original_size += original_size
            os.remove(temp_path)
            
            # 降低精度
            if precision == "float16":
                normalized = normalized.astype(np.float16)
            elif precision == "float32":
                normalized = normalized.astype(np.float32)
            
            # 保存为优化格式
            base_name = os.path.basename(path).split('.')[0]
            
            if compress_data:
                # 使用压缩存储
                save_path = os.path.join(output_dir, f"{base_name}.npz")
                np.savez_compressed(save_path, data=normalized)
            else:
                save_path = os.path.join(output_dir, f"{base_name}.npy")
                np.save(save_path, normalized)
            
            # 计算优化后的大小
            optimized_size = os.path.getsize(save_path)
            total_optimized_size += optimized_size
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            skipped_files += 1
            continue
    
    # 计算总节省空间百分比
    if total_original_size > 0:
        saved_percent = (1 - total_optimized_size / total_original_size) * 100
        print(f"Dataset: {dataset_name}")
        print(f"Storage optimization: {saved_percent:.2f}% space saved")
        print(f"Original size: {total_original_size/1024/1024:.2f} MB")
        print(f"Optimized size: {total_optimized_size/1024/1024:.2f} MB")
    
    if skipped_files > 0:
        print(f"Warning: Skipped {skipped_files} files when processing {dataset_name}")
    
    return output_dir

def main():
    config = Config()
    data_dirs = [
        f"{config.data_dir}/FAS/CASIA",
        f"{config.data_dir}/FAS/idiap",
        f"{config.data_dir}/FAS/MSU",
        f"{config.data_dir}/FAS/OULU"
    ]
    
    start = time.time()
    # 划分数据集
    all_splits = split_datasets(data_dirs)
    
    # 用于保存所有数据集的统计信息
    stats = {}
    splits_info = {}
    
    # 生成频域特征根目录
    output_freq_base_dir = f"{config.data_dir}/dataset/frequency"
    
    # 为每个数据集分别处理
    for dataset_name, splits in all_splits.items():
        print(f"Processing dataset: {dataset_name}")
        stats[dataset_name] = {}
        splits_info[dataset_name] = splits
        
        # 生成频域特征
        for split_name in tqdm(['train', 'val', 'test'], 
                             desc=f"Processing {dataset_name} splits", 
                             ncols=100):
            spatial_paths = splits[split_name][0]
            freq_dir = generate_frequency_features(
                dataset_name,
                spatial_paths,
                output_freq_base_dir,
                compress_data=True,
                precision="float32"
                )
            
            # 只对训练集计算统计信息
            if split_name == 'train':
                # 计算空域统计信息
                spatial_mean, spatial_std = compute_mean_std(spatial_paths, is_spatial=True)
                
                # 频域数据路径
                freq_paths = [os.path.join(freq_dir, f"{os.path.basename(p).split('.')[0]}.npy") 
                            for p in spatial_paths]
                freq_mean, freq_std = compute_mean_std(freq_paths, is_spatial=False)
                
                # 保存统计信息
                stats[dataset_name] = {
                    'spatial': {'mean': spatial_mean, 'std': spatial_std},
                    'frequency': {'mean': freq_mean, 'std': freq_std}
                }
    
    # 保存统计信息
    os.makedirs('configs', exist_ok=True)
    with open('configs/dataset_stats.json', 'w') as f:
        json.dump(stats, f)
    
    # 保存数据划分路径
    with open('configs/splits.json', 'w') as f:
        json.dump(splits_info, f)

    print(f"All datasets processed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

if __name__ == '__main__':
    main()