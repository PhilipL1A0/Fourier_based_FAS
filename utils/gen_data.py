import os
import cv2
import json
import time
import numpy as np
from tqdm import tqdm
from configs import Config
from sklearn.model_selection import train_test_split


def split_datasets(data_dirs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    all_images = []
    all_labels = []
    for data_dir in data_dirs:
        for category in ['living', 'spoofing']:
            label = 1 if category == 'living' else 0
            img_dir = os.path.join(data_dir, category)
            for img_name in os.listdir(img_dir):
                if img_name.lower().endswith(('.jpg', '.png', '.bmp')):
                    img_path = os.path.join(img_dir, img_name)
                    all_images.append(img_path)
                    all_labels.append(label)
    
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
    
    return {
        'train': (X_train.tolist(), y_train.tolist()),
        'val': (X_val.tolist(), y_val.tolist()),
        'test': (X_test.tolist(), y_test.tolist())
    }

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

def generate_frequency_features(spatial_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 添加进度条
    for path in tqdm(spatial_paths, 
                    desc="Generating Frequency Features", 
                    ncols=100, 
                    leave=False):
        # 生成频域特征并保存
        img_bgr = cv2.imread(path)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(img_gray)
        magnitude = np.abs(fft)
        magnitude_shift = np.fft.fftshift(magnitude)
        log_magnitude = np.log(1 + magnitude_shift)
        normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)
        
        # 保存为.npy文件
        base_name = os.path.basename(path).split('.')[0]
        save_path = os.path.join(output_dir, f"{base_name}.npy")
        np.save(save_path, normalized)

def main():
    config = Config()
    data_dirs = [
        f"{config.data_dir}/FAS/CASIA",
        f"{config.data_dir}/FAS/idiap",
        f"{config.data_dir}/data/FAS/MSU",
        f"{config.data_dir}/data/FAS/OULU"
    ]
    
    start = time.time()
    # 划分数据集
    splits = split_datasets(data_dirs)
    
    # 生成频域特征并保存
    output_freq_dir = f"{config.data_dir}/dataset/frequency"
    
    # 添加进度条显示每个split的处理
    for split in tqdm(['train', 'val', 'test'], 
                     desc="Processing Splits", 
                     ncols=100):
        spatial_paths = splits[split][0]
        generate_frequency_features(spatial_paths, output_freq_dir)
    
    # 计算空域和频域的均值和标准差
    train_paths = splits['train'][0]
    spatial_mean, spatial_std = compute_mean_std(train_paths, is_spatial=True)
    freq_mean, freq_std = compute_mean_std([os.path.join(output_freq_dir, f"{os.path.basename(p).split('.')[0]}.npy") for p in train_paths], is_spatial=False)
    
    # 保存统计信息
    stats = {
        'spatial': {'mean': spatial_mean, 'std': spatial_std},
        'frequency': {'mean': freq_mean, 'std': freq_std}
    }
    with open('configs/dataset_stats.json', 'w') as f:
        json.dump(stats, f)
    
    # 保存数据划分路径
    with open('configs/splits.json', 'w') as f:
        json.dump(splits, f)

    print(f"Data generation completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

if __name__ == '__main__':
    main()