import os
import cv2
import json
import numpy as np
from sklearn.model_selection import train_test_split

def split_datasets(data_dirs, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    all_images = []
    all_labels = []
    for data_dir in data_dirs:
        for category in ['living', 'spoofing']:
            label = 1 if category == 'living' else 0  # 标签逻辑调整（living=1，spoofing=0）
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

def compute_mean_std(image_paths, is_spatial=True):
    total_pixels = 0
    sum_channels = 0
    sum_squares = 0
    
    for path in image_paths:
        if is_spatial:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, c = img.shape
            sum_channels += np.sum(img, axis=(0, 1))
            sum_squares += np.sum(img**2, axis=(0, 1))
            total_pixels += h * w
        else:
            # 频域图像假设为单通道灰度图
            img = np.load(path)  # 假设频域特征以.npy格式保存
            sum_channels += np.sum(img)
            sum_squares += np.sum(img**2)
            total_pixels += img.size
    
    if is_spatial:
        mean = sum_channels / total_pixels
        std = np.sqrt(sum_squares / total_pixels - (mean ** 2))
    else:
        mean = sum_channels / total_pixels
        std = np.sqrt(sum_squares / total_pixels - (mean ** 2))
    
    return mean, std

def generate_frequency_features(spatial_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for path in spatial_paths:
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
    data_dirs = [
        '/media/main/lzf/FAS/Fourier_based_cnn/data/FAS/CASIA',
        '/media/main/lzf/FAS/Fourier_based_cnn/data/FAS/idiap',
        '/media/main/lzf/FAS/Fourier_based_cnn/data/FAS/MSU',
        '/media/main/lzf/FAS/Fourier_based_cnn/data/FAS/OULU'
    ]
    
    # 划分数据集
    splits = split_datasets(data_dirs)
    
    # 生成频域特征并保存
    output_freq_dir = "preprocessed/frequency"
    for split in ['train', 'val', 'test']:
        spatial_paths = splits[split][0]
        generate_frequency_features(spatial_paths, output_freq_dir)
    
    # 计算空域和频域的均值和标准差
    train_paths = splits['train'][0]
    spatial_mean, spatial_std = compute_mean_std(train_paths, is_spatial=True)
    freq_mean, freq_std = compute_mean_std([os.path.join(output_freq_dir, f"{os.path.basename(p).split('.')[0]}.npy") for p in train_paths], is_spatial=False)
    
    # 保存统计信息
    stats = {
        'spatial': {'mean': spatial_mean.tolist(), 'std': spatial_std.tolist()},
        'frequency': {'mean': freq_mean.item(), 'std': freq_std.item()}
    }
    with open('configs/dataset_stats.json', 'w') as f:
        json.dump(stats, f)
    
    # 保存数据划分路径
    with open('configs/splits.json', 'w') as f:
        json.dump(splits, f)

if __name__ == '__main__':
    main()