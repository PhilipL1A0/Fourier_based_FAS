import os
import cv2
import sys
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import Config
from utils.face_detection import FaceDetection

def validate_image(img_path, min_size=64, blur_thresh=10):
    img = cv2.imread(img_path)
    if img is None or img.size == 0:
        return False, None
    h, w = img.shape[:2]
    if h < min_size or w < min_size:
        return False, None
    lap_var = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if lap_var < blur_thresh:
        return False, None
    return True, img

def process_image(img, face_detector=None, detect_face=True, target_size=(224, 224)):
    if detect_face and face_detector is not None:
        face_img = face_detector.face_detect(img)
        if face_img is None:
            return None, None
    else:
        face_img = img
    face_img = cv2.resize(face_img, target_size)
    img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(img_gray)
    magnitude = np.abs(fft)
    magnitude_shift = np.fft.fftshift(magnitude)
    freq_data = np.log1p(magnitude_shift)
    freq_data = (freq_data - freq_data.min()) / (freq_data.max() - freq_data.min() + 1e-8)
    return face_img, freq_data

def save_data(spatial_dir, freq_dir, base_name, img, freq_data, compress=True):
    os.makedirs(spatial_dir, exist_ok=True)
    os.makedirs(freq_dir, exist_ok=True)
    spatial_path = os.path.join(spatial_dir, f"{base_name}.jpg")
    freq_path = os.path.join(freq_dir, f"{base_name}.npz" if compress else f"{base_name}.npy")
    cv2.imwrite(spatial_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if compress:
        np.savez_compressed(freq_path, data=freq_data.astype(np.float32))
    else:
        np.save(freq_path, freq_data.astype(np.float32))
    return spatial_path, freq_path

def compute_mean_std(image_paths, is_spatial=True):
    total = 0
    sum_ = None
    sum_sq = None
    for path in tqdm(image_paths, desc="Mean/Std"):
        try:
            if is_spatial:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                if sum_ is None:
                    sum_ = np.zeros(3)
                    sum_sq = np.zeros(3)
                sum_ += img.mean(axis=(0, 1))
                sum_sq += (img ** 2).mean(axis=(0, 1))
            else:
                arr = np.load(path)
                freq = arr['data'] if isinstance(arr, np.lib.npyio.NpzFile) else arr
                if sum_ is None:
                    sum_ = 0.0
                    sum_sq = 0.0
                sum_ += freq.mean()
                sum_sq += (freq ** 2).mean()
            total += 1
        except Exception as e:
            print(f"Error: {e}")
    mean = sum_ / total
    std = np.sqrt(np.maximum(sum_sq / total - mean ** 2, 0))
    return mean.tolist(), std.tolist()

def collect_images(data_dir):
    images, labels = [], []
    for label_name, label in [('living', 1), ('spoofing', 0)]:
        cat_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(cat_dir):
            continue
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith(('.jpg', '.png', '.bmp')):
                images.append(os.path.join(cat_dir, fname))
                labels.append(label)
    return np.array(images), np.array(labels)

def main():
    config = Config()
    datasets = {
        'CASIA': f"{config.data_dir}/FAS/CASIA",
        'IDIAP': f"{config.data_dir}/FAS/idiap",
        'MSU': f"{config.data_dir}/FAS/MSU",
        'OULU': f"{config.data_dir}/FAS/OULU"
    }
    stats = {}
    splits_info = {}
    for dataset_name, data_path in datasets.items():
        print(f"\nProcessing {dataset_name} ...")
        X, y = collect_images(data_path)
        if len(X) == 0:
            print(f"No valid images in {dataset_name}")
            continue
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.8, stratify=y, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
        splits = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}
        face_detector = FaceDetection("TF", base_dir=config.base_dir) if config.detect_face else None
        split_paths = {'spatial': {}, 'frequency': {}}
        dataset_splits_info = {'spatial': {}, 'frequency': {}}
        
        for split, (split_X, split_y) in splits.items():
            spatial_dir = os.path.join(config.data_dir, 'dataset', 'spatial', dataset_name, split)
            freq_dir = os.path.join(config.data_dir, 'dataset', 'frequency', dataset_name, split)
            spatial_paths, freq_paths = [], []
            valid_indices = []  # 记录有效样本的索引
            
            for i, img_path in enumerate(tqdm(split_X, desc=f"{dataset_name}-{split}")):
                valid, img = validate_image(img_path)
                if not valid:
                    continue
                face_img, freq_data = process_image(img, face_detector, config.detect_face)
                if face_img is None:
                    continue
                    
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                s_path, f_path = save_data(spatial_dir, freq_dir, base_name, face_img, freq_data, compress=config.compress_data)
                
                spatial_paths.append(s_path)
                freq_paths.append(f_path)
                valid_indices.append(i)  # 记录成功处理的样本索引
            
            # 只保留成功处理的样本对应的标签
            valid_labels = split_y[valid_indices].tolist()
            
            split_paths['spatial'][split] = spatial_paths
            split_paths['frequency'][split] = freq_paths
            
            # 为当前split更新信息
            dataset_splits_info['spatial'][split] = [
                [os.path.basename(p) for p in spatial_paths],
                valid_labels
            ]
            dataset_splits_info['frequency'][split] = [
                [os.path.basename(p) for p in freq_paths],
                valid_labels
            ]
            
            # 打印统计信息进行验证
            print(f"\n{dataset_name} {split} 统计:")
            print(f"处理前样本数: {len(split_X)}")
            print(f"处理后样本数: {len(spatial_paths)}")
            print(f"标签分布: {Counter(valid_labels)}")
        
        # 所有split处理完后，再更新到全局splits_info
        splits_info[dataset_name] = dataset_splits_info
        
        # 只统计训练集
        if 'train' in splits:
            stats[dataset_name] = {
                'spatial': {},
                'frequency': {}
            }
            mean, std = compute_mean_std(split_paths['spatial']['train'], is_spatial=True)
            stats[dataset_name]['spatial']['mean'] = mean
            stats[dataset_name]['spatial']['std'] = std
            mean, std = compute_mean_std(split_paths['frequency']['train'], is_spatial=False)
            stats[dataset_name]['frequency']['mean'] = mean
            stats[dataset_name]['frequency']['std'] = std
    os.makedirs('configs', exist_ok=True)
    with open('configs/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    with open('configs/splits.json', 'w') as f:
        json.dump(splits_info, f, indent=4)

if __name__ == '__main__':
    main()