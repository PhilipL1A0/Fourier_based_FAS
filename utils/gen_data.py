import os
import sys
import cv2
import json
import time
import numpy as np
import scipy as sp
from scipy import spatial
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.config import Config
from sklearn.model_selection import train_test_split
from utils.face_detection import FaceDection, LandmarksDetection


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
            # 如果是npz格式
            if path.endswith('.npz'):
                img = np.load(path)['data']
            else:
                img = np.load(path)
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

def process_image_data(dataset_name, spatial_paths, base_dir, 
                      compress_data=True, precision="float32", 
                      detect_face=True, face_align=False, target_size=(224, 224)):
    """
    同时处理空域和频域数据，添加人脸检测和裁剪功能
    
    参数:
        dataset_name: 数据集名称
        spatial_paths: 空域图像路径
        base_dir: 基础目录
        compress_data: 是否压缩频域数据
        precision: 数据精度
        detect_face: 是否进行人脸检测
        face_align: 是否进行人脸对齐（需要detect_face为True）
        target_size: 输出图像目标大小
    """
    # 创建输出目录
    freq_dataset_dir = os.path.join(base_dir, "data", "frequency", dataset_name)
    spatial_dataset_dir = os.path.join(base_dir, "data", "spatial", dataset_name)
    os.makedirs(freq_dataset_dir, exist_ok=True)
    os.makedirs(spatial_dataset_dir, exist_ok=True)
    
    # 初始化人脸检测器
    face_detector = None
    landmarks_detector = None
    if detect_face:
        face_detector = FaceDection("TF", base_dir=base_dir)  # 使用TF模型效果更好
        if face_align:
            landmarks_detector = LandmarksDetection()
    
    skipped_files = 0
    face_detection_failed = 0
    
    for path in tqdm(spatial_paths, 
                    desc=f"Processing images for {dataset_name}", 
                    ncols=100, 
                    leave=False):
        try:
            # 读取图像
            img_bgr = cv2.imread(path)
            if img_bgr is None:
                print(f"Warning: Unable to read image at {path}. Skipping file.")
                skipped_files += 1
                continue
            
            base_name = os.path.basename(path)
            file_name_no_ext = os.path.splitext(base_name)[0]
            
            # 人脸检测与裁剪
            if detect_face and face_detector is not None:
                face_img = face_detector.face_detect(img_bgr)
                if face_img is None:
                    print(f"Warning: No face detected in {path}. Using original image.")
                    face_detection_failed += 1
                    face_img = img_bgr  # 使用原始图像
                else:
                    # 调整大小使其统一
                    face_img = cv2.resize(face_img, target_size)
                    
                    # 可选：人脸关键点提取和对齐
                    if face_align and landmarks_detector is not None:
                        try:
                            landmarks = landmarks_detector.landmarks_detect(face_img, display=False)
                        except Exception as e:
                            print(f"Warning: Landmark detection failed for {path}: {e}")
            else:
                face_img = cv2.resize(img_bgr, target_size)  # 不进行人脸检测时也确保统一大小
            
            # 处理并保存空域图像
            spatial_save_path = os.path.join(spatial_dataset_dir, base_name)
            cv2.imwrite(spatial_save_path, face_img)
            
            # 生成并保存频域特征 - 基于处理后的人脸图像
            img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            fft = np.fft.fft2(img_gray)
            magnitude = np.abs(fft)
            magnitude_shift = np.fft.fftshift(magnitude)
            log_magnitude = np.log(1 + magnitude_shift)
            normalized = (log_magnitude - log_magnitude.min()) / (log_magnitude.max() - log_magnitude.min() + 1e-8)
            
            # 降低精度
            if precision == "float16":
                normalized = normalized.astype(np.float16)
            elif precision == "float32":
                normalized = normalized.astype(np.float32)
            
            # 保存频域特征
            if compress_data:
                freq_save_path = os.path.join(freq_dataset_dir, f"{file_name_no_ext}.npz")
                np.savez_compressed(freq_save_path, data=normalized)
            else:
                freq_save_path = os.path.join(freq_dataset_dir, f"{file_name_no_ext}.npy")
                np.save(freq_save_path, normalized)
            
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
            skipped_files += 1
            continue
    
    # 统计信息
    if skipped_files > 0:
        print(f"Warning: Skipped {skipped_files} files when processing {dataset_name}")
    if detect_face and face_detection_failed > 0:
        print(f"Warning: Face detection failed for {face_detection_failed} images in {dataset_name}")
    
    return freq_dataset_dir, spatial_dataset_dir

def main():
    config = Config()
    data_dirs = [
        f"{config.data_dir}/FAS/CASIA",
        f"{config.data_dir}/FAS/idiap",
        f"{config.data_dir}/FAS/MSU",
        f"{config.data_dir}/FAS/OULU"
    ]
    
    detect_face = True  # 是否进行人脸检测
    face_align = False  # 是否进行人脸对齐 (可选)
    target_size = (224, 224)  # 输出图像大小
    
    start = time.time()
    # 划分数据集
    all_splits = split_datasets(data_dirs)
    
    # 用于保存所有数据集的统计信息
    stats = {}
    splits_info = {}
    
    # 生成频域特征根目录
    output_freq_base_dir = f"{config.data_dir}/frequency"
    output_spacial_base_dir = f"{config.data_dir}/spatial"
    
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
            freq_dir, spatial_dir = process_image_data(
                dataset_name,
                spatial_paths,
                base_dir=config.base_dir,
                compress_data=config.compress_data,
                precision="float32",
                detect_face=detect_face,
                face_align=face_align,
                target_size=target_size
            )
            
            # 只对训练集计算统计信息
            if split_name == 'train':
                # 计算空域统计信息 - 此处是基于处理后的图像
                spatial_processed_paths = [os.path.join(spatial_dir, os.path.basename(p)) for p in spatial_paths]
                spatial_mean, spatial_std = compute_mean_std(spatial_processed_paths, is_spatial=True)
                
                # 频域数据路径
                file_type = "npz" if config.compress_data else "npy"
                freq_paths = [os.path.join(freq_dir, f"{os.path.basename(p).split('.')[0]}.{file_type}") 
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
        json.dump(stats, f, indent=4)
    
    # 保存数据划分路径
    with open('configs/splits.json', 'w') as f:
        json.dump(splits_info, f)

    print(f"All datasets processed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start))}")

if __name__ == '__main__':
    main()