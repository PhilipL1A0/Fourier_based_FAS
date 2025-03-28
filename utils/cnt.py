import os
from pathlib import Path

def count_files(directory):
    parent_dir = Path(directory)
    if not parent_dir.is_dir():
        print("错误：路径不是文件夹！")
        return

    # 遍历直接子文件夹
    for child_dir in parent_dir.iterdir():
        if child_dir.is_dir():
            file_count = 0
            # 递归统计所有文件（包括嵌套目录）
            for root, dirs, files in os.walk(child_dir):
                file_count += len(files)
            print(f"{child_dir.name}: {file_count} 个文件")

if __name__ == "__main__":
    target_dir = "/media/main/lzf/FBFAS/data/FAS/MSU"
    count_files(target_dir)