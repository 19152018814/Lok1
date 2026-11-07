import os
import shutil
from collections import defaultdict
from sklearn.model_selection import train_test_split


task = 'exp_cu_7/'

# 输入路径和输出路径
input_dir = task + 'dataset_10_cnn_color_augmented'
train_output_dir = task + 'train_dataset_10_cnn_color_augmented'
test_output_dir = task + 'test_dataset_10_cnn_color_augmented'

# 创建输出目录
os.makedirs(train_output_dir, exist_ok=True)
os.makedirs(test_output_dir, exist_ok=True)

# 收集每个类别的图片路径
category_images = defaultdict(list)
for filename in os.listdir(input_dir):
    if "_" in filename:
        label = filename.split('_')[0]  # 提取类别 (label)
        category_images[label].append(os.path.join(input_dir, filename))

# 按类别划分训练集和测试集
for label, images in category_images.items():
    train_images, test_images = train_test_split(images, test_size=0.2, random_state=42)

    # 创建类别子目录
    train_label_dir = os.path.join(train_output_dir, label)
    test_label_dir = os.path.join(test_output_dir, label)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(test_label_dir, exist_ok=True)

    # 复制图片到训练集目录
    for img_path in train_images:
        shutil.copy(img_path, train_label_dir)

    # 复制图片到测试集目录
    for img_path in test_images:
        shutil.copy(img_path, test_label_dir)

print("Dataset split completed.")
