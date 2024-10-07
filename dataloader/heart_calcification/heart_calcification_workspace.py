
import os
import numpy as np
from PIL import Image

from config import config
from dataloader.heart_calcification.heart_calcification_data_processor import HeartCalcificationDataProcessor
from dataloader.heart_calcification.heart_calcification_results_display import HeartCalcificationResultsDisplay


## 繪製資料集

# 初始化数据处理器
data_dir = "D://Paper/RGB_SFM/data/HeartCalcification/basic"  # 请替换为您的数据目录
grid_size = 45  # 请根据您的实际情况调整
resize_height =  900
data_processor = HeartCalcificationDataProcessor(grid_size, data_dir, resize_height)

# 获取数据字典
data_dict = data_processor.get_data_dict()

# 初始化结果处理器
results_processor = HeartCalcificationResultsDisplay()

# 准备图像和标签数据
images = {}
labels = {}
# 修改这部分代码
for image_name, image_data in data_dict.items():
    # 读取原始图像
    img = Image.open(image_data.image_path)
    img_array = np.array(img)

    # 确保图像是 3 通道 RGB
    if len(img_array.shape) == 2:  # 如果是灰度图
        img_array = np.stack((img_array,) * 3, axis=-1)
    elif img_array.shape[2] == 4:  # 如果是 RGBA
        img_array = img_array[:, :, :3]

    # 确保数据类型是 uint8，范围在 0-255
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = img_array.astype(np.uint8)

    # 创建标签数组
    label_array = np.zeros(image_data.split_count, dtype=int)
    for (i, j), label in image_data.labels.items():
        label_array[i, j] = label
    
    images[image_name] = img_array
    labels[image_name] = label_array


# 可视化数据集
save_dir = 'D://Paper/RGB_SFM/data/HeartCalcification/visual_data_45'  # 请替换为您想保存可视化结果的目录
results_processor.visualize_dataset(images, labels, save_dir)