
import os
import numpy as np
from PIL import Image
from dataloader.heart_calcification.heart_calcification_data_processor import HeartCalcificationDataProcessor
from dataloader.heart_calcification.heart_calcification_results_processor import HeartCalcificationResultsProcessor


## 繪製資料集

# 初始化数据处理器
data_dir = "D://Paper/RGB_SFM/data/HeartCalcification/basic"  # 请替换为您的数据目录
grid_size = 45  # 请根据您的实际情况调整
data_processor = HeartCalcificationDataProcessor(grid_size, data_dir)

# 生成数据集
data_processor.generate_dataset()

# 获取数据字典
data_dict = data_processor.get_data_dict()

# 初始化结果处理器
results_processor = HeartCalcificationResultsProcessor()

# 准备图像和标签数据
images = {}
labels = {}
for image_name, image_data in data_dict.items():
    # 读取原始图像
    img = Image.open(image_data.image_path)
    img_array = np.array(img)
    
    # 创建标签数组
    label_array = np.zeros(image_data.split_count, dtype=int)
    for (i, j), label in image_data.labels.items():
        label_array[i, j] = label
    
    images[image_name] = img_array
    labels[image_name] = label_array

# 可视化数据集
save_dir = 'D://Paper/RGB_SFM/data/HeartCalcification/visual_data'  # 请替换为您想保存可视化结果的目录
results_processor.visualize_dataset(images, labels, save_dir)