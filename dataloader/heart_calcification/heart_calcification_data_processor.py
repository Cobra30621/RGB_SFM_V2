import os
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np
from image_split_data import ImageSplitData

class HeartCalcificationDataProcessor:
    def __init__(self, grid_size: int, data_dir: str):
        self.grid_size = grid_size
        self.data_dir = data_dir
        self.data_dict: Dict[str, ImageSplitData] = {}
        self.image_files, self.yolo_files, self.mask_image_files = self.get_file_paths(data_dir)

    def get_file_paths(self, data_dir: str) -> Tuple[List[str], List[str], List[str]]:
        image_files = [f for f in os.listdir(data_dir) if f.endswith('.png')]
        yolo_files = [f.replace('.png', '.txt') for f in image_files]
        mask_image_files = [os.path.join(data_dir, os.path.splitext(f)[0], '14_2-down_refind.png') for f in image_files]
        return image_files, yolo_files, mask_image_files

    def generate_dataset(self):
        for img_path, yolo_path, mask_file in zip(self.image_files, self.yolo_files, self.mask_image_files):
            img = Image.open(os.path.join(self.data_dir, img_path))
            width, height = img.size
            num_blocks_h = height // self.grid_size
            num_blocks_w = width // self.grid_size

            label = np.zeros((num_blocks_h, num_blocks_w), dtype=np.int8)
            label = self.filter_vessel_mask(label, mask_file)
            label = self.filter_heart_calcification_points(label, yolo_path)

            split_images = self.split_image(img)
            
            # 创建符合要求的 labels 字典
            labels = {
                (i, j): int(label[i, j]) for i in range(num_blocks_h) for j in range(num_blocks_w)
            }
            
            image_split_data = ImageSplitData(
                image_name=img_path,
                image_path=os.path.join(self.data_dir, img_path),
                split_count=(num_blocks_h, num_blocks_w),
                labels=labels
            )
            self.data_dict[img_path] = image_split_data

    def split_image(self, img: Image.Image) -> List[Image.Image]:
        width, height = img.size
        split_images = []
        for i in range(0, height, self.grid_size):
            for j in range(0, width, self.grid_size):
                box = (j, i, j + self.grid_size, i + self.grid_size)
                split_images.append(img.crop(box))
        return split_images

    def filter_vessel_mask(self, label: np.ndarray, mask_file: str) -> np.ndarray:
        mask = Image.open(mask_file).convert('L')
        mask_array = np.array(mask)
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                y_start, y_end = i * self.grid_size, (i + 1) * self.grid_size
                x_start, x_end = j * self.grid_size, (j + 1) * self.grid_size
                if np.any(mask_array[y_start:y_end, x_start:x_end] > 0):
                    label[i, j] = 2
        return label

    def filter_heart_calcification_points(self, label: np.ndarray, yolo_file: str) -> np.ndarray:
        with open(os.path.join(self.data_dir, yolo_file), 'r') as f:
            lines = f.readlines()
        
        img_height, img_width = label.shape[0] * self.grid_size, label.shape[1] * self.grid_size
        
        for line in lines:
            _, x_center, y_center, w, h = map(float, line.split())
            x_min = int((x_center - w/2) * img_width)
            y_min = int((y_center - h/2) * img_height)
            x_max = int((x_center + w/2) * img_width)
            y_max = int((y_center + h/2) * img_height)
            
            i_min, j_min = y_min // self.grid_size, x_min // self.grid_size
            i_max, j_max = y_max // self.grid_size, x_max // self.grid_size
            
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    if 0 <= i < label.shape[0] and 0 <= j < label.shape[1]:
                        label[i, j] = 1
        
        return label

    def get_model_ready_data(self) -> List[Tuple[str, Image.Image, int]]:
        model_ready_data = []
        for image_name, image_data in self.data_dict.items():
            img = Image.open(image_data.image_path)
            split_images = self.split_image(img)
            for (i, j), label in image_data.labels.items():
                index = i * image_data.split_count[1] + j
                model_ready_data.append((image_name, split_images[index], label))
        return model_ready_data

    def get_data_dict(self) -> Dict[str, ImageSplitData]:
        return self.data_dict
