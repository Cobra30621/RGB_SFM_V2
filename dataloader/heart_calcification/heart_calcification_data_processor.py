import os
from typing import List, Dict, Tuple
from PIL import Image
import numpy as np

from .image_split_data import ImageSplitData
from shapely.geometry import Polygon


class HeartCalcificationDataProcessor:
    """
    心脏钙化数据处理器类。
    用于处理心脏钙化相关的图像数据、标签和掩码。
    """

    def __init__(self, grid_size: int, data_dir: str,
                 need_resize_height: bool, resize_height: int, threshold:float):
        """
        初始化心脏钙化数据处理器。

        参数:
        grid_size (int): 网格大小
        data_dir (str): 数据目录路径
        resize_height (int): 图像缩放后的高度
        """
        self.grid_size = grid_size
        self.data_dir = data_dir
        self.need_resize_height = need_resize_height
        self.resize_height = resize_height
        self.data_dict: Dict[str, ImageSplitData] = {}
        self.image_files, self.yolo_files, self.mask_image_files = self.get_file_paths(data_dir)

        # 針對鈣化點框框，判斷是否為鈣化點，將框框縮小
        self.threshold = threshold

        self._generate_dataset(threshold=self.threshold)

    def get_file_paths(self, data_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """
        获取数据目录中的文件路径。

        参数:
        data_dir (str): 数据目录路径

        返回:
        Tuple[List[str], List[str], List[str]]: 图像文件、YOLO文件和掩码文件的路径列表
        """
        mask_files = [f for f in os.listdir(data_dir) if f.endswith('_vessel.txt')]
        image_files = [f.replace('_vessel.txt', '.png') for f in mask_files]
        yolo_files = [f.replace('_vessel.txt', '.txt') for f in mask_files]
        return image_files, yolo_files, mask_files

    def _generate_dataset(self, threshold: float = 1.0):
        """
        生成数据集。
        处理图像、标签和掩码文件，创建ImageSplitData对象并存储在data_dict中。
        """
        for img_path, yolo_path, mask_file in zip(self.image_files, self.yolo_files, self.mask_image_files):
            img = Image.open(os.path.join(self.data_dir, img_path))
            
            # 调用 resize_image 方法
            img = self.resize_image(img)
            
            width, height = img.size
            num_blocks_h = height // self.grid_size
            num_blocks_w = width // self.grid_size

            label = np.full((num_blocks_h, num_blocks_w), -1, dtype=np.int8)
            label = self.filter_vessel_mask(label, mask_file)
            label = self.filter_heart_calcification_points(label, yolo_path, threshold)

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

    def resize_image(self, img: Image.Image) -> Image.Image:
        if not self.need_resize_height:
            return img

        """根据 self.resize_height 缩放图像"""
        original_width, original_height = img.size
        scale_factor = self.resize_height / original_height
        new_width = int(original_width * scale_factor)
        
        # 缩放图像
        return img.resize((new_width, self.resize_height), Image.LANCZOS)

    def split_image(self, img: Image.Image) -> List[Image.Image]:
        """
        将图像分割成网格。

        参数:
        img (Image.Image): 输入图像

        返回:
        List[Image.Image]: 分割后的图像列表
        """
        width, height = img.size
        split_images = []
        for i in range(0, height, self.grid_size):
            for j in range(0, width, self.grid_size):
                box = (j, i, j + self.grid_size, i + self.grid_size)
                split_images.append(img.crop(box))
        return split_images

    def filter_vessel_mask(self, label: np.ndarray, mask_file: str) -> np.ndarray:
        """
        根据血管掩码过滤标签。

        参数:
        label (np.ndarray): 标签数组
        mask_file (str): 掩码文件路径
        scale_factor (float): 缩放因子

        返回:
        np.ndarray: 过滤后的标签数组
        """
        img_height, img_width = label.shape[0] * self.grid_size, label.shape[1] * self.grid_size
        
        with open(os.path.join(self.data_dir, mask_file), 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, *polygon = line.strip().split()
            if class_id == '0':
                polygon = [(float(polygon[i]) * img_width, float(polygon[i+1]) * img_height ) for i in range(0, len(polygon), 2)]
                
                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        grid_box = [
                            (j * self.grid_size, i * self.grid_size),
                            ((j+1) * self.grid_size, i * self.grid_size),
                            ((j+1) * self.grid_size, (i+1) * self.grid_size),
                            (j * self.grid_size, (i+1) * self.grid_size)
                        ]
                        if self.polygon_intersects_grid(polygon, grid_box):
                            label[i, j] = 0
        
        return label

    def polygon_intersects_grid(self, polygon, grid_box):
        """
        检查多边形是否与网格相交。

        参数:
        polygon: 多边形坐标列表
        grid_box: 网格坐标列表

        返回:
        bool: 是否相交
        """
        poly = Polygon(polygon)
        grid = Polygon(grid_box)
        return poly.intersects(grid)

    def filter_heart_calcification_points(self, label: np.ndarray, yolo_file: str, threshold: float) -> np.ndarray:
        """
        根据心脏钙化点过滤标签。

        参数:
        label (np.ndarray): 标签数组
        yolo_file (str): YOLO格式的标注文件路径
        threshold (float): 阈值
        scale_factor (float): 缩放因子

        返回:
        np.ndarray: 过滤后的标签数组
        """
        with open(os.path.join(self.data_dir, yolo_file), 'r') as f:
            lines = f.readlines()
        
        img_height, img_width = label.shape[0] * self.grid_size, label.shape[1] * self.grid_size
        
        for line in lines:
            _, x_center, y_center, w, h = map(float, line.split())
            w *= threshold
            h *= threshold
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

    def get_model_ready_data(self) -> List[Tuple[Tuple[str, int, int], Image.Image, int]]:
        """
        获取模型就绪的数据。

        返回:
        List[Tuple[Tuple[str, int, int], Image.Image, int]]: 包含图像名称、位置、分割图像和标签的列表
        """
        model_ready_data = []
        for image_name, image_data in self.data_dict.items():
            img = Image.open(image_data.image_path)
            
            # 调用 resize_image 方法
            img = self.resize_image(img)
            
            split_images = self.split_image(img)
            for (i, j), label in image_data.labels.items():
                if label != -1:
                    index = i * image_data.split_count[1] + j
                    model_ready_data.append(((image_name, i, j), split_images[index], label))
        return model_ready_data


    def display_label_counts(self):
        label_counts = {}
        for image_name, image_data in self.data_dict.items():
            for label in image_data.labels.values():
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1
        
        for label, count in label_counts.items():
            print(f"Label: {label}, Count: {count}")

    def get_data_dict(self) -> Dict[str, ImageSplitData]:
        """
        获取数据字典。

        返回:
        Dict[str, ImageSplitData]: 包含图像名称和对应ImageSplitData对象的字典
        """
        return self.data_dict