import os
from typing import List, Dict, Tuple
from PIL import Image, ImageEnhance
import numpy as np

from .image_enhance import ENHANCE_FUNCTIONS
from .image_split_data import ImageSplitData
from shapely.geometry import Polygon

from .mask_processor import create_polygon, mask_image_with_polygon


from .image_tool import resize_image

class HeartCalcificationDataProcessor:
    """
    心脏钙化数据处理器类。
    用于处理心脏钙化相关的图像数据、标签和掩码。
    """

    def __init__(self, grid_size: int, data_dir: str,
                 need_resize_height: bool, resize_height: int, threshold:float,
                 contrast_factor: float = 1.0, enhance_method: str = 'none',
                 use_vessel_mask: bool = False):
        """
        初始化心脏钙化数据处理器。

        参数:
        grid_size (int): 网格大小
        data_dir (str): 数据目录路径
        resize_height (int): 图像缩放后的高度
        contrast_factor (float): 对比度因子，默认为 1.0（无变化）
        enhance_method (str): 增强方法的名称
        use_vessel_mask (bool): 是否使用血管遮罩，默认为 False
        """
        self.grid_size = grid_size
        self.data_dir = data_dir
        self.need_resize_height = need_resize_height
        self.resize_height = resize_height
        self.data_dict: Dict[str, ImageSplitData] = {}
        self.image_files, self.calcification_mask_files, self.vessel_mask_files = self.get_file_paths(data_dir)

        # 針對鈣化點框框，判斷是否為鈣化點，將框框縮小
        self.threshold = threshold
        self.contrast_factor = contrast_factor  # 存储对比度因子
        self.enhance_method = enhance_method  # 存储增强方法名称
        self.use_vessel_mask = use_vessel_mask  # 存储 use_vessel_mask 参数

        self._generate_dataset(threshold=self.threshold)

    def get_file_paths(self, data_dir: str) -> Tuple[List[str], List[str], List[str]]:
        """
        获取数据目录中的文件路径。

        参数:
        data_dir (str): 数据目录路径

        返回:
        Tuple[List[str], List[str], List[str]]: 图像文件、YOLO文件和掩码文件的路径列表
        """
        calcification_mask_files = [f for f in os.listdir(data_dir) if f.endswith('_calcification.txt')]
        print(len(calcification_mask_files))
        vessel_mask_files = [f.replace('_calcification.txt', '_vessel.txt') for f in calcification_mask_files]
        image_files = [f.replace('_calcification.txt', '.png') for f in calcification_mask_files]
        return image_files, calcification_mask_files, vessel_mask_files

    def _generate_dataset(self, threshold: float = 1.0):
        """
        生成數據集。
        處理圖像、標籤和掩碼文件，創建 ImageSplitData 對象並存儲在 data_dict 中。
        """
        for img_path, calcification_mask_path, vessel_mask_file in zip(self.image_files, self.calcification_mask_files, self.vessel_mask_files):
            img = Image.open(os.path.join(self.data_dir, img_path)).convert('L')  # 将图像转换为灰阶
            img = np.array(img)
            # 调用 resize_image 方法
            if self.need_resize_height:
                img = resize_image(img, self.resize_height)

            width, height = img.shape[1], img.shape[0]  # 注意：np.ndarray 的 shape 是 (高度, 宽度, 通道数)
            num_blocks_h = height // self.grid_size
            num_blocks_w = width // self.grid_size

            label = np.full((num_blocks_h, num_blocks_w), -1, dtype=np.int8)
            label = self.filter_with_mask(label, vessel_mask_file, height, width, 0    , 1)  # 根據血管給 0
            label = self.filter_with_mask(label, calcification_mask_path, height, width,1, 0.5)  # 根據鈣化點給 1

            # 根据增强方法对图像进行增强
            enhance_func = ENHANCE_FUNCTIONS.get(self.enhance_method)
            if enhance_func:
                img = enhance_func(img)
            else:
                print(f"未知的增强方法: {self.enhance_method}")

            # 將圖片用血管做遮罩
            if self.use_vessel_mask:
                vessel_mask_path = os.path.join(self.data_dir, vessel_mask_file)
                img = mask_image_with_polygon(img, vessel_mask_path)

            # 切割圖片
            split_images = self.split_image(img)
            
            # 创建符合要求的 labels 字典
            labels = {
                (i, j): int(label[i, j]) for i in range(num_blocks_h) for j in range(num_blocks_w)
            }
            
            image_split_data = ImageSplitData(
                image_name=img_path,
                image_path=os.path.join(self.data_dir, img_path),
                split_count=(num_blocks_h, num_blocks_w),
                labels=labels,
                split_images=split_images,  # 将切割后的图像存储到 img 属性中
                vessel_mask_file=vessel_mask_file,  # 新增 vessel_mask_file
                calcification_path=calcification_mask_path  # 新增 calcification_path
            )
            self.data_dict[img_path] = image_split_data

    def enhance_image(self, img: np.ndarray) -> np.ndarray:
        """
        使用指定的增强方法增强图像。

        参数:
        img (np.ndarray): 输入图像

        返回:
        np.ndarray: 增强后的图像
        """
        enhance_func = ENHANCE_FUNCTIONS.get(self.enhance_method)  # 获取增强函数
        enhanced_np_img = enhance_func(img)  # 使用增强函数
        return enhanced_np_img  # 返回增强后的 NumPy 数组

    def enhance_all_split_images(self):
        """
        对所有的分割图像执行增强。
        """
        for image_data in self.data_dict.values():
            for i in range(len(image_data.split_images)):
                image_data.split_images[i] = self.enhance_image(image_data.split_images[i])


    def split_image(self, img: np.ndarray) -> List[np.ndarray]:
        """
        将图像分割成网格。

        参数:
        img (np.ndarray): 输入图像

        返回:
         List[np.ndarray]: 分割后的图像数组
        """
        height, width = img.shape[:2]
        split_images = []
        num_blocks_h = height // self.grid_size
        num_blocks_w = width // self.grid_size

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                box = (j * self.grid_size, i * self.grid_size, (j + 1) * self.grid_size, (i + 1) * self.grid_size)
                split_images.append(img[i * self.grid_size:(i + 1) * self.grid_size, j * self.grid_size:(j + 1) * self.grid_size])
        return split_images

    def filter_with_mask(self, label: np.ndarray, mask_file: str, img_height: int = None, img_width: int = None
                         , value: int = 0, scale: float = 1.0) -> np.ndarray:
        """
        根据遮罩过滤标签。

        参数:
        label (np.ndarray): 标签数组
        mask_file (str): 掩码文件路径
        img_height (int): 图像高度
        img_width (int): 图像宽度
        value (int): 要赋予标签的值，默认为 0
        scale (float): 多边形缩放比，默认为 1.0（无缩放）


        返回:
        np.ndarray: 过滤后的标签数组
        """
        with open(os.path.join(self.data_dir, mask_file), 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            class_id, *polygon = line.strip().split()
            if class_id == '0':
                # 将字符串转换为浮点数
                polygon = create_polygon(polygon, img_width, img_height, scale)

                for i in range(label.shape[0]):
                    for j in range(label.shape[1]):
                        grid_box = [
                            (j * self.grid_size, i * self.grid_size),
                            ((j+1) * self.grid_size, i * self.grid_size),
                            ((j+1) * self.grid_size, (i+1) * self.grid_size),
                            (j * self.grid_size, (i+1) * self.grid_size)
                        ]

                        if self.polygon_intersects_grid(polygon, grid_box):
                            label[i, j] = value  # 使用指定的值更新标签

        
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
        根据 Yolo 方框过滤标签。

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

    def get_model_ready_data(self, use_min_count: bool = False) -> List[Tuple[Tuple[str, int, int], np.ndarray, int]]:
        """
        获取模型就绪的数据。

        参数:
        use_min_count (bool): 是否使用最小数量进行筛选，默认为 True

        返回:
        List[Tuple[Tuple[str, int, int], Image.Image, int]]: 包含图像名称、位置、分割图像和标签的列表
        """
        model_ready_data = []
        label_counts = {}  # 记录每个标签的计数

        # 统计每个标签的数量
        for image_name, image_data in self.data_dict.items():
            for label in image_data.labels.values():
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += 1

        # 计算每个标签的最小数量
        min_count = min(label_counts.values()) if use_min_count else float('inf')

        # 创建一个字典来存储每个标签的随机选择的索引
        selected_indices = {label: [] for label in label_counts.keys()}

        for image_name, image_data in self.data_dict.items():

            split_images = image_data.split_images

            for (i, j), label in image_data.labels.items():
                if label != -1:
                    index = i * image_data.split_count[1] + j

                    # 随机选择标签
                    if len(selected_indices[label]) < min_count:
                        if index < len(split_images):  # 检查 index 是否在范围内
                            selected_indices[label].append((image_name, index, label))  # 存储 image_name 和 index

        # 将随机选择的图像添加到 model_ready_data
        for label, indices in selected_indices.items():
            for image_name, index, label in indices:
                if index < len(self.data_dict[image_name].split_images):  # 确保使用当前 image_data 的 split_images
                    model_ready_data.append(((image_name, i, j), self.data_dict[image_name].split_images[index], label))

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







