from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
import numpy as np

class HeartCalcificationResultsProcessor:
    def __init__(self):
        self.field: Any = None  # 用于存储结果处理器特定的数据类型

    def compile_training_results(self, list_of_image_names: List[str], data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        编译训练结果。

        参数:
        list_of_image_names: 图像名称列表
        data_dict: 包含训练数据的字典

        返回:
        Dict[str, Any]: 图像名称及其相关数据的字典
        """
        compiled_results = {}
        for image_name in list_of_image_names:
            if image_name in data_dict:
                image_data = data_dict[image_name]
                compiled_results[image_name] = {
                    'split_count': image_data.split_count,
                    'labels': image_data.labels,
                    # 可以根据需要添加更多的训练相关数据
                }
            else:
                print(f"Warning: Image {image_name} not found in data_dict")
        
        return compiled_results

    def compile_prediction_results(self, list_of_image_names: List[str], prediction_results: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
        """
        编译预测结果。

        参数:
        list_of_image_names: 图像名称列表
        prediction_results: 预测结果列表,每个元素为 (key, true_label, predicted_label)

        返回:
        List[Dict[str, Any]]: 包含图像名称和预测结果的字典列表
        """
        compiled_results = []
        prediction_dict = {key: (true, pred) for key, true, pred in prediction_results}
        
        for image_name in list_of_image_names:
            image_predictions = {}
            for key, (true_label, pred_label) in prediction_dict.items():
                if key.startswith(image_name):
                    # 假设 key 的格式为 "image_name_row_col"
                    _, row, col = key.rsplit('_', 2)
                    image_predictions[(int(row), int(col))] = {
                        'true_label': true_label,
                        'predicted_label': pred_label
                    }
            
            compiled_results.append({
                'image_name': image_name,
                'predictions': image_predictions
            })
        
        return compiled_results

    def calculate_accuracy(self, compiled_results: List[Dict[str, Any]]) -> float:
        """
        计算整体预测准确率。

        参数:
        compiled_results: compile_prediction_results 方法的输出

        返回:
        float: 预测准确率
        """
        total_predictions = 0
        correct_predictions = 0
        
        for result in compiled_results:
            for prediction in result['predictions'].values():
                total_predictions += 1
                if prediction['true_label'] == prediction['predicted_label']:
                    correct_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0

    def get_misclassified_images(self, compiled_results: List[Dict[str, Any]]) -> List[str]:
        """
        获取被错误分类的图像列表。

        参数:
        compiled_results: compile_prediction_results 方法的输出

        返回:
        List[str]: 被错误分类的图像名称列表
        """
        misclassified_images = []
        
        for result in compiled_results:
            if any(pred['true_label'] != pred['predicted_label'] for pred in result['predictions'].values()):
                misclassified_images.append(result['image_name'])
        
        return misclassified_images

    def visualize_results(self, compiled_results: List[Dict[str, Any]], images: Dict[str, np.ndarray], save_dir: str):
        """
        可视化预测结果并保存为图片。

        参数:
        compiled_results: compile_prediction_results 方法的输出
        images: 字典,键为图像名称,值为图像数组
        save_dir: 保存图片的目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for result in compiled_results:
            image_name = result['image_name']
            img = images[image_name]
            predictions = result['predictions']

            true_label = np.zeros(img.shape[:2], dtype=int)
            pred_label = np.zeros(img.shape[:2], dtype=int)

            for (row, col), pred in predictions.items():
                true_label[row, col] = pred['true_label']
                pred_label[row, col] = pred['predicted_label']

            # 保存真实标签图
            self._visualize_and_save_image(img, true_label, os.path.join(save_dir, f'{image_name}_true.png'))

            # 保存预测标签图
            self._visualize_and_save_image(img, pred_label, os.path.join(save_dir, f'{image_name}_pred.png'))

    def _visualize_and_save_image(self, img: np.ndarray, label: np.ndarray, save_path: str):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        height, width = img.shape[:2]
        num_blocks_h, num_blocks_w = label.shape

        # 绘制网格线
        for i in range(1, num_blocks_h):
            plt.axhline(y=i * height / num_blocks_h, color='w', linestyle='-', linewidth=1)
        for j in range(1, num_blocks_w):
            plt.axvline(x=j * width / num_blocks_w, color='w', linestyle='-', linewidth=1)

        # 在标签为真的格子中绘制 'O'
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if label[i, j] == 1:
                    plt.text(j * width / num_blocks_w + width / (2 * num_blocks_w),
                             i * height / num_blocks_h + height / (2 * num_blocks_h), 'O',
                             color='r', fontsize=12, ha='center', va='center')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"图像已保存到: {save_path}")

    def visualize_dataset(self, images: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], save_dir: str):
        """
        可视化数据集并保存为图片。

        参数:
        images: 字典,键为图像名称,值为图像数组
        labels: 字典,键为图像名称,值为标签数组
        save_dir: 保存图片的目录
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for image_name, img in images.items():
            label = labels[image_name]
            save_path = os.path.join(save_dir, f'{image_name}_dataset.png')
            self._visualize_and_save_dataset_image(img, label, save_path)

    def _visualize_and_save_dataset_image(self, img: np.ndarray, label: np.ndarray, save_path: str):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        height, width = img.shape[:2]
        num_blocks_h, num_blocks_w = label.shape

        # 绘制网格线
        for i in range(1, num_blocks_h):
            plt.axhline(y=i * height / num_blocks_h, color='w', linestyle='-', linewidth=1)
        for j in range(1, num_blocks_w):
            plt.axvline(x=j * width / num_blocks_w, color='w', linestyle='-', linewidth=1)

        # 在标签为1或2的格子中绘制不同颜色的 'O'
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if label[i, j] == 1:
                    color = 'r'  # 红色
                elif label[i, j] == 2:
                    color = 'b'  # 蓝色
                else:
                    continue  # 如果标签为0,不绘制任何内容

                plt.text(j * width / num_blocks_w + width / (2 * num_blocks_w),
                         i * height / num_blocks_h + height / (2 * num_blocks_h), 'O',
                         color=color, fontsize=12, ha='center', va='center')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"数据集图像已保存到: {save_path}")