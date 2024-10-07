from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import os
import numpy as np

class HeartCalcificationResultsProcessor:
    """
    心臟鈣化結果處理器類別。
    用於處理和分析心臟鈣化相關的訓練和預測結果。
    """

    def __init__(self):
        """
        初始化 HeartCalcificationResultsProcessor 類別。
        """
        self.field: Any = None  # 用於存儲結果處理器特定的數據類型

    def compile_training_results(self, list_of_image_names: List[str], data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        編譯訓練結果。

        參數:
        list_of_image_names: 圖像名稱列表
        data_dict: 包含訓練數據的字典

        返回:
        Dict[str, Any]: 圖像名稱及其相關數據的字典
        """
        compiled_results = {}
        for image_name in list_of_image_names:
            if image_name in data_dict:
                image_data = data_dict[image_name]
                compiled_results[image_name] = {
                    'split_count': image_data.split_count,
                    'labels': image_data.labels,
                    # 可以根據需要添加更多的訓練相關數據
                }
            else:
                print(f"Warning: Image {image_name} not found in data_dict")
        
        return compiled_results

    def compile_prediction_results(self, list_of_image_names: List[str], prediction_results: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
        """
        編譯預測結果。

        參數:
        list_of_image_names: 圖像名稱列表
        prediction_results: 預測結果列表,每個元素為 (key, true_label, predicted_label)

        返回:
        List[Dict[str, Any]]: 包含圖像名稱和預測結果的字典列表
        """
        compiled_results = []
        prediction_dict = {key: (true, pred) for key, true, pred in prediction_results}
        
        for image_name in list_of_image_names:
            image_predictions = {}
            for key, (true_label, pred_label) in prediction_dict.items():
                if key.startswith(image_name):
                    # 假設 key 的格式為 "image_name_row_col"
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
        計算整體預測準確率。

        參數:
        compiled_results: compile_prediction_results 方法的輸出

        返回:
        float: 預測準確率
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
        獲取被錯誤分類的圖像列表。

        參數:
        compiled_results: compile_prediction_results 方法的輸出

        返回:
        List[str]: 被錯誤分類的圖像名稱列表
        """
        misclassified_images = []
        
        for result in compiled_results:
            if any(pred['true_label'] != pred['predicted_label'] for pred in result['predictions'].values()):
                misclassified_images.append(result['image_name'])
        
        return misclassified_images

    def visualize_results(self, compiled_results: List[Dict[str, Any]], images: Dict[str, np.ndarray], save_dir: str):
        """
        可視化預測結果並保存為圖片。

        參數:
        compiled_results: compile_prediction_results 方法的輸出
        images: 字典,鍵為圖像名稱,值為圖像數組
        save_dir: 保存圖片的目錄
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

            # 保存真實標籤圖
            self._visualize_and_save_image(img, true_label, os.path.join(save_dir, f'{image_name}_true.png'))

            # 保存預測標籤圖
            self._visualize_and_save_image(img, pred_label, os.path.join(save_dir, f'{image_name}_pred.png'))

    def _visualize_and_save_image(self, img: np.ndarray, label: np.ndarray, save_path: str):
        """
        可視化單個圖像並保存。

        參數:
        img: 圖像數組
        label: 標籤數組
        save_path: 保存路徑
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        height, width = img.shape[:2]
        num_blocks_h, num_blocks_w = label.shape

        # 繪製網格線
        for i in range(1, num_blocks_h):
            plt.axhline(y=i * height / num_blocks_h, color='w', linestyle='-', linewidth=1)
        for j in range(1, num_blocks_w):
            plt.axvline(x=j * width / num_blocks_w, color='w', linestyle='-', linewidth=1)

        # 在標籤為真的格子中繪製 'O'
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

        print(f"圖像已保存到: {save_path}")

    def visualize_dataset(self, images: Dict[str, np.ndarray], labels: Dict[str, np.ndarray], save_dir: str):
        """
        可視化數據集並保存為圖片。

        參數:
        images: 字典,鍵為圖像名稱,值為圖像數組
        labels: 字典,鍵為圖像名稱,值為標籤數組
        save_dir: 保存圖片的目錄
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for image_name, img in images.items():
            label = labels[image_name]
            save_path = os.path.join(save_dir, f'{image_name}_dataset.png')
            self._visualize_and_save_dataset_image(img, label, save_path)

    def _visualize_and_save_dataset_image(self, img: np.ndarray, label: np.ndarray, save_path: str):
        """
        可視化單個數據集圖像並保存。

        參數:
        img: 圖像數組
        label: 標籤數組
        save_path: 保存路徑
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        height, width = img.shape[:2]
        num_blocks_h, num_blocks_w = label.shape

        # 繪製網格線
        for i in range(1, num_blocks_h):
            plt.axhline(y=i * height / num_blocks_h, color='w', linestyle='-', linewidth=1)
        for j in range(1, num_blocks_w):
            plt.axvline(x=j * width / num_blocks_w, color='w', linestyle='-', linewidth=1)

        # 在標籤為1或2的格子中繪製不同顏色的 'O'
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                if label[i, j] == 1:
                    color = 'r'  # 紅色
                elif label[i, j] == 0:
                    color = 'b'  # 藍色
                else:
                    continue  # 如果標籤為0,不繪製任何內容

                plt.text(j * width / num_blocks_w + width / (2 * num_blocks_w),
                         i * height / num_blocks_h + height / (2 * num_blocks_h), 'O',
                         color=color, fontsize=12, ha='center', va='center')

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        print(f"數據集圖像已保存到: {save_path}")