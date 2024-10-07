from typing import List, Tuple, Any
from heart_calcification_data_processor import HeartCalcificationDataProcessor

class ModelPrediction:
    def predict(self, model: Any, data_processor: HeartCalcificationDataProcessor) -> List[Tuple[Tuple[str, int, int], int, int]]:
        """
        使用给定的模型对心脏钙化数据进行预测。

        参数:
        model: 用于预测的机器学习模型
        data_processor: HeartCalcificationDataProcessor 实例,包含处理后的数据

        返回:
        List[Tuple[Tuple[str, int, int], int, int]]: 预测结果列表,每个元素为 (key, true_label, predicted_label)
        """
        results = []
        model_ready_data = data_processor.get_model_ready_data()

        for key, image, true_label in model_ready_data:
            # 这里假设模型有一个 predict 方法,接受图像作为输入
            # 实际使用时可能需要根据您的模型接口进行调整
            predicted_label = model.predict(image)
            
            # 将预测结果转换为整数
            predicted_label_int = int(predicted_label)
            
            results.append((key, true_label, predicted_label_int))

        return results

    def evaluate(self, predictions: List[Tuple[str, int, int]]) -> dict:
        """
        评估预测结果。

        参数:
        predictions: 预测结果列表,每个元素为 (key, true_label, predicted_label)

        返回:
        dict: 包含准确率等评估指标的字典
        """
        correct = sum(true == pred for _, true, pred in predictions)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0

        return {
            "accuracy": accuracy,
            "total_samples": total,
            "correct_predictions": correct
        }