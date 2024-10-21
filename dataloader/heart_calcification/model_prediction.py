from typing import List, Tuple, Any
from heart_calcification_data_processor import HeartCalcificationDataProcessor

def predict(model: Any, data_processor: HeartCalcificationDataProcessor) -> List[Tuple[str, List[int], List[int]]]:
    """
    使用给定的模型对心脏钙化数据进行预测。

    参数:
    model: 用于预测的机器学习模型
    data_processor: HeartCalcificationDataProcessor 实例,包含处理后的数据

    返回:
    List[Tuple[str, List[int], List[int]]]: 预测结果列表,每个元素为 (imageName, true_labels, predicted_labels)
    """

    results = []
    for image_name, image_data in data_processor.data_dict.items():
        predict_labels = model.predict(image_data.split_images)
        results.append((image_data.image_name, predict_labels, image_data.labels))

    return results
