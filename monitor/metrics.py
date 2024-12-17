import numpy as np
import scipy.stats
import torch

def create_metrics_dict():
    """
    創建包含多種統計指標計算方法的字典

    返回:
    - 包含指標計算方法的字典
    """
    metrics = {
        # 计算每个通道的最大值大于 0.9 的比例
        'each channel max > 0.9': lambda stats: len([max_val for max_val in stats["max"] if max_val > 0.9]) / len(
            stats["max"]),

        # 计算所有通道中，反应值大于 0.99 的比例小于 0.1 的通道的比例
        'ratio_above_0.99 < 0.1': lambda stats: len(
            [ratio for ratio in stats["ratio_above_0.99"] if ratio < 0.1]) / len(stats["ratio_above_0.99"]),

        # 计算所有通道中，反应值大于 0.9 的比例小于 0.8 的通道的比例
        'ratio_above_0.9 < 0.8': lambda stats: len([ratio for ratio in stats["ratio_above_0.9"] if ratio < 0.8]) / len(
            stats["ratio_above_0.9"]),

        # 计算所有通道中，反应值小于 0.1 的比例小于 1 的通道的比例
        'ratio_below_0.1 < 1': lambda stats: len([ratio for ratio in stats["ratio_below_0.1"] if ratio < 1]) / len(
            stats["ratio_below_0.1"]),
    }

    return metrics


def calculate_layer_metrics(stats):
    """
    計算層的各種指標

    參數:
    - stats: 統計信息

    返回:
    - 包含各種指標的字典
    """

    # 創建指標計算方法字典
    metrics = create_metrics_dict()

    # 計算所有指標
    layer_metrics = {}
    for metric_name, metric_func in metrics.items():
        try:
            layer_metrics[metric_name] = metric_func(stats)
        except Exception as e:
            print(f"Error calculating {metric_name}: {e}")
            layer_metrics[metric_name] = None

    return layer_metrics