import numpy as np
import scipy.stats
import torch


def smooth_threshold(tensor, condition, threshold):
    """
    使用 Sigmoid 來平滑地統計 tensor 中符合指定條件的元素比例。

    參數:
    - tensor: 要統計的 Tensor
    - condition: 字符串，'>' 表示大於，'<' 表示小於
    - threshold: 用於比較的閾值

    返回:
    - 平滑統計結果 (float)
    """
    if condition == '>':
        result = torch.sigmoid(10 * (tensor - threshold)).mean()
    elif condition == '<':
        result = torch.sigmoid(10 * (threshold - tensor)).mean()
    else:
        raise ValueError("condition 參數僅限 '>' 或 '<'")

    return result


def create_metrics_dict():
    """
    創建包含多種統計指標計算方法的字典
    將整個資料集輸入模型，分析每個濾波器通道的輸出分布。以下為濾波器通道需滿足的條件與指標：

    返回:
    - 包含指標計算方法的字典
    """
    metrics = {
        '避免高效反應 (ratio_above_0.9 < 20%)': lambda stats: (
            smooth_threshold(stats['ratio_above_0.9'], '<', 0.2)
        ),
        '避免低效反應 (ratio_above_0.1 > 1%)': lambda stats: (
            smooth_threshold(stats['ratio_above_0.1'], '>', 0.01)
        ),
        '最大值限制 (each channel max > 0.8)': lambda stats: (
            smooth_threshold(stats['max'], '>', 0.8)
        ),


        # 1. 避免過於極端的反應
        # ## 避免高效反應：濾波器通道輸出大於 0.9 的數據比例需低於 50%。
        # '避免高效反應 (ratio_above_0.9 < 50%)': lambda stats: len(
        #     [ratio for ratio in stats["ratio_above_0.9"] if ratio < 0.5]) / len(
        #     stats["ratio_above_0.9"]),
        # ## 避免低效反應 ：濾波器通道輸出大於 0.1 的數據比例需高於 1%。
        # '避免低效反應 (ratio_above_0.1 > 1%)': lambda stats: len([ratio for ratio in stats["ratio_above_0.1"] if ratio > 0.01]) / len(
        #     stats["ratio_above_0.1"]),
        # # 2.針對特定特徵的有效反應
        # ## 最大值限制 ：每個濾波器通道的輸出最大值需大於 0.8
        # '最大值限制 (each channel max > 0.8)': lambda stats: len([max_val for max_val in stats["max"] if max_val > 0.8]) / len(
        #     stats["max"]),
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