import numpy as np
import scipy.stats
import torch


def calculate_RM(layers, layer_num, images):
    """
    計算 layer_num 層，對於所有圖片的輸出值
    """
    RM = layers[layer_num](images)

    filter_count = RM.shape[1]
    reshape_RM = RM.permute(1, 0, 2, 3).reshape(filter_count, -1)

    return reshape_RM


def get_stats(reshape_RM):
    """
    计算层的统计数据，包括每个通道的均值、标准差、最大值、最小值、峰值和偏值等。

    参数:
    - reshape_RM: 重塑后的特征图，形状为 (通道数, 像素数)

    返回:
    - channel_stats: 每个通道的统计信息字典
    - overall_stats: 整个层的统计信息字典
    """

    overall_stats = {
        "mean": float(reshape_RM.mean()),
        "std": float(reshape_RM.std()),
        "max": float(reshape_RM.max()),
        "min": float(reshape_RM.min()),
    }

    # 归一化到 0-1 之间
    min_value = reshape_RM.min()
    max_value = reshape_RM.max()
    normalized_reshape_RM = (reshape_RM - min_value) / (max_value - min_value)  # 归一化处理

    # 计算每个通道的统计指标
    channel_mean_values = normalized_reshape_RM.mean(dim=1)  # 每个通道的平均值
    channel_std_values = normalized_reshape_RM.std(dim=1)  # 每个通道的标准差
    channel_max_values = normalized_reshape_RM.max(dim=1).values  # 每个通道的最大值
    channel_min_values = normalized_reshape_RM.min(dim=1).values  # 每个通道的最小值

    # 计算每个通道的峰值和偏值
    channel_kurtosis_values = scipy.stats.kurtosis(normalized_reshape_RM.cpu().detach().numpy(), axis=1)  # 每个通道的峰值
    channel_skewness_values = scipy.stats.skew(normalized_reshape_RM.cpu().detach().numpy(), axis=1)  # 每个通道的偏值

    # 计算相对于最大值的阈值统计
    count_above_0_99 = (normalized_reshape_RM > 0.99).sum(dim=1)
    ratio_above_0_99 = (count_above_0_99 / normalized_reshape_RM.shape[1]).tolist()

    count_above_0_9 = (normalized_reshape_RM > 0.9).sum(dim=1)
    ratio_above_0_9 = (count_above_0_9 / normalized_reshape_RM.shape[1]).tolist()

    count_below_0_1 = (normalized_reshape_RM < 0.1).sum(dim=1)
    ratio_below_0_1 = (count_below_0_1 / normalized_reshape_RM.shape[1]).tolist()

    # 计算每个通道的统计指标并存储在字典中
    channel_stats = {
        "mean": channel_mean_values.tolist(),
        "std": channel_std_values.tolist(),
        "max": channel_max_values.tolist(),
        "min": channel_min_values.tolist(),
        "kurtosis": channel_kurtosis_values.tolist(),  # 峰值
        "skewness": channel_skewness_values.tolist(),  # 偏值
        "ratio_above_0.99": ratio_above_0_99,
        "ratio_above_0.9": ratio_above_0_9,
        "ratio_below_0.1": ratio_below_0_1,
    }

    return channel_stats, overall_stats