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

    # print(f"reshape_RM : {reshape_RM}")
    # print(f"reshape_RM : {reshape_RM.requires_grad}")

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
        # "mean": reshape_RM.mean().item(),
        # "std": reshape_RM.std().item(),
        "total_max": reshape_RM.max().item(),
        "total_min": reshape_RM.min().item(),
    }



    # 归一化到 0-1 之间
    min_value = reshape_RM.min()
    max_value = reshape_RM.max()
    normalized_reshape_RM = (reshape_RM - min_value) / (max_value - min_value)  # 归一化处理

    # 计算每个通道的统计指标
    # channel_mean_values = normalized_reshape_RM.mean(dim=1)  # 每个通道的平均值
    # channel_std_values = normalized_reshape_RM.std(dim=1)  # 每个通道的标准差
    channel_max_values = normalized_reshape_RM.max(dim=1).values  # 每个通道的最大值
    channel_min_values = normalized_reshape_RM.min(dim=1).values  # 每个通道的最小值

    # print(f"channel_max_values {channel_min_values}")
    # print(f"channel_max_values {channel_min_values.requires_grad}")

    # 计算每个通道的峰值和偏值
    # channel_kurtosis_values = torch.mean((normalized_reshape_RM - normalized_reshape_RM.mean(dim=1, keepdim=True)) ** 4,
    #                                      dim=1)
    # channel_skewness_values = torch.mean((normalized_reshape_RM - normalized_reshape_RM.mean(dim=1, keepdim=True)) ** 3,
    #                                      dim=1)

    # 计算相对于最大值的阈值统计
    count_above_0_99 = (normalized_reshape_RM > 0.99).sum(dim=1)
    ratio_above_0_99 = (count_above_0_99 / normalized_reshape_RM.shape[1])

    count_above_0_9 = (normalized_reshape_RM > 0.9).sum(dim=1)
    ratio_above_0_9 = (count_above_0_9 / normalized_reshape_RM.shape[1])

    count_above_0_1 = (normalized_reshape_RM > 0.1).sum(dim=1)
    ratio_above_0_1 = (count_above_0_1 / normalized_reshape_RM.shape[1])

    # 计算每个通道的统计指标并存储在字典中
    channel_stats = {
        # "mean": channel_mean_values.tolist(),
        # "std": channel_std_values.tolist(),
        "max": channel_max_values,
        "min": channel_min_values,
        # "kurtosis": channel_kurtosis_values.tolist(),  # 峰值
        # "skewness": channel_skewness_values.tolist(),  # 偏值
        "ratio_above_0.99": ratio_above_0_99,
        "ratio_above_0.9": ratio_above_0_9,
        "ratio_above_0.1": ratio_above_0_1
    }

    return channel_stats, overall_stats