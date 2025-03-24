from torch import torch
import torch.nn as nn
import torch.nn.functional as F

from monitor.monitor_method import get_all_layers_stats


def get_loss_function(loss_name: str, weight=None, reduction='mean'):
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    elif loss_name == 'MSELoss':
        return nn.MSELoss()
    elif loss_name == 'MetricBaseLoss':
        return MetricBaseLoss()
    # 可以根據需要添加更多損失函數
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


class MetricBaseLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        """
        自定義損失函數，包含：
        - 基礎損失 (CrossEntropyLoss)
        - 平均預測值的懲罰
        - 每個濾波器通道的額外懲罰

        Args:
            weight (Tensor, optional): CrossEntropyLoss 的類別權重。
        """
        super(MetricBaseLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.each_channel_max_weight = 1
        self.min_weight = 1


    def forward(self, predictions, targets, model, layers, layers_infos, images):
        """
        計算損失和懲罰。

        Args:
            predictions (Tensor): 預測分數，形狀為 (batch_size, num_classes)。
            targets (Tensor): 目標類別，形狀為 (batch_size)。

        Returns:
            Tensor: 總損失值。
        """
        # 基礎損失
        base_loss = self.loss_fn(predictions, targets)


        images.requires_grad_()
        layer_stats, overall_stats = get_all_layers_stats(model, layers, layers_infos, images, keep_tensor=True, without_RGBConv0=True)

        # print(layer_stats['RGB_convs_2']['最大值限制 (each channel max > 0.8)'])

        # each_channel_max =  torch.tensor(overall_stats['最大值限制 (each channel max > 0.8)'], requires_grad=True)
        # max_loss = (1 - each_channel_max) * self.each_channel_max_weight
        max_loss = overall_stats['最大值限制 (each channel max > 0.8)']
        min_loss = overall_stats['避免低效反應 (ratio_above_0.1 > 1%)']

        total_loss = base_loss - max_loss * self.each_channel_max_weight - min_loss * self.min_weight

        # print(overall_stats)

        # print(overall_stats['最大值限制 (each channel max > 0.8)'])
        print(f"base: {base_loss}")
        print(f"max_loss: {max_loss}")
        print(f"min_loss: {min_loss}")

        # 返回總損失
        return total_loss