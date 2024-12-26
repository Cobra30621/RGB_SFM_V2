from torch import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_function(loss_name: str, weight=None, reduction='mean'):
    if loss_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss(weight=weight, reduction=reduction)
    elif loss_name == 'MSELoss':
        return nn.MSELoss()
    elif loss_name == 'CustomLoss':
        return CustomLoss()  # 假設您已經定義了 CustomLoss
    # 可以根據需要添加更多損失函數
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


class CustomLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean',
                 penalty_scale=10.0, upper_threshold=0.9, lower_threshold=0.1,
                 filter_penalty_scale=5.0, reflect_threshold=0.9, reflect_ratio_threshold=0.5):
        """
        自定義損失函數，包含：
        - 基礎損失 (CrossEntropyLoss)
        - 平均預測值的懲罰
        - 每個濾波器通道的額外懲罰

        Args:
            weight (Tensor, optional): CrossEntropyLoss 的類別權重。
            reduction (str, optional): 損失的縮減方式 ('mean' 或 'sum')。
            penalty_scale (float): 平均預測值的懲罰倍率。
            upper_threshold (float): 預測平均值的上限閾值。
            lower_threshold (float): 預測平均值的下限閾值。
            filter_penalty_scale (float): 濾波器懲罰的倍率。
            reflect_threshold (float): 通道輸出反映值的閾值。
            reflect_ratio_threshold (float): 輸出反映值超過閾值的比例閾值。
        """
        super(CustomLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
        self.penalty_scale = penalty_scale
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.filter_penalty_scale = filter_penalty_scale
        self.reflect_threshold = reflect_threshold
        self.reflect_ratio_threshold = reflect_ratio_threshold

    def forward(self, predictions, targets):
        """
        計算損失和懲罰。

        Args:
            predictions (Tensor): 預測分數，形狀為 (batch_size, num_classes)。
            targets (Tensor): 目標類別，形狀為 (batch_size)。

        Returns:
            Tensor: 總損失值。
        """
        print(f"prediction: {predictions.shape}")
        # 基礎損失
        base_loss = self.loss_fn(predictions, targets)

        print(base_loss)

        # 每個濾波器通道的懲罰
        filter_penalty = 0.0
        # 假設 predictions 維度為 (batch_size, num_filters, feature_map_size)
        if predictions.dim() > 2:  # 檢查 predictions 是否為高維數據
            max_per_channel = predictions.amax(dim=(0, 2))  # 每個濾波器通道的最大值
            reflect_ratio_per_channel = (predictions > self.reflect_threshold).float().mean(dim=(0, 2))  # 超過閾值的比例

            # 懲罰條件
            filter_penalty += torch.sum((max_per_channel < self.reflect_threshold).float() * self.filter_penalty_scale)
            filter_penalty += torch.sum(
                (reflect_ratio_per_channel > self.reflect_ratio_threshold).float() * self.filter_penalty_scale)

        # 返回總損失
        return base_loss +  filter_penalty