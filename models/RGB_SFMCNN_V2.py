import torch

from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from torch.nn import init
from torch import nn
from torch import Tensor
from torch.autograd import Variable

import torchvision.transforms
import torch.nn.functional as F
import torch
import math

'''
來自 2025 陳俊宇的碩士論文:
'''

def get_feature_extraction_layers(model):
    """
    獲取模型中各層特徵圖的提取器（RGB 與灰階分支）

    參數:
        model: 要分析的模型

    回傳:
        (dict, dict): RGB 分支與灰階分支的特徵圖提取器字典
    """
    rgb_layers = extract_branch_layers(model.RGB_convs, "RGB_convs", CI_only=False)
    gray_layers = extract_branch_layers(model.Gray_convs, "Gray_convs", CI_only=False)
    return rgb_layers, gray_layers


def get_CI_target_layers(model):
    """
    獲取用於 CI（Critical Input）分析所需的層（僅提取包含 activation 的部分）

    參數:
        model: 要分析的模型

    回傳:
        (dict, dict): RGB 分支與灰階分支的 CI 分析層字典
    """
    rgb_CI_layers = extract_branch_layers(model.RGB_convs, "RGB_convs", CI_only=True)
    gray_CI_layers = extract_branch_layers(model.Gray_convs, "Gray_convs", CI_only=True)
    return rgb_CI_layers, gray_CI_layers


def extract_branch_layers(conv_blocks, branch_name, CI_only=False):
    """
    建立特定卷積分支（RGB 或 Gray）中各層的特徵提取模組。

    參數:
        conv_blocks: 該分支的 Sequential 卷積區塊（例如 model.RGB_convs）
        branch_name: 層的命名前綴，例如 "RGB_convs"
        CI_only: 若為 True，僅提取經過 activation 的層（供 CI 使用）

    回傳:
        dict: 層名對應的特徵提取模組
    """
    layer_extractors = {}

    for i in range(len(conv_blocks)):
        layer_prefix = f"{branch_name}_{i}"  # ex: RGB_convs_0

        # 取得前 i 層 + 第 i 層的前兩個模組（通常是 conv + activation）
        layer_extractors[f"{layer_prefix}"] = nn.Sequential(
            *(list(conv_blocks[:i])) + list([conv_blocks[i][:2]])
        )

        # 若為完整特徵提取用途，則加入更多細部節點
        if not CI_only:
            # 卷積層輸出（不含 activation）
            layer_extractors[f"{layer_prefix}_after_Conv"] = nn.Sequential(
                *(list(conv_blocks[:i])) + list([conv_blocks[i][:1]])
            )

            # activation 輸出（通常是 conv + activation）
            layer_extractors[f"{layer_prefix}_activation"] = nn.Sequential(
                *(list(conv_blocks[:i])) + list([conv_blocks[i][:2]])
            )

            # 若存在 SFM（空間特徵融合）則包含該層
            if len(conv_blocks[i]) >= 3:
                layer_extractors[f"{layer_prefix}_SFM"] = nn.Sequential(
                    *(list(conv_blocks[:i])) + list([conv_blocks[i][:3]])
                )

    return layer_extractors


'''
    主模型Part
'''
class RGB_SFMCNN_V2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 Conv2d_kernel,
                 channels,
                 SFM_filters,
                 strides,
                 conv_method,
                 initial,
                 rbfs,
                 SFM_methods,
                 paddings,
                 fc_input,
                 device,
                 activate_params) -> None:
        super().__init__()

        # 灰階前處理
        self.gray_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            # Sobel_Conv2d(),
            # Renormalize(),
            NormalizeToRange(),
        ])

        # 彩色卷積第一層
        RGB_conv2d = self._make_RGBBlock(
            in_channels,
            channels[0][0],
            Conv2d_kernel[0],
            stride=strides[0],
            padding=paddings[0],
            rbfs=rbfs[0][0],
            initial='uniform',
            SFM_method= SFM_methods[0][0],
            SFM_filters = SFM_filters[0],
            device=device,
            activate_param=activate_params[0][0])

        #   彩色第二層以後的特徵傳遞區塊(色彩特徵傳遞區塊)
        rgb_basicBlocks = []
        for i in range(1, len(Conv2d_kernel)):
            basicBlock = self._make_BasicBlock(
                channels[0][i - 1],
                channels[0][i],
                Conv2d_kernel[i],
                stride=strides[i],
                padding=paddings[i],
                filter=SFM_filters[i],
                rbfs=rbfs[0][i],
                SFM_method=SFM_methods[0][i],
                initial=initial[0][i],
                device=device,
                activate_param=activate_params[0][i],
                conv_method=conv_method[0][i]
            )
            rgb_basicBlocks.append(basicBlock)

        # 整個彩色部分
        self.RGB_convs = nn.Sequential(
            RGB_conv2d,
            *rgb_basicBlocks
        )

        # 輪廓卷積第一層
        GRAY_conv2d = self._make_GrayBlock(
            1,
            channels[1][0],
            Conv2d_kernel[0],
            stride=strides[0],
            padding=paddings[0],
            rbfs=rbfs[1][0],
            initial=initial[1][0],
            SFM_method=SFM_methods[1][0],
            SFM_filters=SFM_filters[0],
            device=device,
            activate_param=activate_params[1][0],
            conv_method=conv_method[1][0]
        )

        #   灰階第二層以後的特徵傳遞區塊(輪廓特徵傳遞區塊)
        gray_basicBlocks = []
        for i in range(1, len(Conv2d_kernel)):
            basicBlock = self._make_BasicBlock(
                channels[1][i - 1],
                channels[1][i],
                Conv2d_kernel[i],
                stride=strides[i],
                padding=paddings[i],
                filter=SFM_filters[i],
                rbfs=rbfs[1][i],
                SFM_method=SFM_methods[1][i],
                initial=initial[1][i],
                device=device,
                activate_param=activate_params[1][i],
                conv_method=conv_method[1][i]
            )
            gray_basicBlocks.append(basicBlock)

        # 整個灰階部分
        self.Gray_convs = nn.Sequential(
            GRAY_conv2d,
            *gray_basicBlocks,
        )

        self.fc1 = nn.Sequential(
            nn.Linear(fc_input, out_channels)
        )

    def forward(self, x):
        rgb_output = self.RGB_convs(x)
        gray_output = self.Gray_convs(self.gray_transform(x))
        rgb_output = rgb_output.reshape(x.shape[0], -1)
        gray_output = gray_output.reshape(x.shape[0], -1)
        output = torch.concat(([rgb_output, gray_output]), dim=-1)
        output = self.fc1(output)
        return output

    '''
        第一層彩色卷積Block(不含空間合併)
    '''

    def _make_RGBBlock(self,
                       in_channel: int,
                       out_channels: tuple,
                       kernel_size: tuple,
                       stride: int = 1,
                       padding: int = 0,
                       initial: str = "kaiming",
                       rbfs=['gauss', 'cReLU_percent'],
                       activate_param=[0, 0],
                       SFM_method: str = "alpha_mean",
                       SFM_filters: tuple = (1, 1),
                       device: str = "cuda"):
        # 基礎層
        layers = []

        # rgb 卷基層
        out_channel = out_channels[0] * out_channels[1]
        rgb_conv_layer = RGB_Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                             initial=initial, device=device)
        # 建立響應模組
        rbf_layer = make_rbfs(rbfs, activate_param, device, required_grad=False)
        # 建立空間合併模組
        sfm_layer = SFM(filter=SFM_filters, device=device, method=SFM_method)

        layers.append(rgb_conv_layer)
        layers.append(rbf_layer)
        layers.append(sfm_layer)

        return nn.Sequential(*layers)




    '''
        第一層灰階卷積Block(不含空間合併)
    '''

    def _make_GrayBlock(self,
                        in_channel : int,
                        out_channels: tuple,
                        kernel_size,
                        stride: int = 1,
                        padding: int = 0,
                        conv_method: str = "cdist",
                        initial: str = "kaiming",
                        rbfs=['gauss', 'cReLU_percent'],
                        activate_param=[0, 0],
                        SFM_method: str = "alpha_mean",
                        SFM_filters: tuple = (1, 1),
                        device: str = "cuda"):
        # 基礎層
        layers = []
        # gray 卷基層
        out_channel = out_channels[0] * out_channels[1]
        rgb_conv_layer = Gray_Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                            initial=initial, conv_method=conv_method, device=device)
        # 建立響應模組
        rbf_layer = make_rbfs(rbfs, activate_param, device, required_grad=False)
        # 建立空間合併模組
        sfm_layer = SFM(filter=SFM_filters, device=device, method=SFM_method)

        layers.append(rgb_conv_layer)
        layers.append(rbf_layer)
        layers.append(sfm_layer)

        return nn.Sequential(*layers)

    def _make_BasicBlock(self,
                         in_channels: tuple,
                         out_channels: tuple,
                         kernel_size: tuple,
                         stride: int = 1,
                         padding: int = 0,
                         filter: tuple = (1, 1),
                         rbfs=['gauss', 'cReLU_percent'],
                         SFM_method: str ="alpha_mean",
                         initial: str = "kaiming",
                         device: str = "cuda",
                         activate_param=[0, 0],
                         conv_method: str = "cdist"):
        # 基礎層
        layers = []
        # rbf 卷基層
        in_channel = in_channels[0] * in_channels[1]
        out_channel = out_channels[0] * out_channels[1]
        rgf_conv_layer = RBF_Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                           initial=initial,  conv_method=conv_method, device=device)
        # 建立響應模組
        rbf_layer = make_rbfs(rbfs, activate_param, device, required_grad=False)

        layers.append(rgf_conv_layer)
        layers.append(rbf_layer)

        # 建立空間合併模組
        if SFM_method != "none":
            sfm_layer = SFM(filter=filter, device=device, method=SFM_method)
            layers.append(sfm_layer)

        return nn.Sequential(*layers)




'''
    RGB 卷積層
    return output shape = (batches, channels, height, width)
'''

import torch
import torch.nn as nn


class RGB_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 device=None,
                 dtype=None) -> None:
        super().__init__()

        weights = [[255, 255, 255], [219, 178, 187], [210, 144, 98], [230, 79, 56], [207, 62, 108], [130, 44, 28],
                   [91, 31, 58], [209, 215, 63], [194, 202, 119], [224, 148, 36], [105, 147, 29], [131, 104, 50],
                   [115, 233, 72], [189, 211, 189], [109, 215, 133], [72, 131, 77], [69, 81, 65], [77, 212, 193],
                   [101, 159, 190], [120, 142, 215], [121, 102, 215], [111, 42, 240], [75, 42, 185], [57, 41, 119],
                   [42, 46, 71], [216, 129, 199], [214, 67, 205], [147, 107, 128], [136, 48, 133], [0, 0, 0]]

        kernel_h, kernel_w = kernel_size

        # 轉成 tensor 並正規化，shape: (30, 3)
        weight_tensor = torch.tensor(weights, dtype=dtype, device=device) / 255.0

        # 變成 shape: (30, 3, 1, 1)
        weight_tensor = weight_tensor.unsqueeze(-1).unsqueeze(-1)

        # 擴展成 shape: (30, 3, kernel_h, kernel_w)
        weight_tensor = weight_tensor.repeat(1, 1, kernel_h, kernel_w)

        # 註冊為可訓練參數
        self.weight = nn.Parameter(weight_tensor, requires_grad=True)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initial = initial
        self.use_average = True

    def forward(self, input_tensor):
        # input_tensor shape: (B, 3, H, W)
        unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)
        input_unfolded = unfold(input_tensor)

        height, width = int(input_unfolded.shape[2] ** 0.5), int(input_unfolded.shape[2] ** 0.5)
        input_unfolded = input_unfolded.view(-1, 3, self.kernel_size[0], self.kernel_size[1], height, width)
        input_unfolded = input_unfolded.permute(0, 4, 5, 1, 2, 3)

        # weights shape: (30, 3, kernel_h, kernel_w)
        distances = self.batched_LAB_distance(
            input_unfolded.unsqueeze(1),
            self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        return distances


    '''
        LAB Distance
    '''

    def batched_LAB_distance(self, windows_RGBcolor, weights_RGBcolor):
        # RGB 顏色數據從 [0, 1] 映射到 [0, 255]
        R_1, G_1, B_1 = (windows_RGBcolor[:, :, :, :, 0] * 255, windows_RGBcolor[:, :, :, :, 1] * 255,
                         windows_RGBcolor[:, :, :, :, 2] * 255)
        R_2, G_2, B_2 = weights_RGBcolor[:, :, :, :, 0] * 255, weights_RGBcolor[:, :, :, :, 1] * 255, weights_RGBcolor[:, :, :, :, 2] * 255

        # 計算 rmean 作為矩陣形式
        rmean = (R_1 + R_2) / 2

        # 計算 RGB 分量的差異
        R = R_1 - R_2
        G = G_1 - G_2
        B = B_1 - B_2

        # 計算加權的歐幾里得距離
        distance = torch.sqrt(
            (2 + rmean / 256) * (R ** 2) +
            4 * (G ** 2) +
            (2 + (255 - rmean) / 256) * (B ** 2) +
            1e-8
        ).sum(dim=4).sum(dim=4)

        distance = distance / (25 * 768)
        #
        # print(f"distance {distance.shape}")
        # print(f"distance {distance}")

        return distance

    def color_similarity(self, input_patch, weight_kernel, method="delta_e"):
        """
        計算輸入圖像區塊與權重之間的顏色相似度
        input_patch: (1000, 6, 6, 3, 5, 5) -> 區塊圖像
        weight_kernel: (30, 3, 5, 5) -> 權重過濾器
        method: 選擇的顏色公式, 預設使用 Delta E

        返回: (1000, 30, 6, 6) 相似度張量
        """
        if method == "delta_e":
            # 假設 RGB 是近似 Lab 顏色空間 (不是真的 Lab 轉換)
            delta = (input_patch - weight_kernel) ** 2
            # print(f"delta {delta.shape}")
            # print(f"distance {torch.sqrt(delta.sum(dim=4).sum(dim=4).sum(dim=4))}")

            return torch.sqrt(delta.sum(dim=4).sum(dim=4).sum(dim=4)) / 10 # (1000, 30, 6, 6)

        elif method == "cosine":
            # Cosine 相似度 = dot(A, B) / (||A|| * ||B||)
            input_flat = input_patch.flatten(start_dim=3)  # (1000, 6, 6, 3*5*5)
            weight_flat = weight_kernel.flatten(start_dim=2)  # (30, 3*5*5)

            dot_product = (input_flat * weight_flat.unsqueeze(0).unsqueeze(2).unsqueeze(3)).sum(dim=4)
            norm_input = torch.norm(input_flat, dim=4)
            norm_weight = torch.norm(weight_flat, dim=2, keepdim=True)

            return dot_product / (norm_input * norm_weight)  # (1000, 30, 6, 6)

        else:
            raise ValueError("不支援的方法，請選擇 'delta_e' 或 'cosine'")


    # 使用卷積
    def _dot_product(self, input: Tensor) -> Tensor:
        # 使用卷積層進行計算
        result = F.conv2d(input, self.weight.view(self.out_channels, self.out_channels, *self.kernel_size),
                          stride=self.stride, padding=self.padding)
        return result

    def transform_weights(self):
        return self.weight

    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {self.weight.shape}, cal_dist = LAB"

    def rgb_to_hsv(self, RGB):
        r, g, b = RGB
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc
        if minc == maxc:
            return 0.0, 0.0, v
        s = (maxc - minc) / maxc
        rc = (maxc - r) / (maxc - minc)
        gc = (maxc - g) / (maxc - minc)
        bc = (maxc - b) / (maxc - minc)
        if r == maxc:
            h = bc - gc
        elif g == maxc:
            h = 2.0 + rc - bc
        else:
            h = 4.0 + gc - rc
        h = (h / 6.0) % 1.0
        return h, s, v



'''
    Gray 卷積層
    return output shape = (batches, channels, height, width)
'''


class Gray_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 conv_method: str = "cdist",
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial = initial
        self.conv_method = conv_method

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]),
                                  **factory_kwargs)
        self.reset_parameters(initial)

    def reset_parameters(self, initial) -> None:
        if initial == "kaiming":
            # kaiming 初始化
            # bound  = sqrt(6/(1 + a^2 * fan))
            # fan = self.weight.size(1) * 1
            init.kaiming_uniform_(self.weight)
        elif initial == "uniform":
            init.uniform_(self.weight)
        else:
            raise "RBF_Conv2d initial error"

        # 将第一个输出通道的权重设置为0
        with torch.no_grad():
            self.weight[0].fill_(-1)
            self.weight[1].fill_(1)

        # self.weight = (self.weight - torch.min(self.weight)) / (torch.max(self.weight) - torch.min(self.weight))

        self.weight = nn.Parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if self.conv_method == "cdist":
            return self._cdist(input)
        elif self.conv_method == "dot_product":
            return self._dot_product(input)
        elif self.conv_method == "squared_cdist":
            return self._squared_cdist(input)
        elif self.conv_method == "cosine":
            return self._cosine(input)
        else:
            print(f"Can't find {self.conv_method} conv method")

    # 使用距離公式
    def _cdist(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        result = torch.cdist(windows, self.weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result

    # Weight 取平方的後，距離公式
    def _squared_cdist(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        # 將 self.weight 取平方
        squared_weight = self.weight ** 2

        result = torch.cdist(windows, squared_weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result

    # 使用餘弦相似度
    def _cosine(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2, 1)

        # 計算 windows 和 weight 的 L2 範數
        windows_norm = torch.norm(windows, p=2, dim=2, keepdim=True)
        weight_norm = torch.norm(self.weight, p=2, dim=1, keepdim=True)


        # 計算點積
        dot_product = torch.matmul(windows, self.weight.t())


        # 計算餘弦相似度
        cosine = dot_product / (windows_norm * weight_norm.t() + 1e-8)
        # 調整維度順序並重塑
        result = cosine.permute(0, 2, 1)
        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result

    # 使用卷積
    def _dot_product(self, input: Tensor) -> Tensor:
        # 使用卷積層進行計算
        result = F.conv2d(input, self.weight.view(self.out_channels, self.in_channels, *self.kernel_size),
                          stride=self.stride, padding=self.padding)
        return result

    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"




'''
    RBF 卷積層
    return output shape = (batches, channels, height, width)
'''


class RBF_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 conv_method: str = "cdist",
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial = initial
        self.conv_method = conv_method

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]),
                                  **factory_kwargs)
        self.reset_parameters(initial)

    def reset_parameters(self, initial) -> None:
        if initial == "kaiming":
            # kaiming 初始化
            # bound  = sqrt(6/(1 + a^2 * fan))
            # fan = self.weight.size(1) * 1
            init.kaiming_uniform_(self.weight)
        elif initial == "uniform":
            init.uniform_(self.weight)
        else:
            raise "RBF_Conv2d initial error"

        self.weight = nn.Parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        if self.conv_method == "cdist":
            return self._cdist(input)
        elif self.conv_method == "dot_product":
            return self._dot_product(input)
        elif self.conv_method == "squared_cdist":
            return self._squared_cdist(input)
        elif self.conv_method == "cosine":
            return self._cosine(input)
        else:
            print(f"Can't find {self.conv_method} conv method")


    # 使用距離公式
    def _cdist(self, input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        result = torch.cdist(windows, self.weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result

    # Weight 取平方的後，距離公式
    def _squared_cdist(self,  input: Tensor) -> Tensor:
        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)
        # 將 self.weight 取平方
        squared_weight = self.weight ** 2

        result = torch.cdist(windows, squared_weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)

        return result


    # 使用卷積
    def _dot_product(self, input: Tensor) -> Tensor:
        # 使用卷積層進行計算
        result = F.conv2d(input, self.weight.view(self.out_channels, self.in_channels, *self.kernel_size),
                           stride=self.stride, padding=self.padding)
        return result

    # 使用餘弦相似度
    def _cosine(self, input: Tensor) -> Tensor:

        output_width = math.floor(
            (input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor(
            (input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).permute(0, 2,
                                                                                                                  1)

        # 計算 windows 和 weight 的 L2 範數(向量的歐基里德距離)
        windows_norm = torch.norm(windows, p=2, dim=2, keepdim=True)
        weight_norm = torch.norm(self.weight, p=2, dim=1, keepdim=True)

        # 計算點積
        dot_product = torch.matmul(windows, self.weight.t())

        # 計算餘弦相似度
        cosine = dot_product / (windows_norm * weight_norm.t() + 1e-8)

        # 調整維度順序並重塑
        result = cosine.permute(0, 2, 1)
        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result


    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"


'''
    前處理Part(自由取用)
'''


class Renormalize(object):
    def __call__(self, images):
        batch, channel = images.shape[0], images.shape[1]
        min_vals = images.view(batch, channel, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_vals = images.view(batch, channel, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        normalized_images = (images - min_vals) / (max_vals - min_vals)
        result = torch.where(max_vals == min_vals, torch.zeros_like(images), normalized_images)
        return result


class NormalizeToRange(object):
    def __call__(self, images):
        """
        将图像归一化到 [-1, 1] 区间

        Args:
            images: 输入图像张量

        Returns:
            归一化后的图像张量，范围在 [-1, 1] 之间
        """
        batch, channel = images.shape[0], images.shape[1]
        min_vals = images.view(batch, channel, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_vals = images.view(batch, channel, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)

        # 先归一化到 [0, 1]
        normalized_images = (images - min_vals) / (max_vals - min_vals + 1e-8)
        # 然后转换到 [-1, 1]
        result = normalized_images * 2 - 1

        # 处理特殊情况：当 max_vals == min_vals 时，返回零张量
        return torch.where(max_vals == min_vals, torch.zeros_like(images), result)


class AvoidAll0(object):
    def __call__(self, images):
        """
        将图像轉到 [-1, 1] 区间，避免全黑圖片都是 0

        Args:
            images: 输入图像张量

        Returns:
            归一化后的图像张量，范围在 [-1, 1] 之间
        """

        # 然后转换到 [-1, 1]
        result = images * 2 - 1

        # 处理特殊情况：当 max_vals == min_vals 时，返回零张量
        return result


'''
    響應過濾模組(輸入陣列)
'''
def make_rbfs(rbfs, activate_param, device, required_grad=True):
    # 建立響應模組
    rbf_layers = []
    for rbf in rbfs:
        print(rbf)
        rbf_layers.append(get_rbf(rbf, activate_param, device, required_grad=required_grad))
    return nn.Sequential(*rbf_layers)


'''
    響應過濾模組(所有可能性)
'''
def get_rbf(rbf, activate_param, device, required_grad = True):
    if rbf == "triangle":
        return triangle(w=activate_param[0], requires_grad=required_grad, device=device)
    elif rbf == "gauss":
        return gauss(std=activate_param[0], device=device)
    elif rbf == 'sigmoid':
        return Sigmoid()
    elif rbf == 'cReLU':
        return cReLU()
    elif rbf == 'cReLU_percent':
        return cReLU_percent(percent=activate_param[1])
    elif rbf == 'bia_gauss':
        return bia_gauss(std=activate_param[0], device=device)
    elif rbf == 'regularization':
        return MeanVarianceRegularization()
    else:
        raise ValueError(f"Unknown RBF type: {rbf}")


class triangle(nn.Module):
    def __init__(self, w: float, requires_grad: bool = True, device: str = "cuda"):
        super().__init__()
        self.w = torch.Tensor([w]).to(device)
        if requires_grad:
            self.w = nn.Parameter(self.w, requires_grad=True)

    def forward(self, d):
        # Ensure that values below w are linearly increasing and values above w are linearly decreasing
        w_tmp = self.w
        d_clamped = torch.clamp(d, max=w_tmp)  # Clamp d values at w_tmp to ensure triangular shape

        # Create the triangular function:
        # When d < w_tmp, the function increases linearly, and when d >= w_tmp, it decreases
        result = torch.abs(torch.ones_like(d) - torch.div(d_clamped, w_tmp))
        return result

    def extra_repr(self) -> str:
        return f"w = {self.w.item()}"

class gauss(nn.Module):
    def __init__(self, std, requires_grad: bool = True, device: str = "cuda"):
        super().__init__()
        self.std = torch.Tensor([std]).to(device)
        if requires_grad:
            self.std = nn.Parameter(self.std)

    def forward(self, d):
        # print('Before Gauss:', torch.max(d), torch.min(d))
        # self.std = torch.std(d.reshape(d.shape[0], -1), dim = -1).reshape(-1, 1, 1, 1)
        # print('std:', torch.max(self.std), torch.min(self.std))
        # self.std = self.std.repeat(1, *d.shape[1:])
        result = torch.exp(torch.pow(d, 2) / (-2 * torch.pow(self.std, 2)))
        return result

    def extra_repr(self) -> str:
        return f"std={self.std.item()}"


class cReLU(nn.Module):
    def __init__(self,
                 bias: float = 0.7,
                 requires_grad: bool = True,
                 device: str = "cuda") -> None:
        super().__init__()
        self.bias = torch.tensor([bias]).to(device)
        if requires_grad:
            self.bias = nn.Parameter(self.bias, requires_grad=True)

    def forward(self, x):
        bias_tmp = self.bias
        result = x * torch.ge(x, bias_tmp.repeat(x.shape[0]).view(-1, 1, 1, 1)).float()
        return result

    def extra_repr(self) -> str:
        return f"bias={self.bias.item()}"


class cReLU_percent(nn.Module):
    def __init__(self, percent: float = 0.5, device: str = "cuda") -> None:
        super().__init__()
        self.percent = torch.tensor([percent]).to(device)
        self.device = device

    def forward(self, x):
        # 攤平成 [batch, h * w, count]
        batch_size, count, h, w = x.shape
        x_flatten = x.permute(0, 2, 3, 1).reshape(batch_size, h * w, count)

        # 計算每個位置要保留的前 k 個元素
        k = math.ceil(self.percent * count)  # 保留前 0.4 * count 的元素
        top_k, _ = x_flatten.topk(k, dim=2, largest=True)  # 找到每個 [h * w] 的 top-k 值
        threshold = top_k[:, :, -1].unsqueeze(2)  # 每個 [h * w] 的最小保留值，形狀為 [batch, h * w, 1]

        # 應用閥值篩選，確保同時滿足 >= threshold 和 >= 0
        x_filtered = torch.where(x_flatten >= threshold, x_flatten, torch.tensor(0.0))
        x_filtered = torch.clamp(x_filtered, min=0)  # 確保所有值都大於等於 0

        # 恢復到原始形狀
        result = x_filtered.reshape(batch_size, h, w, count).permute(0, 3, 1, 2)

        return result


    def extra_repr(self) -> str:
        return f"percent={self.percent.item()}"


class MeanVarianceRegularization(nn.Module):
    def __init__(self, target_range=(0, 1), epsilon=1e-8):
        super().__init__()
        self.target_min, self.target_max = target_range
        self.epsilon = epsilon

    def forward(self, x):
        # 計算每個 feature map 的最小值和最大值
        min_per_map = x.amin(dim=(0, 2, 3), keepdim=True)  # shape = [1, C, 1, 1] 
        max_per_map = x.amax(dim=(0, 2, 3), keepdim=True)  # shape = [1, C, 1, 1]

        # Min-max 正規化到目標範圍
        x_norm = (x - min_per_map) / (max_per_map - min_per_map + self.epsilon)
        x_final = x_norm * (self.target_max - self.target_min) + self.target_min
        
        return x_final


'''
    時序合併層
    parameters:
        filter: 合併的範圍
'''
class SFM(nn.Module):
    def __init__(self,
                 filter: _size_2_t,
                 alpha_max: float = 0.9,
                 alpha_min: float = 1.0,
                 device: str = "cuda",
                 method: str = "alpha_mean") -> None:
        super(SFM, self).__init__()
        self.filter = filter
        self.alpha = torch.linspace(start=alpha_min, end=alpha_max, steps=math.prod(self.filter),
                                    requires_grad=False).reshape(*self.filter)
        self.device = device
        self.method = method
        
        # 新增可訓練的卷積權重參數
        # if method == "conv":
        #     self.conv_weight = nn.Parameter(
        #         torch.empty(1, 1, *self.filter).uniform_(-0.1, 0.1)
        #     )
        # else:
        #     self.register_buffer("conv_weight", torch.zeros(1, 1, *self.filter))

    def forward(self, input: Tensor) -> Tensor:
        batch_num, channels, height, width = input.shape
        
        # if self.method == "conv":
        #     # 將權重擴展到對應的通道數
        #     expanded_weight = self.conv_weight.repeat(channels, 1, 1, 1)
        #
        #     # 進行卷積操作
        #     output = F.conv2d(
        #         input=input,
        #         weight=expanded_weight,
        #         stride=self.filter,  # 使用 filter 作為 stride
        #         groups=channels  # 每個通道獨立卷積
        #     )
        #     return output
            
        # 其他方法保持不變
        alpha_pows = self.alpha.repeat(input.shape[1], 1, 1).to(self.device)
        _, filter_h, filter_w = alpha_pows.shape

        unfolded_input = input.unfold(2, filter_h, filter_h).unfold(3, filter_w, filter_w).reshape(batch_num, channels,
                                                                                               -1,
                                                                                               filter_h * filter_w)

        if self.method == "max":
            # 直接取最大值，不進行與 alpha 的乘法計算
            output_width = math.floor((width - (filter_w - 1) - 1) / filter_w + 1)
            output_height = math.floor((height - (filter_h - 1) - 1) / filter_h + 1)
            output = unfolded_input.max(dim=-1).values.reshape(batch_num, channels, output_height, output_width)
        elif self.method == "alpha_mean":  # 預設為 mean

            expanded_filter = alpha_pows.reshape(channels, 1, -1)
            expanded_filter = expanded_filter.repeat(batch_num, 1, 1, 1)

            result = unfolded_input * expanded_filter

            output_width = math.floor((width - (filter_w - 1) - 1) / filter_w + 1)
            output_height = math.floor((height - (filter_h - 1) - 1) / filter_h + 1)
            output = result.mean(dim=-1).reshape(batch_num, channels, output_height, output_width)
        else:
            raise ValueError(f"Unknown method: {self.method}")  # 處理未知方法的情況


        return output

    def extra_repr(self) -> str:
        base_repr = f"filter={self.filter}, method={self.method}"
        if self.method == "conv":
            return base_repr + f", conv_weight_shape={self.conv_weight.shape}"
        return base_repr + f", alpha={self.alpha.detach().numpy()}"


class Sigmoid(nn.Module):
    def __init__(self,
                 requires_grad: bool = True,
                 device: str = "cuda") -> None:
        super().__init__()
        self.requires_grad = requires_grad
        self.device = device

    def forward(self, x):
        return torch.sigmoid(x)

    def extra_repr(self) -> str:
        return "Sigmoid activation"


class bia_gauss(nn.Module):
    def __init__(self, std: float = 1.0, requires_grad: bool = True, device: str = "cuda"):
        super().__init__()
        self.std = torch.tensor(std).to(device)
        if requires_grad:
            self.std = nn.Parameter(self.std)
 
    def forward(self, x):
        # 計算 1 - 以 0 為中心的高斯函數
        return 1 - torch.exp(-((x) ** 2) / (2 * self.std ** 2))
 
    def extra_repr(self) -> str:
        return f"std={self.std.item()}"