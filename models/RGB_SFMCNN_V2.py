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
                 SFM_method,
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
        self.RGB_conv2d = self._make_RGBBlock(
            3,
            channels[0][0],
            Conv2d_kernel[0],
            stride=strides[0],
            padding=paddings[0],
            rbfs=rbfs[0][0],
            initial='uniform',
            device=device,
            activate_param=activate_params[0][0])

        # 輪廓卷積第一層
        self.GRAY_conv2d = self._make_GrayBlock(
            1,
            channels[1][0],
            Conv2d_kernel[0],
            stride=strides[0],
            padding=paddings[0],
            rbfs=rbfs[1][0],
            initial=initial[1][0],
            device=device,
            activate_param=activate_params[1][0],
            conv_method=conv_method[1][0]
        )

        #   彩色第二層以後的特徵傳遞區塊(色彩特徵傳遞區塊)
        rgb_basicBlocks = []
        for i in range(1, len(Conv2d_kernel) - 1):
            basicBlock = self._make_BasicBlock(
                channels[0][i - 1],
                channels[0][i],
                Conv2d_kernel[i],
                stride=strides[i],
                padding=paddings[i],
                filter=SFM_filters[i],
                rbfs=rbfs[0][i],
                SFM_method=SFM_method,
                initial=initial[0][i],
                device=device,
                activate_param=activate_params[0][i],
                conv_method=conv_method[0][i]
            )
            rgb_basicBlocks.append(basicBlock)

        # 整個彩色部分
        self.RGB_convs = nn.Sequential(
            self.RGB_conv2d,
            SFM(filter=SFM_filters[0], device=device, method=SFM_method),
            *rgb_basicBlocks,
            self._make_ConvBlock(
                channels[0][-2],
                channels[0][-1],
                Conv2d_kernel[-1],
                stride=strides[-1],
                padding=paddings[-1],
                rbfs=rbfs[0][-1],
                device=device,
                activate_param=activate_params[0][-1],
                conv_method=conv_method[0][-1]
            )
        )

        #   灰階第二層以後的特徵傳遞區塊(輪廓特徵傳遞區塊)
        gray_basicBlocks = []
        for i in range(1, len(Conv2d_kernel) - 1):
            basicBlock = self._make_BasicBlock(
                channels[1][i - 1],
                channels[1][i],
                Conv2d_kernel[i],
                stride=strides[i],
                padding=paddings[i],
                filter=SFM_filters[i],
                rbfs=rbfs[1][i],
                SFM_method=SFM_method,
                initial=initial[1][i],
                device=device,
                activate_param=activate_params[1][i],
                conv_method=conv_method[1][i]
            )
            gray_basicBlocks.append(basicBlock)

        # 整個灰階部分
        self.Gray_convs = nn.Sequential(
            self.GRAY_conv2d,
            SFM(filter=SFM_filters[0], device=device, method=SFM_method),
            *gray_basicBlocks,
            self._make_ConvBlock(
                channels[1][-2],
                channels[1][-1],
                Conv2d_kernel[-1],
                stride=strides[-1],
                padding=paddings[-1],
                rbfs=rbfs[1][-1],
                device=device,
                initial=initial[1][-1],
                activate_param=activate_params[1][-1],
                conv_method=conv_method[1][-1]
            )
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
                       in_channels: int,
                       out_channels: int,
                       kernel_size: tuple,
                       stride: int = 1,
                       padding: int = 0,
                       rbfs=['gauss', 'cReLU_percent'],
                       initial: str = "kaiming",
                       device: str = "cuda",
                       activate_param=[0, 0]):
        layers = [RGB_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                             initial=initial, device=device)]

        # 建立響應模組
        for rbf in rbfs:
            print(rbf)
            layers.append(get_rbf(rbf, activate_param, device, required_grad=False))

        return nn.Sequential(*layers)

    '''
        第一層灰階卷積Block(不含空間合併)
    '''

    def _make_GrayBlock(self,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride: int = 1,
                        padding: int = 0,
                        initial: str = "kaiming",
                        conv_method: str = "cdist",
                        rbfs=['gauss', 'cReLU_percent'],
                        device: str = "cuda",
                        activate_param=[0, 0]):
        layers = [Gray_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                            initial=initial, conv_method=conv_method, device=device)]

        # 建立響應模組
        for rbf in rbfs:
            layers.append(get_rbf(rbf, activate_param, device))

        return nn.Sequential(*layers)

    def _make_BasicBlock(self,
                         in_channels: int,
                         out_channels: int,
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

        layers = [RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           initial=initial,  conv_method=conv_method, device=device)]

        # 建立響應模組
        for rbf in rbfs:
            layers.append(get_rbf(rbf, activate_param, device))

        layers.append(SFM(filter=filter, device=device, method = SFM_method))

        return nn.Sequential(*layers)


    def _make_ConvBlock(self,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride: int = 1,
                        padding: int = 0,
                        initial: str = "kaiming",
                        rbfs=['gauss', 'cReLU_percent'],
                        device: str = "cuda",
                        activate_param=[0, 0],
                        conv_method: str = "cdist"):
        layers = [RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                             initial=initial, conv_method=conv_method, device=device)]

        # 建立響應模組
        for rbf in rbfs:
            layers.append(get_rbf(rbf, activate_param, device))


        return nn.Sequential(*layers)



'''
    RGB 卷積層
    return output shape = (batches, channels, height, width)
'''


class RGB_Conv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial: str = "kaiming",
                 device=None,
                 dtype=None) -> None:
        super().__init__()  # TODO RGB_Conv2d function
        # 30 個濾波器
        # color_weights = [[255, 255, 255], [219, 178, 187], [210, 144, 98], [230, 79, 56], [207, 62, 108], [130, 44, 28], [91, 31, 58], [209, 215, 63], [194, 202, 119], [224, 148, 36], [105, 147, 29], [131, 104, 50], [115, 233, 72], [189, 211, 189], [109, 215, 133], [72, 131, 77], [69, 81, 65], [77, 212, 193], [101, 159, 190], [120, 142, 215], [121, 102, 215], [111, 42, 240], [75, 42, 185], [57, 41, 119], [42, 46, 71], [216, 129, 199], [214, 67, 205], [147, 107, 128], [136, 48, 133], [0, 0, 0]]

        # self.color_weights = torch.Tensor(color_weights).to(device=device, dtype=dtype)
        # self.color_weights = self.color_weights / 255
        # self.color_weights =  nn.Parameter(self.color_weights, requires_grad=False)

        # 30 個濾波器
        weights = [[255, 255, 255], [219, 178, 187], [210, 144, 98], [230, 79, 56], [207, 62, 108], [130, 44, 28],
                   [91, 31, 58], [209, 215, 63], [194, 202, 119], [224, 148, 36], [105, 147, 29], [131, 104, 50],
                   [115, 233, 72], [189, 211, 189], [109, 215, 133], [72, 131, 77], [69, 81, 65], [77, 212, 193],
                   [101, 159, 190], [120, 142, 215], [121, 102, 215], [111, 42, 240], [75, 42, 185], [57, 41, 119],
                   [42, 46, 71], [216, 129, 199], [214, 67, 205], [147, 107, 128], [136, 48, 133], [0, 0, 0]]
        

        self.weights = torch.Tensor(weights).to(device=device, dtype=dtype)
        self.weights = self.weights / 255
        self.weights = nn.Parameter(self.weights, requires_grad=False)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initial = initial

        self.use_average = True



    def forward(self, input_tensor):

        weights = self.weights.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 5, 5)  # 變成 (30, 3, 5, 5)

        # 使用 unfold 取得 5x5 的卷積區域 (展開操作)
        unfold = torch.nn.Unfold(kernel_size=5, stride=4)
        input_unfolded = unfold(input_tensor)  # shape = (-1, 75, 36)
        # print(f"input_unfolded{input_unfolded.shape}")

        # 重新塑形成適合計算的格式
        input_unfolded = input_unfolded.view(-1, 3, 5, 5, 6, 6)  # (1000, 3, 5, 5, 6, 6)
        input_unfolded = input_unfolded.permute(0, 4, 5, 1, 2, 3)  # (1000, 6, 6, 3, 5, 5)
        #
        # print(f"input_unfolded after {input_unfolded.shape}")
        # print(f"weight {weights.shape}")
        #
        # print(f"input_unfolded.unsqueeze(1) {input_unfolded.unsqueeze(1).shape}")
        # print(f"weights.unsqueeze(0).unsqueeze(2).unsqueeze(3) {weights.unsqueeze(0).unsqueeze(2).unsqueeze(3).shape}")
        #

        # 計算顏色相似度 (可以選擇不同公式)
        distances = self. batched_LAB_distance(input_unfolded.unsqueeze(1), weights.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        # distances shape = (1000, 30, 6, 6)
        # print("Output shape:", distances.shape)

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
        return self.weights

    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {self.weights.shape}, cal_dist = LAB"

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


    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"





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



'''
    時序合併層
    parameters:
        filter: 合併的範圍
'''


class SFM(nn.Module):
    def __init__(self,
                 filter: _size_2_t,
                 alpha_max: float = 0.9,
                 alpha_min: float = 0.99,
                 device: str = "cuda",
                 method: str = "alpha_mean") -> None:
        super(SFM, self).__init__()
        self.filter = filter
        self.alpha = torch.linspace(start=alpha_min, end=alpha_max, steps=math.prod(self.filter),
                                    requires_grad=True).reshape(*self.filter)
        self.device = device
        self.method = method

    def forward(self, input: Tensor) -> Tensor:
        batch_num, channels, height, width = input.shape
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
        return f"filter={self.filter}, alpha={self.alpha.detach().numpy()}"


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