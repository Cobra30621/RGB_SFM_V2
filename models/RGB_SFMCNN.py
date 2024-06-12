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
import matplotlib.pyplot as plt
import numpy as np

# 二值化
class ThresholdTransform(object):
    def __call__(self,x):
        threshold = x.mean()
        result = (x>threshold).to(x.dtype)
        return result
        
class Renormalize(object):
    def __call__(self,images):
        batch, channel = images.shape[0], images.shape[1]
        min_vals = images.view(batch, channel, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        max_vals = images.view(batch, channel, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        normalized_images = (images - min_vals) / (max_vals - min_vals)
        result = torch.where(max_vals == min_vals, torch.zeros_like(images), normalized_images)
        return result

class Sobel_Conv2d(object):
    def __call__(self,image):
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1).to(device = image.device)
        edge_detect = F.conv2d(Variable(image), G, padding=1)
        edge_detect = torch.mul(edge_detect, edge_detect)
        edge_detect = torch.sum(edge_detect, dim=1, keepdim=True)
        result = torch.sqrt(edge_detect)
        return edge_detect


class RGB_SFMCNN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 Conv2d_kernel, 
                 channels, 
                 SFM_filters, 
                 strides,
                 rbfs, 
                 paddings,
                 fc_input,
                 device,
                 activate_params) -> None:
        super().__init__()

        self.gray_transform = torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            # Sobel_Conv2d(),
            Renormalize(),
        ])

        is_weight_cdist = False

        self.RGB_conv2d = self._make_RGBBlock(
            3,
            channels[0][0], 
            Conv2d_kernel[0], 
            stride = strides[0],
            padding=paddings[0],
            rbf =  rbfs[0][0],
            initial='uniform', 
            device=device,
            activate_param = activate_params[0][0])

        self.GRAY_conv2d = self._make_GrayBlock(
            1,
            channels[1][0], 
            Conv2d_kernel[0], 
            stride = strides[0],
            padding=paddings[0],
            rbf =  rbfs[1][0],
            initial='kaiming', 
            device=device,
            activate_param = activate_params[1][0]
        )

        rgb_basicBlocks = []
        for i in range(1, len(Conv2d_kernel)-1):
            is_weight_cdist = False
            if i == 1:
                is_weight_cdist = False
            basicBlock = self._make_BasicBlock(
                channels[0][i-1], 
                channels[0][i], 
                Conv2d_kernel[i], 
                stride = strides[i],
                padding = paddings[i], 
                filter = SFM_filters[i],
                rbf = rbfs[0][i],  
                initial="kaiming",
                is_weight_cdist = is_weight_cdist,
                device = device,
                activate_param = activate_params[0][i])
            rgb_basicBlocks.append(basicBlock)


        self.RGB_convs = nn.Sequential(
                self.RGB_conv2d,
                SFM(filter = SFM_filters[0], device = device),
                *rgb_basicBlocks,
                self._make_ConvBlock(
                    channels[0][-2], 
                    channels[0][-1], 
                    Conv2d_kernel[-1], 
                    stride = strides[-1],
                    padding = paddings[-1], 
                    rbf = rbfs[0][-1],
                    device = device,
                    activate_param = activate_params[0][-1]
                    )
            )

        gray_basicBlocks = []
        for i in range(1, len(Conv2d_kernel)-1):
            is_weight_cdist = False
            if i == 1:
                is_weight_cdist = False
            basicBlock = self._make_BasicBlock(
                channels[1][i-1], 
                channels[1][i], 
                Conv2d_kernel[i], 
                stride = strides[i],
                padding = paddings[i], 
                filter = SFM_filters[i],
                rbf = rbfs[1][i],  
                initial="kaiming",
                is_weight_cdist = is_weight_cdist,
                device = device,
                activate_param = activate_params[1][i])
            gray_basicBlocks.append(basicBlock)

        self.Gray_convs = nn.Sequential(
                self.GRAY_conv2d,
                SFM(filter = SFM_filters[0], device = device),
                *gray_basicBlocks,
                self._make_ConvBlock(
                    channels[1][-2], 
                    channels[1][-1], 
                    Conv2d_kernel[-1], 
                    stride = strides[-1],
                    padding = paddings[-1], 
                    rbf = rbfs[1][-1],
                    device = device,
                    activate_param = activate_params[1][-1]
                    )
            )
        
        self.fc1 = nn.Sequential(
            nn.Linear(fc_input, out_channels)
        )

    def forward(self, x):
        rgb_output = self.RGB_convs(x)
        gray_output = self.Gray_convs(self.gray_transform(x))
        output = torch.concat(([rgb_output, gray_output]), dim=1)
        output = self.fc1(output.reshape(x.shape[0], -1))
        return output

    def _make_RGBBlock(self,
                    in_channels:int, 
                    out_channels:int, 
                    kernel_size:tuple,
                    stride:int = 1,
                    padding:int = 0,
                    rbf = "triangle",
                    initial: str = "kaiming",
                    device:str = "cuda",
                    activate_param = [0,0]):
        if rbf == "triangle":
            return nn.Sequential(
                RGB_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
                triangle(w=activate_param[0], requires_grad=True, device=device),
                cReLU_percent(percent=activate_param[1]),
            )
        elif rbf == "gauss":
            return nn.Sequential(
                RGB_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
                gauss(std=activate_param[0], device=device),
                cReLU(bias=activate_param[1]),
            )
        elif rbf == 'triangle and cReLU':
            return nn.Sequential(
                RGB_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = device),
                triangle(w=activate_param[0], requires_grad=True, device=device),
                cReLU(bias=activate_param[1]),
            )
        elif rbf == 'guass and cReLU_percent':
            return nn.Sequential(
                RGB_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = device),
                gauss(std=activate_param[0], device=device),
                cReLU_percent(percent=activate_param[1]),
            )
    
    def _make_GrayBlock(self,
                    in_channels, 
                    out_channels, 
                    kernel_size,
                    stride:int = 1,
                    padding:int = 0,
                    initial: str = "kaiming",
                    rbf = 'triangle',
                    device:str = "cuda",
                    activate_param = [0,0]):
        if rbf == "triangle":
            return nn.Sequential(
                Gray_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
                triangle_cReLU(w=activate_param[0], percent=activate_param[1], requires_grad = True, device=device),
            )
        elif rbf == "gauss":
            return nn.Sequential(
                Gray_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
                gauss(std=activate_param[0], device=device),
                cReLU(bias=activate_param[1]),
            )
        elif rbf == 'triangle and cReLU':
            return nn.Sequential(
                Gray_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = device),
                triangle(w=activate_param[0], requires_grad=True, device=device),
                cReLU(bias=activate_param[1]),
            )
        elif rbf == 'guass and cReLU_percent':
            return nn.Sequential(
                Gray_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = device),
                gauss(std=activate_param[0], device=device),
                cReLU_percent(percent=activate_param[1]),
            )

    def _make_BasicBlock(self,
                    in_channels:int, 
                    out_channels:int, 
                    kernel_size:tuple,
                    stride:int = 1,
                    padding:int = 0,
                    filter:tuple = (1,1),
                    rbf = "triangle",
                    initial: str = "kaiming",
                    is_weight_cdist = False,
                    device:str = "cuda",
                    activate_param = [0,0]):
        if rbf == "triangle":
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial, is_weight_cdist=is_weight_cdist, device = device),
                triangle_cReLU(w=activate_param[0], percent=activate_param[1], requires_grad = True, device=device),
                SFM(filter = filter, device = device)
            )
        elif rbf == "gauss":
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial, is_weight_cdist=is_weight_cdist, device = device),
                gauss(std=activate_param[0], device=device),
                cReLU(bias=activate_param[1]),
                SFM(filter = filter, device = device)
            )
        elif rbf == 'triangle and cReLU':
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, is_weight_cdist=is_weight_cdist, device = device),
                triangle(w=activate_param[0], requires_grad=True, device=device),
                cReLU(bias=activate_param[1]),
                SFM(filter = filter, device = device)
            )
        elif rbf == 'guass and cReLU_percent':
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, is_weight_cdist=is_weight_cdist, device = device),
                gauss(std=activate_param[0], device=device),
                cReLU_percent(percent=activate_param[1]),
                SFM(filter = filter, device = device)
            )

    def _make_ConvBlock(self,
                    in_channels, 
                    out_channels, 
                    kernel_size,
                    stride:int = 1,
                    padding:int = 0,
                    initial: str = "kaiming",
                    rbf = 'triangle',
                    device:str = "cuda",
                    activate_param = [0,0]):
        if rbf == "triangle":
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
                triangle_cReLU(w=activate_param[0], percent=activate_param[1], requires_grad = True, device=device),
            )
        elif rbf == "gauss":
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
                gauss(std=activate_param[0], device=device),
                cReLU(bias=activate_param[1]),
            )
        elif rbf == 'triangle and cReLU':
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = device),
                triangle(w=activate_param[0], requires_grad=True, device=device),
                cReLU(bias=activate_param[1]),
            )
        elif rbf == 'guass and cReLU_percent':
            return nn.Sequential(
                RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = device),
                gauss(std=activate_param[0], device=device),
                cReLU_percent(percent=activate_param[1]),
            )

'''
    RGB 卷積層
    return output shape = (batches, channels, height, width)
'''
class RGB_Conv2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial:str = "kaiming",
                 device=None,
                 dtype=None) -> None:
        super().__init__() # TODO RGB_Conv2d function
        # 24 色色相環 + RGB + black + white
        weights = [[185, 31, 87], 
                    [208, 47, 72],
                    [221, 68, 59],
                    [233, 91, 35],
                    [230, 120, 0],
                    [244, 157, 0],
                    [241, 181, 0],
                    [238, 201, 0],
                    [210, 193, 0],
                    [168, 187, 0],
                    [88, 169, 29],
                    [0, 161, 90],
                    [0, 146, 110],
                    [0, 133, 127],
                    [0, 116, 136],
                    [0, 112, 155],
                    [0, 96, 156],
                    [0, 91, 165],
                    [26, 84, 165],
                    [83, 74, 160],
                    [112, 63, 150],
                    [129, 55, 138],
                    [143, 46, 124],
                    [173, 46, 108],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [128, 128, 128]]
        
        self.weights = torch.Tensor(weights).to(device=device, dtype=dtype)
        self.weights = self.weights / 255
        self.weights = nn.Parameter(self.weights, requires_grad=False)

        self.black_block = torch.zeros((1,3)).to(device=device, dtype=dtype)
        self.white_block = torch.ones((1,3)).to(device=device, dtype=dtype)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initial = initial
        
    def forward(self, input):
        # 1. 計算色塊歐式距離
        # # weights shape = (out_channels, 3, prod(self.kernel_size))
        # weights = torch.cat([self.weights, self.black_block, self.white_block], dim=0)
        # weights = weights.reshape(*weights.shape, 1)
        # weights = weights.repeat(1,1,math.prod(self.kernel_size))

        # output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride + 1)
        # output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride + 1)
        # batch_num = input.shape[0]

        # # windows shape = (batch_num, output_width * output_height, 1, 3, prod(self.kernel_size))
        # windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)
        # windows = windows.reshape(*windows.shape[:-1], 3, math.prod(self.kernel_size)).unsqueeze(2)

        # # result shape = (batch_num, output_width * output_height, self.out_channels)
        # result = torch.pow(windows - weights, 2).reshape(batch_num, output_width * output_height, self.out_channels, -1)
        # result = torch.sum(result, dim=-1)
        # result = torch.sqrt(result + 1e-8)

        # result = result.permute(0,2,1).reshape(batch_num,self.out_channels,output_height,output_width)

        # 2. 計算代表色距離
        output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride + 1)
        output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride + 1)
        batch_num = input.shape[0]

        # weights shape = (out_channels, 3)
        weights = torch.cat([self.weights, self.black_block, self.white_block], dim=0)

        # windows shape = (batch_num, output_width * output_height, 1, 3)
        windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)
        windows = windows.reshape(*windows.shape[:-1], 3, math.prod(self.kernel_size))
        windows_RGBcolor = windows.mean(dim=-1)

        result = self.batched_LAB_distance(windows_RGBcolor.unsqueeze(-2), weights)
        result = result / 765

        # result = torch.cdist(windows_RGBcolor, weights)
        # result = result / math.sqrt(3)
        
        # result = self.weight_cdist(windows_RGBcolor.unsqueeze(-2), weights)
        # result = result / 3

        result = result.permute(0,2,1).reshape(batch_num,self.out_channels,output_height,output_width)
        
        # 3. 計算反映代表色
        # output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride + 1)
        # output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride + 1)
        # batch_num = input.shape[0]

        # # weights shape = (out_channels, 3)
        # weights = torch.cat([self.weights, self.black_block, self.white_block], dim=0)

        # # windows shape = (batch_num, output_width * output_height, 3, prod(self.kernel_size))
        # windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)
        # windows = windows.reshape(*windows.shape[:-1], 3, math.prod(self.kernel_size)).permute(0,1,3,2)

        # result = torch.cdist(windows, weights).argmax(dim=-1)
        # count_tensor = torch.zeros(*result.shape[:-1], self.out_channels, device = result.device)
        # indices = result.to(torch.int64) 
        # updates = torch.ones_like(count_tensor, dtype=count_tensor.dtype)
        # count_tensor.scatter_add_(2, indices, updates)      
        # result = count_tensor / math.prod(self.kernel_size)
        # result = result.permute(0,2,1).reshape(batch_num,self.out_channels,output_height,output_width)
        
        return result
    
    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {self.weights.shape}, cal_dist = LAB"

    def rgb_to_hsv(self, RGB):
        r, g, b = RGB
        maxc = max(r, g, b)
        minc = min(r, g, b)
        v = maxc
        if minc == maxc:
            return 0.0, 0.0, v
        s = (maxc-minc) / maxc
        rc = (maxc-r) / (maxc-minc)
        gc = (maxc-g) / (maxc-minc)
        bc = (maxc-b) / (maxc-minc)
        if r == maxc:
            h = bc-gc
        elif g == maxc:
            h = 2.0+rc-bc
        else:
            h = 4.0+gc-rc
        h = (h/6.0) % 1.0
        return h, s, v
    
    def batched_LAB_distance(self, windows_RGBcolor, weights_RGBcolor):
        R_1,G_1,B_1 = windows_RGBcolor[:, :, :, 0] * 255, windows_RGBcolor[:, :, :, 1] * 255, windows_RGBcolor[:, :, :, 2] * 255
        R_2,G_2,B_2 = weights_RGBcolor[:, 0] * 255, weights_RGBcolor[:, 1] * 255, weights_RGBcolor[:, 2] * 255
        rmean = torch.mean(torch.concat([R_1.flatten(), R_2.flatten()]))
        R = R_1 - R_2
        G = G_1 -G_2
        B = B_1 - B_2
        return torch.sqrt((2 + rmean/256) * (R**2) + 4*(G**2) + (2+(255-rmean)/256)*(B**2) + 1e-8)
    
    def weight_cdist(self, windows_RGBcolor, weights_RGBcolor):
        R_1,G_1,B_1 = windows_RGBcolor[:, :, :, 0], windows_RGBcolor[:, :, :, 1], windows_RGBcolor[:, :, :, 2]
        R_2,G_2,B_2 = weights_RGBcolor[:, 0], weights_RGBcolor[:, 1], weights_RGBcolor[:, 2]
        R = R_1 - R_2
        G = G_1 -G_2
        B = B_1 - B_2
        result = torch.sqrt(2 * (R**2) + 4 * (G**2) + 3 * (B**2) + 1e-8)
        return result


'''
    Gray 卷積層
    return output shape = (batches, channels, height, width)
'''
class Gray_Conv2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial:str = "kaiming",
                 is_weight_cdist = False,
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device':device, 'dtype':dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial = initial
        self.is_weight_cdist = is_weight_cdist

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]), **factory_kwargs)
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

        # self.weight = (self.weight - torch.min(self.weight)) / (torch.max(self.weight) - torch.min(self.weight))

        self.weight = nn.Parameter(self.weight)
    
    def forward(self, input: Tensor) -> Tensor:
        # print(input[0, 0, :, :])
        # print(f"RBF weights = {self.weight[0]}")
        output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)

        # TODO weight取平方
        # # 將weight取平方保證其範圍落在 0 ~ 1 之間
        # weights = torch.pow(self.weight, 2)

        # 1. 取絕對值距離
        # weight_expand = self.weight.unsqueeze(1).unsqueeze(2)
        # result = (windows - weight_expand).permute(1,0,2,3)
        # result = torch.abs(result).sum(dim=-1)
        
        # 2. 取歐基里德距離
        # windows = (batch, output_width * output_height, C×∏(kernel_size))
        # weight = (out_channel, C×∏(kernel_size))
        # output = (batch, out_channel, output_width * output_height)
        if self.is_weight_cdist:
            result = self.weight_cdist(windows=windows, weights=self.weight)
        else:
            result = torch.cdist(windows, self.weight).permute(0, 2, 1)


        # union = windows.unsqueeze(2).repeat(1,1,self.weight.shape[0],1).add(self.weight)
        # union = torch.sum((windows < 1e-6).float(), dim=-1)
        # union.clamp_(1e-6)
        # union = union.unsqueeze(1).repeat(1, self.weight.shape[0], 1)
        # result = result.div(union)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result
    
    def extra_repr(self) -> str:
        return f"initial = {self.initial}, is_weight_cdist = {self.is_weight_cdist},weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"

    def weight_cdist(self, windows, weights):
        N, L, C, kernel_size, out_channels = windows.shape[0], windows.shape[1], self.in_channels, self.kernel_size, self.out_channels
        
        # Initialize the weight tensor
        w = torch.zeros(C,device=windows.device)
        w[:75] = 0.25
        w[75:] = 0.75
        
        # Compute the weighted squared differences
        windows_expanded = windows.unsqueeze(1)  # Shape (N, 1, L, C * kernel_size * kernel_size)
        weights_expanded = weights.unsqueeze(0).unsqueeze(2)  # Shape (1, out_channels, 1, C * kernel_size * kernel_size)

        diff = windows_expanded - weights_expanded  # Shape (N, out_channels, L, C * kernel_size * kernel_size)
        squared_diff = diff ** 2  # Shape (N, out_channels, L, C * kernel_size * kernel_size)

        # Sum over each kernel_size * kernel_size block
        squared_diff_reshaped = squared_diff.view(N, out_channels, L, C, kernel_size[0] * kernel_size[1])
        block_sums = squared_diff_reshaped.sum(dim=-1)  # Shape (N, out_channels, L, C)

        # Multiply by the weights
        weighted_block_sums = block_sums * w  # Broadcasting w to match block_sums shape

        # Sum over the feature dimension and take the square root
        distances = torch.sqrt(weighted_block_sums.sum(dim=-1))  # Shape (N, out_channels, L)

        return distances

'''
    RBF 卷積層
    return output shape = (batches, channels, height, width)
'''
class RBF_Conv2d(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial:str = "kaiming",
                 is_weight_cdist = False,
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device':device, 'dtype':dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.initial = initial
        self.is_weight_cdist = is_weight_cdist

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]), **factory_kwargs)
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
        # print(input[0, 0, :, :])
        # print(f"RBF weights = {self.weight[0]}")
        output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        # Unfold output = (batch, output_width * output_height, C×∏(kernel_size))
        windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)

        # TODO weight取平方
        # # 將weight取平方保證其範圍落在 0 ~ 1 之間
        # weights = torch.pow(self.weight, 2)

        # 1. 取絕對值距離
        # weight_expand = self.weight.unsqueeze(1).unsqueeze(2)
        # result = (windows - weight_expand).permute(1,0,2,3)
        # result = torch.abs(result).sum(dim=-1)
        
        # 2. 取歐基里德距離
        # windows = (batch, output_width * output_height, C×∏(kernel_size))
        # weight = (out_channel, C×∏(kernel_size))
        # output = (batch, out_channel, output_width * output_height)
        if self.is_weight_cdist:
            result = self.weight_cdist(windows=windows, weights=self.weight)
        else:
            result = torch.cdist(windows, self.weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result
    
    def extra_repr(self) -> str:
        return f"initial = {self.initial}, is_weight_cdist = {self.is_weight_cdist},weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"

    def weight_cdist(self, windows, weights):
        N, L, C, kernel_size, out_channels = windows.shape[0], windows.shape[1], self.in_channels, self.kernel_size, self.out_channels
        
        # Initialize the weight tensor
        w = torch.zeros(C,device=windows.device)
        w[:75] = 0.25
        w[75:] = 0.75
        
        # Compute the weighted squared differences
        windows_expanded = windows.unsqueeze(1)  # Shape (N, 1, L, C * kernel_size * kernel_size)
        weights_expanded = weights.unsqueeze(0).unsqueeze(2)  # Shape (1, out_channels, 1, C * kernel_size * kernel_size)

        diff = windows_expanded - weights_expanded  # Shape (N, out_channels, L, C * kernel_size * kernel_size)
        squared_diff = diff ** 2  # Shape (N, out_channels, L, C * kernel_size * kernel_size)

        # Sum over each kernel_size * kernel_size block
        squared_diff_reshaped = squared_diff.view(N, out_channels, L, C, kernel_size[0] * kernel_size[1])
        block_sums = squared_diff_reshaped.sum(dim=-1)  # Shape (N, out_channels, L, C)

        # Multiply by the weights
        weighted_block_sums = block_sums * w  # Broadcasting w to match block_sums shape

        # Sum over the feature dimension and take the square root
        distances = torch.sqrt(weighted_block_sums.sum(dim=-1))  # Shape (N, out_channels, L)

        return distances

class triangle(nn.Module):
    def __init__(self, 
                 w: float, 
                 requires_grad: bool = False, 
                 device:str = "cuda"):
        super().__init__()
        self.w = torch.Tensor([w]).to(device)
        if requires_grad:
            self.w = nn.Parameter(self.w, requires_grad = True)

    def forward(self, d):
        w_tmp = self.w
        d[d>=w_tmp] = w_tmp
        return torch.ones_like(d) - torch.div(d, w_tmp)
    
    def extra_repr(self) -> str:
        return f"w = {self.w.item()}"

class gauss(nn.Module):
    def __init__(self, std, requires_grad: bool = True, device:str = "cuda"):
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
                 device:str = "cuda") -> None:
        super().__init__()
        self.bias = torch.tensor([bias]).to(device)
        if requires_grad:
            self.bias = nn.Parameter(self.bias, requires_grad = True)
    
    def forward(self, x):
        bias_tmp = self.bias
        result = x * torch.ge(x, bias_tmp.repeat(x.shape[0]).view(-1,1,1,1)).float()
        return result
    
    def extra_repr(self) -> str:
        return f"bias={self.bias.item()}"
    
class cReLU_percent(nn.Module):
    def __init__(self, 
                 percent: float = 0.5,
                 requires_grad: bool = True,
                 device:str = "cuda") -> None:
        super().__init__()
        self.percent = torch.tensor([percent]).to(device)
    
    def forward(self, x):
        x_flatten = x.reshape(x.shape[0], -1)
        top_k, _ = x_flatten.topk(math.ceil(self.percent * x_flatten.shape[1]), dim=1, largest=True)
        threshold = top_k[:, -1]
        threshold = threshold.view(-1,1,1,1)

        result = torch.where(x >= threshold, x, 0).view(*x.shape)
        return result
    
    def extra_repr(self) -> str:
        return f"percent={self.percent.item()}"

class triangle_cReLU(nn.Module):
    def __init__(self, 
                 w: float,
                 percent: float,
                 requires_grad: bool = False, 
                 device:str = "cuda"):
        super().__init__()
        self.w = torch.Tensor([w]).to(device)
        self.percent = torch.tensor([percent]).to(device)
        if requires_grad:
            self.w = nn.Parameter(self.w, requires_grad = True)

    def forward(self, d):
        # input()
        w_tmp = self.w
        # print(f'd = {d[0]}')

        # 1. 取所有數字的對應percent值當作唯一threshold
        d_flatten = d.reshape(d.shape[0], -1)
        top_k, _ = d_flatten.topk(math.ceil(self.percent * d_flatten.shape[1]), dim=1, largest=False)
        threshold = top_k[:, -1]
        # 將 threshold 中大於 w 的元素設為 w
        threshold[threshold>w_tmp] = w_tmp
        threshold = threshold.view(-1,1,1,1)
        # print(f'threshold = {threshold}')

        # #2. 每個channel獨立計算threshold
        # threshold, _ = d.topk(int(self.percent * d.shape[1]), dim=1, largest=False)
        # threshold = threshold[:, -1, :, :][:, None, :, :]
        # # 將 threshold 中大於 w 的元素設為 w
        # threshold[threshold > w_tmp] = w_tmp
        # # print(f'threshold = {threshold[0]}')

        # #3. 取beta
        # threshold  = w_tmp * (1 - self.beta)

        # # 4. threshold 取 最大值 * weight
        # topk, _ = d.topk(k = 1, dim=1)
        # threshold = topk * self.percent

        d = torch.where(d > threshold, w_tmp, d).view(*d.shape)
        result = (torch.ones_like(d) - torch.div(d, w_tmp))
        return result
    
    def extra_repr(self) -> str:
        return f"w = {self.w.item()}, percent={self.percent.item()}"

'''
    時序合併層
    parameters:
        filter: 合併的範圍
'''
class SFM(nn.Module):
    def __init__(self,
                 filter: _size_2_t,
                 alpha_max: float = 0.99,
                 alpha_min: float = 0.9,
                 device: str = "cuda") -> None:
        super(SFM, self).__init__()
        self.filter = filter
        self.alpha = torch.linspace(start=alpha_min, end=alpha_max, steps = math.prod(self.filter), requires_grad=True).reshape(*self.filter)
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        alpha_pows = self.alpha.repeat(input.shape[1], 1, 1).to(self.device)

        batch_num, channels, height, width = input.shape
        _, filter_h, filter_w = alpha_pows.shape

        # 使用 unfold 將 input 展開成形狀為 (batch_num, channels, (height-filter_h+step)*(width-filter_w+step), filter_h * filter_w) 的二維張量
        unfolded_input = input.unfold(2, filter_h, filter_h).unfold(3, filter_w, filter_w).reshape(batch_num, channels, -1, filter_h * filter_w)
        # print(f"unfolded_input = {unfolded_input.shape}")
        # print(unfolded_input)


        # 將 filter 擴展成形狀為 (1, channels, 1, filter_h * filter_w)
        expanded_filter = alpha_pows.reshape(channels, 1, -1)
        expanded_filter = expanded_filter.repeat(batch_num, 1, 1, 1)
        # print(f"expanded_filter = {expanded_filter.shape}")
        # print(expanded_filter)

        # 對應相乘
        result = unfolded_input * expanded_filter
        # print(f"result = {result.shape}")

        # 將 dim=-1 的維度相加取 mean
        output_width = math.floor((width - (filter_w - 1) - 1) / filter_w + 1)
        output_height = math.floor((height -  (filter_h - 1) - 1) / filter_h + 1)
        output = result.mean(dim=-1).reshape(batch_num, channels, output_height, output_width)
        # print(f"output = {output.shape}")
        return output
    
    def extra_repr(self) -> str:
        return f"filter={self.filter}, alpha={self.alpha.detach().numpy()}"