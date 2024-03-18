import torch

from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from torch.nn import init
from torch import nn
from torch import Tensor
from torchvision.transforms import Grayscale

import torch.nn.functional as F
import torch
import math

class RGB_SFMCNN(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 Conv2d_kernel, 
                 channels, 
                 SFM_filters, 
                 strides, 
                 paddings,
                 w_arr,
                 percent,
                 fc_input,
                 device) -> None:
        super().__init__()

        
        # TODO 檢查是否各個block的initial function
        self.RGB_conv2d = self._make_RGBBlock(75, Conv2d_kernel[0], stride = strides[0], w = w_arr[0], percent = percent[0], initial='uniform', device=device)
        self.GRAY_conv2d = self._make_ConvBlock(1, 25, Conv2d_kernel[0], stride = strides[0], w = w_arr[0], percent = percent[0], initial = 'kaiming', device=device)
        self.SFM = SFM(filter = SFM_filters[0], device = device)

        self.convs = nn.ModuleList([
            nn.Sequential(
                *[self._make_BasicBlock(channels[i], 
                                        channels[i+1], 
                                        Conv2d_kernel[i], 
                                        stride = strides[i],
                                        padding = paddings[i], 
                                        filter = SFM_filters[i], 
                                        percent=percent[i],
                                        w = w_arr[i], 
                                        initial="kaiming",
                                        device = device) for i in range(1, len(SFM_filters))],
                self._make_ConvBlock(channels[-2], 
                                     channels[-1], 
                                     Conv2d_kernel[-1], 
                                     stride = strides[-1],
                                     padding = paddings[-1], 
                                     percent=percent[-1], 
                                     w=w_arr[-1], 
                                     device = device)
            )
        ])


        self.fc1 = nn.Sequential(
            nn.Linear(fc_input, out_channels)
        )

    def _make_BasicBlock(self,
                    in_channels:int, 
                    out_channels:int, 
                    kernel_size:tuple,
                    stride:int = 1,
                    padding:int = 0,
                    filter:tuple = (1,1),
                    w:float = 0.4,
                    percent: float = 0.5,
                    initial: str = "kaiming",
                    device:str = "cuda"):
        return nn.Sequential(
            RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
            triangle_cReLU(w=w, percent=percent, requires_grad = True, device=device),
            SFM(filter = filter, device = device)
        )
    
    def _make_RGBBlock(self,
                    out_channels:int, 
                    kernel_size:tuple,
                    stride:int = 1,
                    padding:int = 0,
                    filter:tuple = (1,1),
                    w:float = 0.4,
                    percent: float = 0.5,
                    initial: str = "kaiming",
                    device:str = "cuda"):
        return nn.Sequential(
            RGB_Conv2d(out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial,device = device),
            triangle_cReLU(w=w, percent=percent, requires_grad = True, device=device),
        )

    def _make_ConvBlock(self,
                    in_channels, 
                    out_channels, 
                    kernel_size,
                    stride:int = 1,
                    padding:int = 0,
                    w = 4.0,
                    percent = 0.4,
                    initial: str = "kaiming",
                    device:str = "cuda"):
        return nn.Sequential(
            RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, initial = initial, device = device),
            triangle_cReLU(w=w, percent=percent, requires_grad = True, device=device),
        )
    
    def forward(self, x):
        rgb_output = self.RGB_conv2d(x)
        gray_output = self.GRAY_conv2d(Grayscale()(x))
        output = torch.concat(([rgb_output, gray_output]), dim=1)
        output = self.SFM(output)
        output = self.convs[0](output)
        # print(torch.max(output), torch.min(output))
        output = self.fc1(output.reshape(x.shape[0], -1))
        return output
    
class RGB_Conv2d(nn.Module):
    def __init__(self,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 initial:str = "kaiming",
                 device=None,
                 dtype=None) -> None:
        super().__init__() # TODO RGB_Conv2d function
        weights_R = torch.empty((out_channels, 1))
        weights_G = torch.empty((out_channels, 1))
        weights_B = torch.empty((out_channels, 1))

        if initial == "kaiming":
            torch.nn.init.kaiming_uniform_(weights_R)
            torch.nn.init.kaiming_uniform_(weights_G)
            torch.nn.init.kaiming_uniform_(weights_B)
        elif initial == "uniform":
            torch.nn.init.uniform(weights_R)
            torch.nn.init.uniform(weights_G)
            torch.nn.init.uniform(weights_B)
        else:
            raise "RGB_Conv2d initial error"
        
        self.weights = torch.cat([weights_R, weights_G, weights_B], dim=-1).to(device=device, dtype=dtype)
        self.weights = nn.Parameter(self.weights)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.initial = initial
        
    def forward(self, input):
        # weights shape = (out_channels, 3, prod(self.kernel_size))
        weights = self.weights.reshape(*self.weights.shape, 1)
        weights = weights.repeat(1,1,math.prod(self.kernel_size))

        output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride + 1)
        output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride + 1)
        batch_num = input.shape[0]

        # windows shape = (batch_num, output_width * output_height, 1, 3, prod(self.kernel_size))
        windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)
        windows = windows.reshape(*windows.shape[:-1], 3, math.prod(self.kernel_size)).unsqueeze(2)

        # result shape = (batch_num, output_width * output_height, self.out_channels)
        result = torch.pow(windows - weights, 2).reshape(batch_num, output_width * output_height, self.out_channels, -1)
        result = torch.sum(result, dim=-1)
        result = torch.sqrt(result)

        result = result.permute(0,2,1).reshape(batch_num,self.out_channels,output_height,output_width)
        return result
    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {self.weights.shape}"

        

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
        windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)

        # TODO weight取平方
        # # 將weight取平方保證其範圍落在 0 ~ 1 之間
        # weights = torch.pow(self.weight, 2)

        #1. 取絕對值距離
        # weight_expand = self.weight.unsqueeze(1).unsqueeze(2)
        # result = (windows - weight_expand).permute(1,0,2,3)
        # result = torch.abs(result).sum(dim=-1)
        
        #2. 取歐基里德距離
        result = torch.cdist(windows, self.weight).permute(0, 2, 1)

        result = result.reshape(result.shape[0], result.shape[1], output_height, output_width)
        return result
    
    def extra_repr(self) -> str:
        return f"initial = {self.initial}, weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"

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
        return torch.exp(torch.pow(d, 2) / (-2 * torch.pow(self.std, 2)))
    
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
        # print(f"triangle_cRelu after: {torch.max(result)} ~ {torch.min(result)}")
        # print('-----')
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