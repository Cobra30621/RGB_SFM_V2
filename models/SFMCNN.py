from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
from torch.nn import init
from torch import nn
from torch import Tensor

import torch.nn.functional as F
import torch
import math

class SOMNetwork(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        Conv2d_kernel = [(5, 5), (3, 3), (3, 3), (1, 1)]
        SFM_filters = [(2, 2), (2, 2), (2, 2)]
        self.convs = nn.ModuleList([
            nn.Sequential(
                self._make_BasicBlock(1, 32, Conv2d_kernel[0], padding = 3, filter = SFM_filters[0], bias=0.4),
                self._make_BasicBlock(32, 64, Conv2d_kernel[1], padding = 4, filter = SFM_filters[1], bias=0.1),
                self._make_BasicBlock(64, 128, Conv2d_kernel[2], filter = SFM_filters[2], bias=0.01),
                self._make_ConvBlock(128, 256, Conv2d_kernel[3], bias=0.01)
            ) for i in range(in_channels)
        ])

        self.fc1 = nn.Sequential(
            nn.Linear(3 * 256 * 4 * 4, out_channels)
        )

    def _make_BasicBlock(self,
                    in_channels:int, 
                    out_channels:int, 
                    kernel_size:tuple,
                    stride:int = 1,
                    padding:int = 0,
                    filter:tuple = (1,1),
                    w = 4.0,
                    bias = 0.4,):
        return nn.Sequential(
            RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, padding = padding, device = "cuda"),
            triangle(w=w, requires_grad = True, device = "cuda"),
            cReLU(bias=bias, requires_grad = True, device = "cuda"),
            SFM(filter = filter, device = "cuda")
        )

    def _make_ConvBlock(self,
                    in_channels, 
                    out_channels, 
                    kernel_size,
                    stride:int = 1,
                    padding:int = 0,
                    w = 4.0,
                    bias = 0.4,):
        return nn.Sequential(
            RBF_Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride = stride, device = "cuda"),
            triangle(w=w, requires_grad = True, device="cuda"),
            cReLU(bias=bias, requires_grad = True, device="cuda"),
        )

    def forward(self, x):
        fc_input = []
        for i, l in enumerate(self.convs):
            fc_input.append(l(x[:, i, :, :][:, None, :, :]))
        output = torch.concat((fc_input), dim=1)
        output = self.fc1(output.reshape(x.shape[0], -1))
        return output


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
                 device=None,
                 dtype=None):
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding
        factory_kwargs = {'device':device, 'dtype':dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.weight = torch.empty((out_channels, in_channels * self.kernel_size[0] * self.kernel_size[1]), **factory_kwargs)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight)
        self.weight = nn.Parameter(self.weight)
    
    def forward(self, input: Tensor) -> Tensor:
        output_width = math.floor((input.shape[-1] + self.padding * 2 - (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1)
        output_height = math.floor((input.shape[-2] + self.padding * 2 - (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1)
        windows = F.unfold(input, kernel_size = self.kernel_size, stride = self.stride, padding = self.padding).permute(0, 2, 1)
        result = torch.pow(torch.cdist(windows, self.weight).permute(0, 2, 1), 2)
        return result.reshape(result.shape[0], result.shape[1], output_height, output_width)
    
    def extra_repr(self) -> str:
        return f"weight shape = {(self.out_channels, self.in_channels, *self.kernel_size)}"

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
        d[d>w_tmp] = w_tmp
        return torch.ones_like(d) - torch.div(d, w_tmp)
    
    def extra_repr(self) -> str:
        return f"w = {self.w.item()}"

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
        return x * torch.ge(x, bias_tmp).float()
    
    def extra_repr(self) -> str:
        return f"bias={self.bias.item()}"

'''
    時序合併層
    parameters:
        filter: 合併的範圍
'''
class SFM(nn.Module):
    def __init__(self,
                 filter: _size_2_t,
                 alpha: float = 0.9,
                 device: str = "cuda") -> None:
        super(SFM, self).__init__()
        self.filter = filter
        self.alpha = torch.nn.Parameter(torch.tensor([alpha]).to(device), requires_grad=True)
        "powNum = 次方數順序"
        self.powerNum = torch.arange(math.prod(filter)).flip(0).to(device)
        self.device = device

    def forward(self, input: Tensor) -> Tensor:
        #將 alpha 進行次方計算
        alpha_pows = torch.pow(self.alpha, self.powerNum).reshape(*self.filter)
        #將alpha 從(1,6)變成(100, 1, 6)
        alpha_pows = alpha_pows.repeat(input.shape[1], 1, 1).to(self.device)

        batch_num, channels, height, width = input.shape
        _, filter_h, filter_w = alpha_pows.shape

        # 使用 unfold 將 input 展開成形狀為 (batch_num, channels, (height-filter_h+step)*(width-filter_w+step), filter_h * filter_w) 的二維張量
        unfolded_input = input.unfold(2, filter_h, filter_h).unfold(3, filter_w, filter_h).reshape(batch_num, channels, -1, filter_h * filter_w)
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
        output = result.mean(dim=-1).reshape(batch_num, channels, math.floor(height/filter_h), math.floor(width/filter_w))
        # print(f"output = {output.shape}")
        return output
    
    def extra_repr(self) -> str:
        return f"filter={self.filter}, alpha={self.alpha}"



        

    