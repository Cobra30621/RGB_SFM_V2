import math
import torch
import torchvision
import time
from torch import nn
from torch import Tensor
from torch.nn import init
from torch.nn.common_types import _size_2_t
from torchvision.transforms import Grayscale
from torch.nn.modules.utils import _pair
from operator import truediv
from utils import get_rbf, get_RM

class SOMNetwork(nn.Module):
    def __init__(self, in_channels, out_channels)->None:
        super().__init__()
        # _log_api_usage_once(self)

        self.in_channels = in_channels
        self.out_channels = out_channels

        stride = 4
        SFM_combine_filters = [(2, 2),  (1, 5), (5, 1), (1, 1)]
        # SFM_combine_filters = [(1, 6), (3, 1), (2, 1), (1, 1)]
        Conv2d_kernel = [(5, 5), (10, 10), (15, 15), (25, 25), (35, 35)]

        self.shape = [(int((64 - 5 + 1)//stride), int((64 - 5 + 1)//stride))]
        for i in range(3):
            self.shape.append((int(self.shape[i][0] / SFM_combine_filters[i][0]), int(self.shape[i][1] / SFM_combine_filters[i][1])))

        self.RGB_preprocess = nn.Sequential(
            RGB_Conv2d(3, 36, kernel_size=Conv2d_kernel[0], stride=stride),
            cReLU(0.4)
        )
        self.GRAY_preprocess = nn.Sequential(
            RBF_Conv2d(1, 10*10 - 36, kernel_size=Conv2d_kernel[0], stride=stride),
            cReLU(0.4)
        )

        if self.in_channels == 1:
            self.layer1 = [
                RBF_Conv2d(in_channels, math.prod(Conv2d_kernel[1]), kernel_size=Conv2d_kernel[0], stride=stride),
                cReLU(0.4),
            ]
        else:
            self.layer1 = []
        self.layer1 += [
            # SFM(kernel_size=Conv2d_kernel[1], shape=self.shape[0], filter=SFM_combine_filters[0]),
        ]
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = nn.Sequential(
            RBF_Conv2d(1, math.prod(Conv2d_kernel[2]), kernel_size=Conv2d_kernel[1], stride=stride),
            cReLU(0.1),
            # SFM(kernel_size=Conv2d_kernel[2], shape=self.shape[1], filter=SFM_combine_filters[1]),
        )

        self.layer3 = nn.Sequential(
            RBF_Conv2d(1, math.prod(Conv2d_kernel[3]), kernel_size=Conv2d_kernel[2], stride=stride),
            cReLU(0.01),
            SFM(kernel_size=Conv2d_kernel[3], shape=self.shape[2], filter=SFM_combine_filters[2]),
        )

        self.layer4 = nn.Sequential(
            RBF_Conv2d(1, math.prod(Conv2d_kernel[4]), kernel_size=Conv2d_kernel[3], stride=stride),
            cReLU(0.01),
            # SFM(kernel_size=Conv2d_kernel[4], shape=self.shape[3], filter=SFM_combine_filters[3]),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(24500, 4096),
            nn.Linear(4096, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, self.out_channels)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out: Tensor
        RGB_output = self.RGB_preprocess(x)
        GRAY_output = self.GRAY_preprocess(Grayscale()(x))
        input = torch.concat((RGB_output, GRAY_output), dim=1)

        input = input.permute(0, 2, 3, 1).reshape(-1, 1, 10, 10)
        output = self.layer1(input)
        output = self.layer2(output)
        output = output.permute(0, 2, 3, 1).reshape(-1, 1, 15, 15)
        output = self.layer3(output)
        output = output.permute(0, 2, 3, 1).reshape(-1, 1, 25, 25)
        output = self.layer4(output)
        output = self.fc1(output.reshape(x.shape[0], -1))
        output = self.sigmoid(output)
        return output

'''
    RBF 卷積層
'''
class RBF_Conv2d(nn.Module):
    def __init__(self, 
                 in_channels:int, 
                 out_channels:int, 
                 kernel_size:_size_2_t, 
                 stride:int, 
                 rbf:str = "gauss",
                 std:float = 2.0,
                 device="cuda",
                 dtype=None) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.std = torch.tensor(std)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.rbf = get_rbf(rbf)
        self.device = device
        
        self.weight = torch.empty((out_channels, 1, *self.kernel_size), **factory_kwargs) 
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight[0] = torch.zeros_like(self.weight[0], device = self.device)
        self.weight = nn.Parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        stride = torch.tensor(self.stride[0])
        batch_size = input.shape[0]
        output_height = torch.div((input.shape[2] - self.kernel_size[0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((input.shape[3] - self.kernel_size[1]),  stride, rounding_mode='floor') + 1
        result = torch.zeros((batch_size, self.out_channels, output_height, output_width), device=input.device)

        for k in range(result.shape[2]):
            for l in range(result.shape[3]):  
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]]
                dist = torch.zeros((input.shape[0], self.weight.shape[0]), device=input.device)
                for in_channel in range(input.shape[1]):
                    dist += torch.cdist(window[:, in_channel].reshape(batch_size, -1), self.weight.reshape(self.weight.shape[0], -1))
                self.std = torch.std(dist)      
                result[:, :, k, l] = self.rbf(dist, self.std)
        return result

    def extra_repr(self) -> str:
        return f"std={self.std}, weight shape={self.weight.shape}, kernel size={self.kernel_size}"

'''
    RGB合併_RBF層
'''
class RGB_Conv2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 std: float = 2.0,
                 rbf: str = 'gauss',
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.std = torch.tensor(std)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.rbf = get_rbf(rbf)
        
        self.rgb_weight = torch.empty((out_channels, 3), **factory_kwargs)
        self.reset_parameters()
 
 
    def reset_parameters(self) -> None:
        #由19種顏色之filter來找出區塊對應之顏色(每個filter代表一個顏色)
        max_color = 1
        min_color = 0
        self.rgb_weight = torch.linspace(max_color, min_color, int((max_color - min_color) / (max_color / self.out_channels)))[:, None].repeat(1, 3).to('cuda')
        self.rgb_weight = nn.Parameter(self.rgb_weight)
    
    
    def forward(self, input: Tensor) -> Tensor:
        #expand rgb_weight
        rgb_weight_expand = torch.repeat_interleave(torch.repeat_interleave(self.rgb_weight.reshape(self.out_channels, self.in_channels, 1, 1), self.kernel_size[0], dim=2), self.kernel_size[1], dim=3)
        #計算output shape
        stride = torch.tensor(self.stride[0])
        batch_size = input.shape[0]
        output_height = torch.div((input.shape[2] - self.kernel_size[0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((input.shape[3] - self.kernel_size[1]),  stride, rounding_mode='floor') + 1
        rgb_result = torch.zeros((batch_size, self.out_channels, output_height, output_width), device = input.device)
        dist = torch.zeros((input.shape[0], rgb_weight_expand.shape[0]), device = input.device)
        
        # RBF
        for k in range(rgb_result.shape[2]):
            for l in range(rgb_result.shape[3]):   
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]].to(input.device)
                dist = torch.zeros((input.shape[0], rgb_weight_expand.shape[0]), device = input.device)
                for in_channel in range(input.shape[1]):
                    dist += torch.cdist(window[:, in_channel].reshape(batch_size, -1), rgb_weight_expand[:, in_channel].reshape(rgb_weight_expand.shape[0], -1))
                self.std = torch.std(dist)
                rgb_result[:, :, k, l] = self.rbf(dist, self.std)
        return rgb_result

        
    def extra_repr(self) -> str:
        return f"weight shape={self.rgb_weight.shape}, kernel size={self.kernel_size}"

class cReLU(nn.Module):
    def __init__(self, bias: float = 0.7, requires_grad: bool = True) -> None:
        super().__init__()
        self.bias = torch.tensor(bias)
        if requires_grad:
            self.bias = torch.nn.Parameter(self.bias)
    
    def forward(self, x):
        return x * torch.ge(x, self.bias).float()
    
    def extra_repr(self) -> str:
        return f"bias={self.bias}"

'''
    RM時序合併層
    parameters:
        kernel_size:下一層卷積的kernel shape
        shape: 這一層的input shape
        filter: 合併的範圍
    
    output.shape 會將所有batch的output concat 在第0維度

'''
class SFM(nn.Module):
    def __init__(self, kernel_size: _size_2_t, shape: _size_2_t, filter: _size_2_t, alpha: float = 0.9, alpha_type: str = 'pow', device=None) -> None:
        super(SFM, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.shape = _pair(shape)
        self.alpha_0 = torch.tensor(alpha)
        self.filter = filter
        self.alpha_type = alpha_type
        self.sequence = torch.nn.Parameter(torch.arange(math.prod(filter)).flip(0), requires_grad=False)
        self.device = device
        
    
    def __filtering(self, input: Tensor):
        input = get_RM(input, (*self.shape, math.prod(self.kernel_size)))
        n_data, seg_h, seg_w, n_filters = input.shape
        
        RM_segment_size = list(map(int, map(truediv, (seg_h, seg_w), self.filter)))
        # [a^n, a^(n-1), ..., a^0] 
        alpha = torch.pow(self.alpha_0, self.sequence)
        alpha = alpha.reshape(self.filter).repeat(*[int(i / j) for i, j in zip((seg_h, seg_w), self.filter)])
        alpha = alpha[:, :, None].to(input.device)
        self.alpha = alpha

        out = torch.mul(input, alpha).reshape(n_data, RM_segment_size[0] ,self.filter[0], RM_segment_size[1], self.filter[1], n_filters)
        out = out.mean(2).mean(3)
        return out.reshape(-1, 1, *self.kernel_size)

    
    def forward(self, input: Tensor) -> Tensor:
        output = self.__filtering(input)
        return output
    
    
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, shape={self.shape}, filter={self.filter}, alpha={self.alpha_0}"