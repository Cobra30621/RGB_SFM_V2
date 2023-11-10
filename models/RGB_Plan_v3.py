import math
import torch
import torchvision
from torch import nn
from torch import Tensor
from torch.nn import init
from torch.nn.common_types import _size_2_t
from torchvision.transforms import Grayscale
from torch.nn.modules.utils import _pair
from operator import truediv
from utils import *

class SOMNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, stride)->None:
        super().__init__()
        # _log_api_usage_once(self)

        self.in_channels = in_channels
        self.out_channels = out_channels

        SFM_combine_filters = [(2, 2), (1, 3), (3, 1), (1, 1)]
        # SFM_combine_filters = [(1, 6), (3, 1), (2, 1), (1, 1)]
        Conv2d_kernel = [(5, 5), (10, 10), (15, 15), (25, 25), (35, 35)]

        self.shape = [(int((28 - 5 + 1)//stride), int((28 - 5 + 1)//stride))]
        for i in range(3):
            self.shape.append((int(self.shape[i][0] / SFM_combine_filters[i][0]), int(self.shape[i][1] / SFM_combine_filters[i][1])))

        self.RGB_preprocess = nn.Sequential(
            RGB_Conv2d(3, 100, kernel_size=Conv2d_kernel[0], stride=stride),
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
            SFM(kernel_size=Conv2d_kernel[1], shape=self.shape[0], filter=SFM_combine_filters[0]),
        ]
        self.layer1 = nn.Sequential(*self.layer1)

        self.layer2 = nn.Sequential(
            RBF_Conv2d(1, math.prod(Conv2d_kernel[2]), kernel_size=Conv2d_kernel[1], stride=stride),
            cReLU(0.1),
            SFM(kernel_size=Conv2d_kernel[2], shape=self.shape[1], filter=SFM_combine_filters[1]),
            RBF_Conv2d(1, math.prod(Conv2d_kernel[3]), kernel_size=Conv2d_kernel[2], stride=stride),
            cReLU(0.01),
            SFM(kernel_size=Conv2d_kernel[3], shape=self.shape[2], filter=SFM_combine_filters[2]),
            RBF_Conv2d(1, math.prod(Conv2d_kernel[4]), kernel_size=Conv2d_kernel[3], stride=stride),
            cReLU(0.01),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1225, self.out_channels)
        )

    def forward(self, x):
        out: Tensor
        if self.in_channels == 3:
            RGB_output = self.RGB_preprocess(x)
            input = RGB_output

        output = self.layer1(input)
        output = self.layer2(output)
        output = self.fc1(output.reshape(x.shape[0], -1))
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
        
        self.weight = torch.empty((out_channels, 1, *self.kernel_size), **factory_kwargs) 
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight[0] = torch.zeros_like(self.weight[0])
        self.weight = nn.Parameter(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        stride = torch.tensor(self.stride[0])
        batch_size = input.shape[0]
        output_height = torch.div((input.shape[2] - self.kernel_size[0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((input.shape[3] - self.kernel_size[1]),  stride, rounding_mode='floor') + 1
        result = torch.zeros((batch_size, self.out_channels, output_height, output_width)).to(input.device)

        for k in range(result.shape[2]):
            for l in range(result.shape[3]):  
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]]
                dist = torch.zeros((input.shape[0], self.weight.shape[0]), device = input.device)
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
        
        self.rgb_weight = torch.empty((out_channels, 3, *self.kernel_size), **factory_kwargs)
        self.reset_parameters()
 
 
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.rgb_weight, a=math.sqrt(5))
        self.rgb_weight[0] = torch.zeros_like(self.rgb_weight[0])
        self.rgb_weight = nn.Parameter(self.rgb_weight)
    
    
    def forward(self, input: Tensor) -> Tensor:
        #計算output shape
        stride = torch.tensor(self.stride[0])
        batch_size = input.shape[0]
        output_height = torch.div((input.shape[2] - self.kernel_size[0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((input.shape[3] - self.kernel_size[1]),  stride, rounding_mode='floor') + 1
        rgb_result = torch.zeros((batch_size, self.out_channels, output_height, output_width), device = input.device)
        
        # RBF
        for k in range(rgb_result.shape[2]):
            for l in range(rgb_result.shape[3]):   
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]]
                dist = torch.zeros((input.shape[0], self.rgb_weight.shape[0]), device = input.device)
                for in_channel in range(input.shape[1]):
                    dist += torch.cdist(window[:, in_channel].reshape(batch_size, -1), self.rgb_weight[:, in_channel].reshape(self.rgb_weight.shape[0], -1))
                self.std = torch.std(dist)
                rgb_result[:, :, k, l] = self.rbf(dist, self.std)
        return rgb_result

        
    def extra_repr(self) -> str:
        return f"weight shape={self.rgb_weight.shape}, kernel size={self.kernel_size}"

'''
    將兩個input合併用RBF函數計算之層
'''
class Combine_Conv2d(nn.Module):
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
        self.std = torch.nn.Parameter(torch.tensor(std))
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.rbf = get_rbf(rbf)
        
        self.gray_weight = torch.empty((out_channels, 1, *self.kernel_size[0]), **factory_kwargs)
        self.rgb_weight = torch.empty((out_channels, 1, *self.kernel_size[1]), **factory_kwargs) 
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.gray_weight, a=math.sqrt(5))
        self.gray_weight[0] = torch.zeros_like(self.gray_weight[0])
        self.gray_weight = nn.Parameter(self.gray_weight)

        init.kaiming_uniform_(self.rgb_weight, a=math.sqrt(5))
        self.rgb_weight[0] = torch.zeros_like(self.rgb_weight[0])
        self.rgb_weight = nn.Parameter(self.rgb_weight)

    def forward(self, gray_input: Tensor, rgb_input: Tensor) -> Tensor:
        stride = torch.tensor(self.stride[0])
        batch_size = gray_input.shape[0]
        output_height = torch.div((gray_input.shape[2] - self.kernel_size[0][0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((gray_input.shape[3] - self.kernel_size[0][1]),  stride, rounding_mode='floor') + 1
        result = torch.zeros((batch_size, self.out_channels, output_height, output_width), device = "cuda")

        for k in range(result.shape[2]):
            for l in range(result.shape[3]):
                gray_window = gray_input[:, :, k*stride:k*stride+self.kernel_size[0][0], l*stride:l*stride+self.kernel_size[0][1]]
                gray_dist = torch.cdist(gray_window.reshape(batch_size, -1), self.gray_weight.reshape(-1, self.in_channels*math.prod(self.kernel_size[0])))    

                rgb_window = rgb_input[:, :, k*stride:k*stride+self.kernel_size[1][0], l*stride:l*stride+self.kernel_size[1][1]]
                rgb_dist = torch.cdist(rgb_window.reshape(batch_size, -1), self.rgb_weight.reshape(-1, self.in_channels*math.prod(self.kernel_size[1])))
                
                dist = gray_dist + rgb_dist

                result[:, :, k, l] = self.rbf(dist, self.std)
        return result

    def extra_repr(self) -> str:
        return f"std={self.std}, gray_weight shape={self.gray_weight.shape}, rgb_weight shape={self.rgb_weight.shape}, kernel size={self.kernel_size}"

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
    def __init__(self, kernel_size: _size_2_t, shape: _size_2_t, filter: _size_2_t, alpha: float = 0.9, alpha_type: str = 'pow') -> None:
        super(SFM, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.shape = _pair(shape)
        self.alpha_0 = torch.tensor(alpha)
        self.filter = filter
        self.alpha_type = alpha_type
        self.sequence = torch.nn.Parameter(torch.arange(math.prod(filter)).flip(0), requires_grad=False)
        
    
    def __filtering(self, input: Tensor):
        input = get_RM(input, (*self.shape, math.prod(self.kernel_size)))
        n_data, seg_h, seg_w, n_filters = input.shape
        
        RM_segment_size = list(map(int, map(truediv, (seg_h, seg_w), self.filter)))
        # [a^n, a^(n-1), ..., a^0] 
        alpha = torch.pow(self.alpha_0, self.sequence)
        alpha = alpha.reshape(self.filter).repeat(*[int(i / j) for i, j in zip((seg_h, seg_w), self.filter)])
        alpha = alpha[:, :, None].to('cuda')
        self.alpha = alpha

        out = torch.mul(input, alpha).reshape(n_data, RM_segment_size[0] ,self.filter[0], RM_segment_size[1], self.filter[1], n_filters)
        out = out.mean(2).mean(3)
        return out.reshape(-1, 1, *self.kernel_size)

    
    def forward(self, input: Tensor) -> Tensor:
        output = self.__filtering(input)
        return output
    
    
    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, shape={self.shape}, filter={self.filter}, alpha={self.alpha_0}"

'''
    印出FM、RM、CI的圖形
'''
class Visualize:
    def __init__(self, model:nn.Module):
        self.model = model

    def get_FM_img(self):
        FMs={}
        FMs['rgb'] = self.model.RGB_preprocess[0].rgb_weight.reshape(10, 10, 3, 5, 5).permute(0, 1, 3, 4, 2)
        FMs[1] = self.model.layer2[0].weight.permute(0, 2, 3, 1).reshape(15, 15, 10, 10, 1)
        FMs[2] = self.model.layer2[3].weight.permute(0, 2, 3, 1).reshape(25, 25, 15, 15, 1)
        FMs[3] = self.model.layer2[6].weight.permute(0, 2, 3, 1).reshape(35, 35, 25, 25, 1)
        return FMs

    def get_RM_img(self, X):
        input = self.model.RGB_preprocess(X)

        RMs={}
        RMs['rgb'] = get_RM(input[:, :, :, :], (6, 6, 10, 10, 1))
        RMs[0] = get_RM(input[:, :, :, :], (6, 6, 10, 10, 1))
        RMs[1] = get_RM(torch.nn.Sequential(self.model.layer1 + self.model.layer2[0:2])(input), (3, 3, 15, 15, 1))
        RMs[2] = get_RM(torch.nn.Sequential(self.model.layer1 + self.model.layer2[0:5])(input), (3, 1, 25, 25, 1))
        RMs[3] = get_RM(torch.nn.Sequential(self.model.layer1 + self.model.layer2)(input), (1, 1, 35, 35, 1))
        return RMs

    def get_CI_img(self, X):
        input = self.model.RGB_preprocess(X)

        CIs = {}
        pred = torch.nn.Sequential(*(list(self.model.layer1)+list(self.model.layer2[:1])))(input)
        CIs[1] = get_ci(X, pred, sfm_filter=self.model.layer1[0].filter, n_filters = self.model.layer2[0].weight.shape[0])
        pred = torch.nn.Sequential(*(list(self.model.layer1)+list(self.model.layer2[:4])))(input)
        CIs[2] = get_ci(X, pred, sfm_filter=tuple(np.multiply(self.model.layer1[0].filter, self.model.layer2[2].filter)), n_filters = self.model.layer2[3].weight.shape[0])
        pred = torch.nn.Sequential(*(list(self.model.layer1)+list(self.model.layer2[:7])))(input)
        CIs[3] = get_ci(X, pred, sfm_filter=tuple(np.multiply(np.multiply(self.model.layer1[0].filter, self.model.layer2[2].filter), self.model.layer2[5].filter)), n_filters = self.model.layer2[6].weight.shape[0])
        return CIs

    def save_RM_CI(self, X, filter, RMs, CIs, RM_save_dir):
        if len(X[filter]) != 0: 
            Path(RM_save_dir).mkdir(parents=True, exist_ok=True)
            plt.clf()
            plt.imshow(X[filter][0].permute(1, 2, 0).detach().cpu().numpy())
            plt.axis('off')
            plt.savefig(RM_save_dir + '/input.png', bbox_inches='tight')

            segments = split(X)
            plot_map(segments[filter][0].permute(1, 2, 3, 4, 0).detach().cpu().numpy(), path = RM_save_dir + '/input_segements.png')

            plt.clf()
            plt.imshow(Grayscale()(X)[filter][0].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(RM_save_dir + '/input_Gray.png', bbox_inches='tight')

            segments = split(Grayscale()(X))
            plot_map(segments[filter][0].permute(1, 2, 3, 4, 0).detach().cpu().numpy(), path = RM_save_dir + '/input_Gray_segements.png')
            
            for key in RMs:
                print(f'{RM_save_dir} \t RMs[{key}] saving\t{RMs[key][filter][0].shape}')
                plot_map(RMs[key][filter][0].detach().cpu().numpy(), path = RM_save_dir + f'/RMs_{key}.png')
            
            for key in CIs:
                _, tmp = torch.topk(RMs[key][filter][0].reshape(RMs[key][filter][0].shape[0] * RMs[key][filter][0].shape[1], -1), k=5, dim=1)
                for i in range(tmp.shape[1]):
                    print(f'{RM_save_dir} \t CIs[{key}][{i}] saving\t{CIs[key][tmp[:,i][:, None].cpu()].reshape(*self.model.shape[key], CIs[key].shape[-3], CIs[key].shape[-2], CIs[key].shape[-1]).permute(0, 1, 3, 4, 2).shape}')
                    plot_map(CIs[key][tmp[:,i][:, None].cpu()].reshape(*self.model.shape[key], CIs[key].shape[-3], CIs[key].shape[-2], CIs[key].shape[-1]).permute(0, 1, 3, 4, 2), path = RM_save_dir + f'/CIs_{key}_{i}.png')
        else:
            print(f"This batch don't have label {y[filter]}")