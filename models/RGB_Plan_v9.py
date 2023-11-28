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
from utils import get_rbf, get_RM

class SOMNetwork(nn.Module):
    def __init__(self, input_shape, out_channels, stride)->None:
        super().__init__()
        # _log_api_usage_once(self)

        self.in_channels = input_shape[0]
        self.out_channels = out_channels

        self.SFM_combine_filters = [(2, 2),  (1, 3), (3, 1), (1, 1)]
        # self.SFM_combine_filters = [(1, 6), (3, 1), (2, 1), (1, 1)]
        self.Conv2d_kernel = [(5, 5), (10, 10), (15, 15), (25, 25), (35, 35)]

        # self.SFM_combine_filters = [(2, 2),  (1, 5), (5, 1), (1, 1)]
        # self.Conv2d_kernel = [(3, 3), (5, 5), (7, 7), (9, 9), (35, 35)]

        #shape = input經過Conv後的width和height
        self.patch_shapes = [(math.floor((input_shape[1] - self.Conv2d_kernel[0][0])//stride) + 1, math.floor((input_shape[2] - self.Conv2d_kernel[0][1])//stride) + 1)]
        for i in range(3):
            self.patch_shapes.append((int(self.patch_shapes[i][0] / self.SFM_combine_filters[i][0]), int(self.patch_shapes[i][1] / self.SFM_combine_filters[i][1])))

        self.convs = nn.ModuleList([
            nn.Sequential(
                RBF_Conv2d(1, math.prod(self.Conv2d_kernel[1]), kernel_size=self.Conv2d_kernel[0], stride=stride),
                cReLU(0.4),
                SFM(kernel_size=self.Conv2d_kernel[1], shape=self.patch_shapes[0], filter=self.SFM_combine_filters[0]),
                RBF_Conv2d(1, math.prod(self.Conv2d_kernel[2]), kernel_size=self.Conv2d_kernel[1], stride=stride),
                cReLU(0.1),
                SFM(kernel_size=self.Conv2d_kernel[2], shape=self.patch_shapes[1], filter=self.SFM_combine_filters[1]),
                RBF_Conv2d(1, math.prod(self.Conv2d_kernel[3]), kernel_size=self.Conv2d_kernel[2], stride=stride),
                cReLU(0.01),
                SFM(kernel_size=self.Conv2d_kernel[3], shape=self.patch_shapes[2], filter=self.SFM_combine_filters[2]),
                RBF_Conv2d(1, math.prod(self.Conv2d_kernel[4]), kernel_size=self.Conv2d_kernel[3], stride=stride),
                cReLU(0.01),
            ) for i in range(self.in_channels)
        ])

        self.fc1 = nn.Sequential(
            nn.Linear(3 * 1225, self.out_channels)
        )

    def forward(self, x):
        fc_input = []
        for i, l in enumerate(self.convs):
            fc_input.append(l(x[:, i, :, :][:, None, :, :]))
        output = torch.concat((fc_input), dim=1)
        output = self.fc1(output.reshape(x.shape[0], -1))
        return output
    
    def get_FM_img(self):
        FMs=[]
        for conv in self.convs:
            tmp = []
            for i in range(len(self.Conv2d_kernel)):
                tmp.append(conv[i * 3].weight.permute(0, 2, 3, 1).reshape(*self.Conv2d_kernel[i+1], *self.Conv2d_kernel[i], 1))
            FMs.append(tmp)
        return FMs

    def get_RM_img(self,X):
        RMs=[]
        for i, conv in enumerate(self.convs):
            tmp = []
            tmp.append(get_RM(conv[0](x[:, i, :, :][:, None, :, :]), (6, 6, 10, 10, 1)))
            tmp.append(torch.nn.Sequential(conv[:4])(x[:, i, :, :][:, None, :, :], (3, 3, 15, 15, 1)))
            tmp.append(torch.nn.Sequential(conv[:7])(x[:, i, :, :][:, None, :, :]), (3, 1, 25, 25, 1))
            tmp.append(torch.nn.Sequential(conv[:10])(x[:, i, :, :][:, None, :, :]), (1, 1, 35, 35, 1))
            RMs.append(tmp)
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
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]]
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

class triangle(nn.Module):
    def __init__(self, w:float = 16) -> None:
        super().__init__()
        self.w = w

    def forward(self, d):
        d[d>self.w] = self.w
        return torch.ones_like(d) - torch.div(d, self.w)
    
    def extra_repr(self) -> str:
        return f"w = {self.w}"

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
        self.patch_shapes = _pair(shape)
        self.alpha_0 = torch.tensor(alpha)
        self.filter = filter
        self.alpha_type = alpha_type
        self.sequence = torch.nn.Parameter(torch.arange(math.prod(filter)).flip(0), requires_grad=False)
        
    
    def __filtering(self, input: Tensor):
        input = get_RM(input, (*self.patch_shapes, math.prod(self.kernel_size)))
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
        return f"kernel_size={self.kernel_size}, shape={self.patch_shapes}, filter={self.filter}, alpha={self.alpha_0}"
