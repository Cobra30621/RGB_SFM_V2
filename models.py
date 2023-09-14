import os
import math
import torch
import threading
import torchvision
import torchvision.models as models
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import init
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.transforms import Grayscale
from operator import truediv
from utils import _pair, get_rbf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
root = os.path.dirname(__file__)

'''
    RM時序合併層
'''
class SFM(Module):
    
    def __init__(self, kernel_size: _size_2_t, shape: _size_2_t, filter: _size_2_t, alpha: float = 0.9, alpha_type: str = 'pow') -> None:
        super(SFM, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.shape = _pair(shape)
        self.alpha_0 = torch.tensor(alpha)
        self.filter = filter
        self.alpha_type = alpha_type
        self.sequence = torch.nn.Parameter(torch.arange(math.prod(filter)).flip(0), requires_grad=False)
        
    
    def __filtering(self, input: Tensor):
        input = input.permute(0, 2, 3, 1).reshape(-1, *self.shape, math.prod(self.kernel_size))
        n_data, seg_h, seg_w, n_filters = input.shape
        
        RM_segment_size = list(map(int, map(truediv, (seg_h, seg_w), self.filter)))
        # [a^n, a^(n-1), ..., a^0] 
        alpha = torch.pow(self.alpha_0, self.sequence)
        alpha = alpha.reshape(self.filter).repeat(*[int(i / j) for i, j in zip((seg_h, seg_w), self.filter)])
        alpha = alpha[:, :, None]
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
    論文model本體
'''
class SOMNetwork(nn.Module):
    def __init__(self, 
                 stride: int = 1, 
                 in_channels: int = 1, 
                 out_channels: int = 10,
                 kernel_size: _size_2_t = (5, 5), 
                 input_size: _size_2_t = (28, 28),
                 rbf: str = 'gauss',
                 kernels: list = [(10, 10), (15, 15), (25, 25), (35, 35)],
                 filters:list = None,):
        super().__init__()
        
        height = (input_size[0] - kernel_size[0]) // stride + 1
        width = (input_size[1] - kernel_size[1]) // stride + 1
        if filters is None:
            if stride == 1:
                filters = [(1, width), (6, 1), (4, 1), (1, 1)] # filters that stride = 1
            elif stride == 2:
                filters = [(1, width), (4, 1), (3, 1), (1, 1)] # filters that stride = 2
            elif stride == 3:
                filters = [(1, width), (4, 1), (2, 1), (1, 1)] # filters that stride = 3
            elif stride == 4:
                # filters = [(1, width), (3, 1), (2, 1), (1, 1)] # filters that stride = 4 28 16312111
                filters = [(2, 2), (1, 3), (3, 1), (1, 1)] # filters that stride = 4 28 22133111
        
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        shapes = [(height, width)]
        for i in range(3):
            shapes.append((int(shapes[i][0] / filters[i][0]), int(shapes[i][1] / filters[i][1])))
            
        self.layer1 = nn.Sequential(         
            RBFConv2d(self.in_channels, math.prod(kernels[0]), kernel_size, stride=stride, rbf=rbf, color_init=True if self.in_channels == 3 else False),
            cReLU(0.4),
            SFM(kernel_size=kernels[0], shape=shapes[0], filter=filters[0])
        )
        
        self.layer2 = nn.Sequential(         
            RBFConv2d(1, math.prod(kernels[1]), kernels[0], stride=1, rbf=rbf),      
            cReLU(0.1),                        
            SFM(kernel_size=kernels[1], shape=shapes[1], filter=filters[1])
        )
        
        self.layer3 = nn.Sequential(         
            RBFConv2d(1, math.prod(kernels[2]), kernels[1], stride=1, rbf=rbf),   
            cReLU(0.01),                           
            SFM(kernel_size=kernels[2], shape=shapes[2], filter=filters[2])
        )
            
        self.layer4 = nn.Sequential(         
            RBFConv2d(1, math.prod(kernels[3]), kernels[2], stride=1, rbf=rbf),   
            cReLU(0.01),
        )

        self.fc1 = nn.Linear(kernels[-1][0]*kernels[-1][1], self.out_channels)
        self.softmax = nn.Softmax(dim=-1)
            
    def forward(self, x):
        out: Tensor
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc1(out.reshape(len(x), -1))
        out = self.softmax(out)
        return out
    

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
    高斯卷積層
''' 
class RBFConv2d(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 std: float = 2.0,
                 rbf: str = 'gauss',
                 color_init: bool = False,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.std = torch.nn.Parameter(torch.tensor(std))
        # self.std = torch.nn.Parameter(torch.full((out_channels, 1), std))
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.color = color_init
        self.rbf = get_rbf(rbf)
        
        self.weight = torch.empty((out_channels, 1, *self.kernel_size), **factory_kwargs) 
        self.register_parameter('bias', None)
        self.reset_parameters()
        if color_init:
            self.color_init()
 
 
    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
        self.weight[0] = torch.zeros_like(self.weight[0])
        self.weight = nn.Parameter(self.weight)

        
    '''
        初始化原SFMNetwork前面的filter
    '''
    def color_init(self) -> None:
        max_color = 1
        min_color = 0
        
        self.rgb_out_channels = 19
        #由19種顏色之filter來找出區塊對應之顏色(每個filter代表一個顏色)
        self.rgb_weight = torch.linspace(max_color, min_color, int((max_color - min_color) / (max_color / self.rgb_out_channels)))[:, None].repeat(1, 3)
        self.rgb_weight = nn.Parameter(self.rgb_weight)
        
        #初始化灰階前面的filter
        self.gray_weight = torch.empty((self.out_channels - self.rgb_out_channels, 1, *self.kernel_size)) 
        init.kaiming_uniform_(self.gray_weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.gray_weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

        self.gray_weight = nn.Parameter(self.gray_weight)
    
             
    def _gray_forward(self, input: Tensor, weight: Tensor, std, stride) -> Tensor:        
        output_height = torch.div((input.shape[2] - self.kernel_size[0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((input.shape[3] - self.kernel_size[1]),  stride, rounding_mode='floor') + 1
        result = torch.zeros((input.shape[0], self.out_channels, output_height, output_width)).to(input.device)
        for k in range(output_height):
            for l in range(output_width):   
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]]
                dist = torch.cdist(window.reshape(input.shape[0], -1), weight.reshape(-1, self.in_channels*math.prod(self.kernel_size)))
                result[:, :, k, l] = self.rbf(dist, std)
        return result


    def conv(self, result, input, weight, stride):
        batch_size = input.shape[0]
        for k in range(result.shape[2]):
            for l in range(result.shape[3]):   
                window = input[:, :, k*stride:k*stride+self.kernel_size[0], l*stride:l*stride+self.kernel_size[1]]
                dist = torch.zeros((batch_size, weight.shape[0])).to(input.device)
                for in_channel in range(input.shape[1]):
                    dist += torch.cdist(window[:, in_channel].reshape(batch_size, -1), weight[:, in_channel].reshape(weight.shape[0], -1))
                result[:, :, k, l] = self.rbf(dist, self.std)

    # RGB 前處理
    def _rgb_forward(self, input: Tensor, gray_weight: Tensor, rgb_weight: Tensor, std, stride) -> Tensor:        
        batch_size = input.shape[0]
        output_height = torch.div((input.shape[2] - self.kernel_size[0]),  stride, rounding_mode='floor') + 1
        output_width = torch.div((input.shape[3] - self.kernel_size[1]),  stride, rounding_mode='floor') + 1
        gray_result = torch.zeros((batch_size, self.out_channels - self.rgb_out_channels, output_height, output_width)).to(input.device)
        rgb_result = torch.zeros((batch_size, self.rgb_out_channels, output_height, output_width)).to(input.device)
        
        if gray_weight.shape[0] == 0:
            self.conv(rgb_result, input, rgb_weight, stride)
            return rgb_result
        elif rgb_weight.shape[0] == 0:
            self.conv(gray_result, Grayscale()(input), gray_weight, stride)
            return gray_result
        else:
            # RGB forward + gray forward
            t = threading.Thread(target = self.conv, args = (gray_result, Grayscale()(input), gray_weight, stride,))
            t.start()
            self.conv(rgb_result, input, rgb_weight, stride)
            t.join()
            # result合併
            return torch.concat((gray_result, rgb_result), dim=1)
    
    
    def forward(self, input: Tensor) -> Tensor:
        if self.color:
            rgb_weight = torch.repeat_interleave(torch.repeat_interleave(self.rgb_weight.reshape(self.rgb_out_channels, self.in_channels, 1, 1), self.kernel_size[0], dim=2), self.kernel_size[1], dim=3)
            return self._rgb_forward(input, self.gray_weight, rgb_weight, self.std, torch.tensor(self.stride[0]))
        return self._gray_forward(input, self.weight, self.std, torch.tensor(self.stride[0]))

        
    def extra_repr(self) -> str:
        return f"std={self.std}, weight shape={self.weight.shape}, kernel size={self.kernel_size}"
    
    
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()        
        self.fc1 = nn.Linear(28*28, 10) 
        self.softmax = nn.Softmax(-1) 
    def forward(self, x):
        x = x.view(x.size(0), -1)       
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    
class CNN(nn.Module):
    def __init__(self, in_channels: int = 1):
        super(CNN, self).__init__()        
        self.conv1 = nn.Sequential(         
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=5, stride=4, padding=3),                              
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=3), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1), 
            nn.ReLU(), 
        )
        self.fc1 = nn.Linear(128, 2) 
        self.softmax = nn.Softmax(-1) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)       
        x = self.fc1(x)
        x = self.softmax(x)
        return x
    
    
class ResNet(nn.Module):
    def __init__(self, layers=18):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        if layers == 34:
            self.model = models.resnet34(pretrained=True)
        elif layers == 50:
            self.model = models.resnet50(pretrained=True)
        elif layers == 101:
            self.model = models.resnet101(pretrained=True)
        elif layers == 152:
            self.model = models.resnet152(pretrained=True)
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=4, padding=2, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    def forward(self, x):
        return self.model(x)
    
    
class AlexNet(nn.Module):   
    def __init__(self, num=10):
        super(AlexNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(32*5*5,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num),
        )
    def forward(self, x):
        x = self.feature(x)
        x = x.view(-1,32*5*5)
        x = self.classifier(x)
        return x
    

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
    
class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.model = models.GoogLeNet(init_weights=True)
        self.model.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 10)
    def forward(self, x):
        if type(self.model(x)) == torchvision.models.GoogLeNetOutputs:
            return self.model(x).logits
        return self.model(x)