import torch
from utils import increment_path
from pathlib import Path
import shutil
import os
from torch import nn


project = "paper experiment"
name = "adjust_initial_triangle"
group = "12/17"
tags = ["RGB_SFMCNN_V2", "MultiColor_Shapes_Database"]
description = """
MultiColor_Shapes_Database
use new color filters by Kmean
change SFM to 0.9 ~ 0.99
only one "uniform"
"""
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'



arch = {
    "name": 'RGB_SFMCNN_V2',
    "args":{
       "in_channels": 3,
        "out_channels": 9,
        "Conv2d_kernel": [(5, 5), (1, 1), (1, 1)],
        "SFM_filters": [(2, 2), (1, 3)],
        "channels": [[30, 225, 625], [70, 625, 1225]],
        "strides": [4, 1, 1],
        "paddings": [0, 0, 0],
        # conv_method:  "cdist", "dot_product"
        "conv_method" : [["none", "cdist", "cdist"], ["cdist", "cdist", "cdist"]],
        # "uniform" "kaiming"
        "initial": [["none", "uniform", "kaiming"], ["kaiming", "kaiming", "kaiming"]],
        # "initial": [["none", "kaiming", "kaiming"], ["kaiming", "kaiming", "kaiming"]],
        # "initial": [["none", "uniform", "uniform"], ["uniform", "uniform", "uniform"]],
        # "rbfs": [[['gauss', 'cReLU_percent'], ['gauss', 'cReLU_percent'], ['gauss', 'cReLU_percent']],
        #                  [['gauss', 'cReLU_percent'], ['gauss', 'cReLU_percent'], ['gauss', 'cReLU_percent']]],
        "rbfs": [[["triangle", 'cReLU_percent'], ["triangle", 'cReLU_percent'], ["triangle", 'cReLU_percent']],
                 [["triangle", 'cReLU_percent'], ["triangle", 'cReLU_percent'], ["triangle", 'cReLU_percent']]],
        # "activate_params": [[[0.5, 0.3], [0.5, 0.4], [0.5, 0.5]], [[2, 0.3], [2, 0.4], [2, 0.5]]],
        # "activate_params": [[[0.5, 0.3], [2, 0.4], [2, 0.5]], [[2, 0.3], [2, 0.4], [2, 0.5]]],
        "activate_params": [[[1.5, 0.3], [4, 0.4], [6, 0.5]], [[4, 0.3], [6, 0.4], [8, 0.5]]],
        "fc_input": 1850 * 1 * 3,
        "device": device

    }
}

# For Monitors
# get all layers for draw CI, RM, RM_Monitor
def get_layers(model):
    layers = {}
    if arch['args']['in_channels'] == 1:
        layers[0] = nn.Sequential(model.convs[0][:2]).to(device)
        layers[1] = nn.Sequential(*(list(model.convs[0]) + list([model.convs[1][:2]]))).to(device)
        layers[2] = nn.Sequential(*(list(model.convs[:2]) + list([model.convs[2][:2]]))).to(device)
        layers[3] = nn.Sequential(*(list(model.convs[:3]) + list([model.convs[3][:2]]))).to(device)
    else:
        layers['RGB_convs_0_Conv'] = model.RGB_convs[0][0].to(device)  # 只跑卷積
        layers['RGB_convs_0'] = model.RGB_convs[0].to(device)  # 空間合併前
        layers['RGB_convs_0_SFM'] = nn.Sequential(*(list(model.RGB_convs[:2]))).to(device)  # 空間合併後

        layers['RGB_convs_1_Conv'] = nn.Sequential(*(list(model.RGB_convs[:2]) + list([model.RGB_convs[2][0]]))).to(device)  # 只跑卷積
        layers['RGB_convs_1'] = nn.Sequential(*(list(model.RGB_convs[:2]) + list([model.RGB_convs[2][:-1]]))).to(device)  # 空間合併前
        layers['RGB_convs_1_SFM'] = nn.Sequential(*(list(model.RGB_convs[:3]))).to(device)  # 空間合併後

        layers['RGB_convs_2_Conv'] = nn.Sequential(*(list(model.RGB_convs[:3]) + list([model.RGB_convs[3][0]]))).to(device)  # 只跑卷積
        layers['RGB_convs_2'] = nn.Sequential(*(list(model.RGB_convs))).to(device)

        layers['Gray_convs_0_Conv'] = model.Gray_convs[0][0].to(device)  # 只跑卷積
        layers['Gray_convs_0'] = model.Gray_convs[0].to(device)  # 空間合併前
        layers['Gray_convs_0_SFM'] = model.Gray_convs[:2].to(device)  # 空間合併後

        layers['Gray_convs_1_Conv'] = nn.Sequential(*(list(model.Gray_convs[:2]) + list([model.Gray_convs[2][0]]))).to(device)  # 只跑卷積
        layers['Gray_convs_1'] = nn.Sequential(*(list(model.Gray_convs[:2]) + list([model.Gray_convs[2][:-1]]))).to(device)  # 空間合併前
        layers['Gray_convs_1_SFM'] = model.Gray_convs[:3].to(device)  # 空間合併後

        layers['Gray_convs_2_Conv'] = nn.Sequential(*(list(model.Gray_convs[:3]) + list([model.Gray_convs[3][0]]))).to(device)  # 只跑卷積
        layers['Gray_convs_2'] = nn.Sequential(*(list(model.Gray_convs))).to(device)

    return layers

# get all layers for draw CI, RM, RM_Monitor
layers_infos = [
    {"layer_num": "RGB_convs_0", "is_gray": False, "plot_shape": (5, 6)},
    {"layer_num": "RGB_convs_1", "is_gray": False, "plot_shape": None},
    {"layer_num": "RGB_convs_2", "is_gray": False, "plot_shape": None},
    {"layer_num": "Gray_convs_0", "is_gray": True, "plot_shape": (7, 10)},
    {"layer_num": "Gray_convs_1", "is_gray": True, "plot_shape": None},
    {"layer_num": "Gray_convs_2", "is_gray": True, "plot_shape": None},
]

# arch = {
#     "name": 'AlexNet',
#     "args":{
#         'in_channels':3,
#         "num": 10
#     }
# }

# arch = {
#     "name": 'ResNet',
#     "args":{
#         'layers':18,
#         'in_channels':3,
#         "out_channels": 10
#     }
# }

# arch = {
#     "name": 'GoogLeNet',
#     "args":{
#         'in_channels':3,
#         "out_channels": 10
#     }
# }

# arch = {
#     "name": 'DenseNet',
#     "args":{
#         'in_channels':3,
#         "out_channels": 10
#     }
# }


create_dir = False
if create_dir:
    save_dir = increment_path('Code/runs/train/exp', exist_ok = False)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
else:
    save_dir = 'Code/runs/train/exp'

print(save_dir)

lr_scheduler = {
    "name": "ReduceLROnPlateau",
    "args":{
        "patience": 100
    }
}

optimizer = {
    "name":"Adam",
    "args":{

    }
}


config = {
    "device": device,
    "root": os.path.dirname(__file__),
    "save_dir": save_dir,
    "model": arch,
    # "dataset": 'Colored_MNIST', # mnist、fashion、cifar10、
    # MultiColor_Shapes_Database、malaria、、MultiGrayShapesDataset、face_dataset  "PathMNIST"
    "dataset": 'MultiColor_Shapes_Database',
    "input_shape": (28, 28),
    "batch_size": 256,
    "epoch" : 200,
    "early_stop" : False,
    "patience" : 30, # How many epochs without progress, early stop
    "lr" : 0.001,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    "loss_fn": "CrossEntropyLoss",
    "use_weightsAdjust" : False,  #  Use class quantity to adjust loss function weights
    "loss_weights_rate" : [1, 5],  # Class loss function weights
    # Heart calcification detection
    "heart_calcification" :{
        "grid_size" : 75, # Image cutting size
        "need_resize_height" : True, # Whether to resize based on image height
        "resize_height" : 900,  # Resize size
        "threshold" : 0.5, # For calcification point bounding box, determine if it's a calcification point, shrink the bounding box
        "enhance_method" : 'none', # Data contrast enhancement method 'contrast' 'normalize' 'histogram_equalization' 'scale_and_offset' 'clahe' 'none'
        "contrast_factor" : 1.5, # Contrast factor, default is 1.0 (no change)
        "use_vessel_mask" : False # Use vessel mask
    },
    "layers_infos" : layers_infos
}