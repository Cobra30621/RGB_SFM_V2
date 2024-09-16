import torch
from utils import increment_path
from pathlib import Path
import shutil
import os

project = "paper experiment"
name = "SFMCNN"
group = "3/6"
tags = ["SFMCNN", "Rewrite"]
description = "3/6 測試 face 資料集， 15 * 15版本，input size 改為 (60,60)，無overlapping，後面加入(1,2) SFM layer 和一層 (2025,1225,1,1) 的 Conv layer"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

arch = {
    "name": 'SFMCNN',
    "args":{
        "in_channels": 1,
        "out_channels": 2,
        "Conv2d_kernel": [(15, 15), (1, 1), (1, 1), (1, 1)],
        "SFM_filters": [(1, 2), (2, 1), (1, 2)],
        "channels": [1, 225, 625, 1225, 2025],
        "strides": [15, 1, 1, 1],
        "paddings": [0, 0, 0, 0],

        "rbfs": ['triangle', 'triangle', 'triangle', 'triangle'],
        "w_arr": [17.45, 17.45, 27.45, 37.45],
        "percents": [0.5, 0.4, 0.3, 0.2],

        # "rbfs": ['gauss', 'gauss', 'gauss', 'gauss'],
        # "stds": [2, 2, 2, 2],
        # "biass": [0.7, 0.7, 0.7, 0.7],

        "fc_input": 2*1*2025,
        "device": device
    }
}

# arch = {
#     "name": 'SFMCNN_old',
#     "args":{
#         "in_channels": 1,
#         "out_channels": 10,
#     }
# }

# arch = {
#     "name": 'AlexNet',
#     "args":{
#         "num_classes":15
#     }
# }

config = {
    "device": device,
    "root": os.path.dirname(__file__),
    "save_dir": './runs/train/exp',
    "model": arch,
    "dataset": 'face_dataset',
    "input_shape": (60, 60),
    "batch_size": 128,
    "epoch" : 20,
    "lr" : 0.001,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    "loss_fn": "CrossEntropyLoss",
}