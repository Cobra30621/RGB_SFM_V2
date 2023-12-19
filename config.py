import torch
from utils import increment_path
from pathlib import Path
import shutil
import os

project = "paper experiment"
name = "SFMCNN"
group = "12/18"
tags = ["SFMCNN", "Rewrite"]
description = "12/18 老師架構圖實作,\n\
step4 時序的方式從成alpha成次方改成alpha扣定值"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = increment_path('./runs/train/exp', exist_ok = False)
Path(save_dir).mkdir(parents=True, exist_ok=True)
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

arch = {
    "name": 'SFMCNN',
    "args":{
        "in_channels": 3,
        "out_channels": 15,
        "Conv2d_kernel": [(3, 3), (1, 1), (1, 1), (1, 1)],
        "SFM_filters": [(5, 1), (1, 5), (2, 2)],
        "channels": [1, 16, 32, 64, 128],
        "strides": [1, 1, 1, 1],
        "paddings": [0, 0, 0, 0],
        "w_arr": [3, 4, 5.65, 8.0],
        "percent": [0.85, 0.7, 0.6, 0.5],
        "fc_input": 3*1*1*128,
        "device": device
    }
}

# arch = {
#     "name": 'CNN',
#     "args":{
#         "in_channels": 3,
#         "out_channels": 15,
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
    "save_dir": save_dir,
    "model": arch,
    "dataset": 'rgb_simple_shape',
    "input_shape": (30, 30),
    "rbf": "triangle",
    "batch_size": 32,
    "epoch" : 200,
    "lr" : 0.001,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    "loss_fn": "CrossEntropyLoss",
}