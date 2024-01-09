import torch
from utils import increment_path
from pathlib import Path
import shutil
import os

project = "paper experiment"
name = "SFMCNN"
group = "1/3"
tags = ["SFMCNN", "Rewrite"]
description = "1/3 想找到當初景豐的設定"
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

# arch = {
#     "name": 'SFMCNN',
#     "args":{
#         "in_channels": 1,
#         "out_channels": 10,
#         "Conv2d_kernel": [(5, 5), (1, 1), (1, 1), (1, 1)],
#         "SFM_filters": [(2, 2), (1, 3), (3, 1)],
#         "channels": [1, 100, 225, 625, 1225],
#         "strides": [4, 1, 1, 1],
#         "paddings": [0, 0, 0, 0],
#         "w_arr": [7.45, 12.45, 17.445, 27.5],
#         "percent": [0.5, 0.4, 0.3, 0.2],
#         "fc_input": 1225,
#         "device": device
#     }
# }

arch = {
    "name": 'SFMCNN_old',
    "args":{
        "in_channels": 1,
        "out_channels": 10,
    }
}

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
    "dataset": 'mnist',
    "input_shape": (28, 28),
    "rbf": "triangle",
    "batch_size": 128,
    "epoch" : 200,
    "lr" : 0.001,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    "loss_fn": "CrossEntropyLoss",
}