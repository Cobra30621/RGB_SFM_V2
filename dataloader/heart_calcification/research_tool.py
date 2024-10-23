from dataloader.heart_calcification.heart_calcification_data_processor import HeartCalcificationDataProcessor

from config import config, arch
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN

from torchsummary import summary
import os
from torchvision.utils import make_grid

import torch


# 初始化数据处理器
data_dir = "D://Paper/RGB_SFM/data/HeartCalcification/basic"  # 请替换为您的数据目录

grid_size = config["heart_calcification"]["grid_size"]
resize_height = config["heart_calcification"]["resize_height"]
need_resize_height = config["heart_calcification"]["need_resize_height"]
threshold = config["heart_calcification"]["threshold"]
contrast_factor = config["heart_calcification"]["contrast_factor"]
enhance_method = config["heart_calcification"]["enhance_method"]



def get_data_processor():
    return HeartCalcificationDataProcessor(grid_size, data_dir,
need_resize_height, resize_height, threshold, contrast_factor, enhance_method)

def load_model():
    models = {'SFMCNN': SFMCNN, 'RGB_SFMCNN':RGB_SFMCNN}
    checkpoint_filename = 'SFMCNN_best'
    checkpoint = torch.load(f'./pth/{config["dataset"]}_pth/{checkpoint_filename}.pth' , weights_only=True)
    model = models[arch['name']](**dict(config['model']['args']))
    model.load_state_dict(checkpoint['model_weights'])
    model.cpu()
    model.eval()
    summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']), device='cpu')
    print(model)
    return model