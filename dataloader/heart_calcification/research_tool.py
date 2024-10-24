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
use_vessel_mask = config["heart_calcification"]["use_vessel_mask"]


# 在全局範圍內紀錄 data_processor 和 model
data_processor = None
model = None

def get_data_processor():
    global data_processor
    if data_processor is None:  # 如果尚未產生過，則創建
        data_processor = HeartCalcificationDataProcessor(
                grid_size=grid_size, data_dir=data_dir,
                need_resize_height=need_resize_height, resize_height=resize_height,
                threshold=threshold, contrast_factor=contrast_factor, enhance_method=enhance_method,
                use_vessel_mask=use_vessel_mask)
    return data_processor

def load_model():
    global model
    if model is None:  # 如果尚未產生過，則創建
        models = {'SFMCNN': SFMCNN, 'RGB_SFMCNN': RGB_SFMCNN}
        checkpoint_filename = 'SFMCNN_best'
        checkpoint = torch.load(f'./pth/{config["dataset"]}_pth/{checkpoint_filename}.pth', weights_only=True)
        model = models[arch['name']](**dict(config['model']['args']))
        model.load_state_dict(checkpoint['model_weights'])
        model.cpu()
        model.eval()
        summary(model, input_size=(config['model']['args']['in_channels'], *config['input_shape']), device='cpu')
        print(model)
    return model
