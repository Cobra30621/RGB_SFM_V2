from pathlib import Path
import glob
import re
import torchvision
import matplotlib.pyplot as plt
import os
import torch
import math
import torch.nn.functional as F
from config import *
from utils import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from dataloader import get_dataloader
from collections import defaultdict

'''
    將 image 按照 kernel size 進行切割
'''
def show_Image_split(test_img, save_path = ""):
    segments = split(test_img.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0], stride = (arch['args']['strides'][0], arch['args']['strides'][0]))[0]
    plot_map(segments.permute(1,2,3,4,0), vmax=1, vmin=0, path='origin_split_0.png', cmap='gray')

    segments = segments.permute(1,2,3,4,0)
    print(segments.shape)
    segments = segments.reshape(segments.shape[0]//2, 2, segments.shape[1]//2, 2, 5, 5, 1)
    segments = segments.permute(0,2,1,4,3,5,6).reshape(segments.shape[0], segments.shape[2], 10, 10, 1)
    if save_path == "":
        plot_map(segments, vmax=1, vmin=0, path=save_path)
    else:
        plot_map(segments, vmax=1, vmin=0)

'''
    讀取圖片
'''
def read_Image(path = 'D:/Project/paper/RGB_SFM/showout/Colored_MNIST_0610_RGB_SFMCNN_best_t1np8eon_LAB/example/0/example_240/origin_240.png'):
    test_img = torchvision.io.read_image(path)
    test_img = test_img.to(torch.float32)
    test_img /= 255
    test_img = test_img[:3, :, :]
    return test_img

'''
    讀取 model
'''
def load_model(checkpoint_filename = 'RGB_SFMCNN_best_t1np8eon'):
        with torch.no_grad():
                # Load Dataset
                train_dataloader, test_dataloader = get_dataloader(dataset=config['dataset'], root=config['root'] + '/data/', batch_size=config['batch_size'], input_size=config['input_shape'])
                images, labels = torch.tensor([]), torch.tensor([])
                for batch in test_dataloader:
                        imgs, lbls = batch
                        images = torch.cat((images, imgs))
                        labels = torch.cat((labels, lbls))
                print(images.shape, labels.shape)

                # Load Model
                models = {'SFMCNN': SFMCNN, 'RGB_SFMCNN':RGB_SFMCNN}
                checkpoint = torch.load(f'./pth/{config["dataset"]}_pth/{checkpoint_filename}.pth')
                model = models[arch['name']](**dict(config['model']['args']))
                model.load_state_dict(checkpoint['model_weights'])
                model.cpu()
                model.eval()
                summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']), device='cpu')
                print(model)

                # Test Model
                batch_num = 1000
                pred = model(images[:batch_num])
                y = labels[:batch_num]
                correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                print("Test Accuracy: " + str(correct/len(pred)))
        return model

def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path

def plot_map(rm, grid_size=None, rowspan=None, colspan = None, path=None, **kwargs):
    rows, cols, e_h, e_w, _ = rm.shape
    if rowspan is None:
        rowspan = int(e_h / min(e_h, e_w))
    if colspan is None:
        colspan = int(e_w / min(e_h, e_w))
    if grid_size is None:
        grid_size = (rows*rowspan, cols*colspan)
    fig = plt.figure(figsize=(grid_size[1], grid_size[0]), facecolor="white")
    for row in range(rows):
        for col in range(cols):
            ax = plt.subplot2grid(grid_size, (row*rowspan, col*colspan), rowspan=rowspan, colspan=colspan)
            im = ax.imshow(rm[row][col], **kwargs)
            ax.axis('off')
    
    if path:
        plt.savefig(path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
        plt.close()
    
def split(input, kernel_size = (5, 5), stride = (5,5)):
    batch, channel, h, w = input.shape
    output_height = math.floor((h  - (kernel_size[0] - 1) - 1) / stride[0] + 1)
    output_width = math.floor((w  - (kernel_size[1] - 1) - 1) / stride[1] + 1)
    input = torch.tensor(input)
    segments = F.unfold(input, kernel_size=kernel_size, stride=stride).reshape(batch, channel, *kernel_size, -1).permute(0,1,4,2,3)
    segments = segments.reshape(batch, channel, output_height, output_width, *kernel_size) 
    return segments

def get_ci(input, layer, kernel_size = (5,5), stride= (5,5), sfm_filter = (1,1)):
    segments = split(input, kernel_size, stride)
    combine_h, combine_w, ci_h, ci_w = (int(segments.shape[2]/sfm_filter[0]), int(segments.shape[3]/sfm_filter[1]), int(segments.shape[4]*sfm_filter[0]), int(segments.shape[5]*sfm_filter[1]))
    segments = segments.reshape(-1, input.shape[1], combine_h, sfm_filter[0], combine_w, sfm_filter[1], segments.shape[4], segments.shape[5])
    segments = segments.permute(0, 2, 4, 3, 6, 5, 7, 1)
    segments = segments.reshape(-1, ci_h, ci_w, input.shape[1])
    print(f"segments shape: {segments.shape}")
    
    with torch.no_grad():
        outputs = layer(input)
        n_filters = outputs.shape[1]
        outputs = outputs.permute(0,2,3,1).reshape(-1, n_filters)
        print(f"output shape: {outputs.shape}")

    k = 1
    CI = torch.empty(n_filters, k, ci_h, ci_w, input.shape[1])
    CI_values = torch.empty(n_filters, k) 
    CI_idx = torch.empty(n_filters, k)    
    for i in range(n_filters):
        values, indices = torch.topk(outputs[:, i], k=k, largest=True)
        CI_idx[i] = indices
        CI_values[i] = values
        CI[i] = segments[indices.tolist()]
    print(f"CI shape: {CI.shape}")
    return CI, CI_idx, CI_values

'''
    使用於讀取cifar10
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

    

