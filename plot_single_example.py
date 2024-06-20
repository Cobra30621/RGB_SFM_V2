import torch
import torchvision
import random
import torch.nn.functional as F
import numpy as np

from torchsummary import summary
from torch import nn

from config import *
from utils import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from dataloader import get_dataloader

import matplotlib
# matplotlib.use('Agg')

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
	checkpoint_filename = '0610_RGB_SFMCNN_best_t1np8eon'
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
	input()

# FMs = {}
# if arch['args']['in_channels'] == 1:
# 	FMs[0] = model.convs[0][0].weight.reshape(-1, *arch['args']['Conv2d_kernel'][0], 1)
# 	print(f'FM[0] shape: {FMs[0].shape}')
# 	FMs[1] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)
# 	print(f'FM[1] shape: {FMs[1].shape}')
# 	FMs[2] = model.convs[2][0].weight.reshape(-1, int(model.convs[2][0].weight.shape[1]**0.5), int(model.convs[2][0].weight.shape[1]**0.5), 1)
# 	print(f'FM[2] shape: {FMs[2].shape}')
# 	FMs[3] = model.convs[3][0].weight.reshape(-1, int(model.convs[3][0].weight.shape[1]**0.5), int(model.convs[3][0].weight.shape[1]**0.5), 1)
# 	print(f'FM[3] shape: {FMs[3].shape}')
# else:
# 	# kernel_size = arch['args']['Conv2d_kernel'][0]
# 	# weights = torch.concat([model.RGB_conv2d[0].weights, model.RGB_conv2d[0].black_block, model.RGB_conv2d[0].white_block])
# 	# weights = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
# 	# weights = weights.repeat(1,1,*kernel_size)
# 	# FMs['RGB_Conv2d'] = weights
# 	# print(f'FM[RGB_Conv2d] shape: {FMs["RGB_Conv2d"].shape}')

# 	# FMs['Gray_Conv2d'] = model.GRAY_conv2d[0].weight.reshape(arch['args']['channels'][0][1],1,*kernel_size)
# 	# print(f'FM[Gray_Conv2d] shape: {FMs["Gray_Conv2d"].shape}')

# 	# print(model.convs[0][0].weight.shape)
# 	# FMs[1] = model.convs[0][0].weight.reshape(-1, int(model.convs[0][0].weight.shape[1]**0.5), int(model.convs[0][0].weight.shape[1]**0.5), 1)
# 	# print(f'FM[1] shape: {FMs[1].shape}')

# 	# FMs[2] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)
# 	# print(f'FM[2] shape: {FMs[2].shape}')

# 	# 平行架構
# 	kernel_size = arch['args']['Conv2d_kernel'][0]
# 	weights = torch.concat([model.RGB_convs[0][0].weights, model.RGB_convs[0][0].black_block, model.RGB_convs[0][0].white_block])
# 	weights = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
# 	weights = weights.repeat(1,1,*kernel_size)
# 	FMs['RGB_convs_0'] = weights
# 	print(f'FM[RGB_convs_0] shape: {FMs["RGB_convs_0"].shape}')

# 	FMs['RGB_convs_1'] = model.RGB_convs[2][0].weight.reshape(-1, 2, 15, 1)
# 	print(f'FM[RGB_convs_1] shape: {FMs["RGB_convs_1"].shape}')

# 	FMs['RGB_convs_2'] = model.RGB_convs[3][0].weight.reshape(-1, int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), 1)
# 	print(f'FM[RGB_convs_2] shape: {FMs["RGB_convs_2"].shape}')


# 	FMs['Gray_convs_0'] = model.Gray_convs[0][0].weight.reshape(arch['args']['channels'][1][0],1,*kernel_size)
# 	print(f'FM[Gray_convs_0] shape: {FMs["Gray_convs_0"].shape}')
# 	FMs['Gray_convs_1'] = model.Gray_convs[2][0].weight.reshape(-1, 7, 10, 1)
# 	print(f'FM[Gray_convs_1] shape: {FMs["Gray_convs_1"].shape}')
# 	FMs['Gray_convs_2'] = model.Gray_convs[3][0].weight.reshape(-1, int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), 1)
# 	print(f'FM[Gray_convs_2] shape: {FMs["Gray_convs_2"].shape}')
	
# layers = {}
# if arch['args']['in_channels'] == 1:
# 	layers[0] = nn.Sequential(model.convs[0][:2])
# 	layers[1] = nn.Sequential(*(list(model.convs[0]) + list([model.convs[1][:2]])))
# 	layers[2] = nn.Sequential(*(list(model.convs[:2]) + list([model.convs[2][:2]])))
# 	layers[3] = nn.Sequential(*(list(model.convs[:3]) + list([model.convs[3][:2]])))
# else:
# 	layers['RGB_convs_0'] = model.RGB_convs[0]
# 	layers['RGB_convs_1'] = nn.Sequential(*(list(model.RGB_convs[:2]) + list([model.RGB_convs[2][:2]])))
# 	layers['RGB_convs_2'] = nn.Sequential(*(list(model.RGB_convs[:3]) + list([model.RGB_convs[3][:2]])))
	
# 	layers['Gray_convs_0'] = model.Gray_convs[0]
# 	layers['Gray_convs_1'] = nn.Sequential(*(list(model.Gray_convs[:2]) + list([model.Gray_convs[2][:2]])))
# 	layers['Gray_convs_2'] = nn.Sequential(*(list(model.Gray_convs[:3]) + list([model.Gray_convs[3][:2]])))
	
# CIs = {}
# kernel_size=arch['args']['Conv2d_kernel'][0]
# stride = (arch['args']['strides'][0], arch['args']['strides'][0]) 
# if arch['args']['in_channels'] == 1:
# 	CIs[0], CI_idx, CI_values = get_ci(images, layers[0], kernel_size=kernel_size, stride=stride)
# 	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
# 	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
# 	CIs[3], CI_idx, CI_values = get_ci(images, layers[3], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:3]), dim=0))
# else:
# 	CIs["RGB_convs_0"], CI_idx, CI_values = get_ci(images, layers['RGB_convs_0'], kernel_size, stride = stride)
# 	CIs["RGB_convs_1"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_1"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
# 	CIs["RGB_convs_2"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_2"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	

# 	CIs["Gray_convs_0"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers['Gray_convs_0'], kernel_size, stride = stride)
# 	CIs["Gray_convs_1"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_1"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
# 	CIs["Gray_convs_2"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_2"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	

# if arch['args']['in_channels'] == 3:
# 	label_to_idx = {}
# 	i = 0
# 	for c in ['red', 'green', 'blue']:
# 	    for n in range(10):
# 	        label_to_idx[c+'_'+str(n)] = i
# 	        i+=1
# 	idx_to_label = {value: key for key, value in label_to_idx.items()}

test_id = "440"
test_img = torchvision.io.read_image(f'./test_images/origin_{test_id}.png')
test_img = test_img.to(torch.float32)
test_img /= 255
test_img = test_img[:3, :, :]
label = "green_5"

save_path = f'./single_detect/{config["dataset"]}_{checkpoint_filename}/example/{label}/{test_id}/'
# save_path = f'./single_detect/{config["dataset"]}_{checkpoint_filename}/example/{labels[test_id].argmax().item()}/example_{test_id}/'
RM_save_path = f'{save_path}/RMs/'
RM_CI_save_path = f'{save_path}/RM_CIs/'
os.makedirs(RM_save_path, exist_ok=True)
os.makedirs(RM_CI_save_path, exist_ok=True)

segments = split(test_img.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0], stride = (arch['args']['strides'][0], arch['args']['strides'][0]))[0]
plot_map(segments.permute(1,2,3,4,0), vmax=1, vmin=0, path=save_path + f'origin_split_0.png')

segments = segments.permute(1,2,3,4,0)
segments = segments.reshape(segments.shape[0]//2, 2, segments.shape[1]//2, 2, 5, 5, 3)
segments = segments.permute(0,2,1,4,3,5,6).reshape(segments.shape[0], segments.shape[2], 10, 10, 3)
plot_map(segments, vmax=1, vmin=0, path=save_path + f'origin_split_1.png')

segments = segments.reshape(segments.shape[0]//1, 1, segments.shape[1]//3, 3, 10, 10, 3)
segments = segments.permute(0,2,1,4,3,5,6).reshape(segments.shape[0], segments.shape[2], 10, 30, 3)
plot_map(segments, vmax=1, vmin=0, path=save_path + f'origin_split_2.png')

# if arch['args']['in_channels'] == 1:
# 	torchvision.utils.save_image(test_img, save_path + f'{test_id}.png')
# else:
# 	plt.imsave(save_path + f'origin_{test_id}.png', test_img.permute(1,2,0).detach().numpy())

# segments = split(test_img.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0], stride = (arch['args']['strides'][0], arch['args']['strides'][0]))[0]
# plot_map(segments.permute(1,2,3,4,0), vmax=1, vmin=0, path=save_path + f'origin_split_{test_id}.png')

# RM_CIs = {}

# if arch['args']['in_channels'] == 1:
# 	# Layer 0
# 	layer_num = 0
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
# 	plot_map(RM_CI.detach().numpy(), vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
# 	RM_CIs[layer_num] = RM_CI

# 	# Layer 1
# 	layer_num = 1
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	plot_map(RM_CI.detach().numpy(), vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
# 	RM_CIs[layer_num] = RM_CI

# 	# Layer 2
# 	layer_num = 2
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	plot_map(RM_CI.detach().numpy(), vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
# 	RM_CIs[layer_num] = RM_CI

# 	# Layer 3
# 	layer_num = 3
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	plot_map(RM_CI.detach().numpy(), vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI', cmap='gray')
# 	RM_CIs[layer_num] = RM_CI

# else:
# 	# 平行架構
# 	# RGB_convs_0
# 	layer_num = 'RGB_convs_0'
# 	plot_shape = (2,15)
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	print(f"{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	FM_H, FM_W = FMs[layer_num].shape[2], FMs[layer_num].shape[3]
# 	RM_FM = FMs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].permute(0,2,3,1).reshape(RM_H,RM_W,FM_H,FM_W,arch['args']['in_channels'])
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
# 	RM_CI = RM_CI.permute(0,1,4,2,3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)
# 	RM_CIs[layer_num] = RM_CI
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	plot_map(RM_FM.detach().numpy(), path=RM_CI_save_path + f'{layer_num}_RM_FM')
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')

# 	# RGB_convs_1
# 	layer_num = 'RGB_convs_1'
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
# 	RM_CI = RM_CI.reshape(RM_H,RM_W, CI_H//RM_CIs['RGB_convs_0'].shape[2], RM_CIs['RGB_convs_0'].shape[2], CI_W//RM_CIs['RGB_convs_0'].shape[3], RM_CIs['RGB_convs_0'].shape[3], arch['args']['in_channels'])
# 	RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6)
# 	RM_CI = RM_CI.reshape(RM_CIs['RGB_convs_0'].shape)
# 	RM_CI = RM_CI.permute(0,1,4,2,3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)
# 	RM_CI = RM_CI.reshape(RM_H, RM_CI.shape[0]//RM_H, RM_W, RM_CI.shape[1]//RM_W, *RM_CI.shape[2:])
# 	RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6).reshape(RM_H, RM_W, CI_H, CI_W, 3)
# 	RM_CIs[layer_num] = RM_CI
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')

# 	# RGB_convs_2
# 	layer_num = 'RGB_convs_2'
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI_origin')
# 	RM_CI = RM_CI.reshape(RM_H,RM_W, CI_H//RM_CIs['RGB_convs_0'].shape[2], RM_CIs['RGB_convs_0'].shape[2], CI_W//RM_CIs['RGB_convs_0'].shape[3], RM_CIs['RGB_convs_0'].shape[3], arch['args']['in_channels'])
# 	RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6)
# 	RM_CI = RM_CI.reshape(RM_CIs['RGB_convs_0'].shape)
# 	RM_CI = RM_CI.permute(0,1,4,2,3).reshape(*RM_CI.shape[:2], arch['args']['in_channels'], -1).mean(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1, 1, *RM_CI.shape[2:4], 1)
# 	RM_CI = RM_CI.reshape(RM_H, RM_CI.shape[0]//RM_H, RM_W, RM_CI.shape[1]//RM_W, *RM_CI.shape[2:])
# 	RM_CI = RM_CI.permute(0, 2, 1, 4, 3, 5, 6).reshape(RM_H, RM_W, CI_H, CI_W, 3)
# 	RM_CIs[layer_num] = RM_CI
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')


# 	# Gray_convs_0
# 	layer_num = 'Gray_convs_0'
# 	plot_shape = (7,10)
# 	print(model.gray_transform)
# 	print(model.gray_transform(test_img.unsqueeze(0)).shape)
# 	RM = layers[layer_num](model.gray_transform(test_img.unsqueeze(0)))[0]
# 	print(f"{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
# 	RM_CIs[layer_num] = RM_CI
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

# 	# Gray_convs_1
# 	layer_num = 'Gray_convs_1'
# 	RM = layers[layer_num](model.gray_transform(test_img.unsqueeze(0)))[0]
# 	plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
# 	RM_CIs[layer_num] = RM_CI
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

# 	# Gray_convs_2
# 	layer_num = 'Gray_convs_2'
# 	RM = layers[layer_num](model.gray_transform(test_img.unsqueeze(0)))[0]
# 	plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
# 	print(f"Layer{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
# 	RM_CIs[layer_num] = RM_CI
# 	plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 	plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI', cmap='gray')

# # 全連接層呈現
# rgb_x = model.RGB_convs(test_img.unsqueeze(0))[0]
# print(torch.topk(rgb_x, k=1, dim=0))
# origin_rgb_output_shape = rgb_x.shape
# rgb_x = rgb_x.reshape(-1)
# rgb_w_x = rgb_x * model.fc1[0].weight[:, : rgb_x.shape[0] ]
# print(rgb_w_x.shape)
# print(torch.topk(rgb_w_x[label_to_idx[label]], k=1).indices)
# max_flatten_index = torch.topk(rgb_w_x[label_to_idx[label]], k=1)
# print(max_flatten_index)
# max_flatten_index = max_flatten_index.indices
# original_index = (
#     max_flatten_index // (origin_rgb_output_shape[1] * origin_rgb_output_shape[2]),
#     (max_flatten_index % (origin_rgb_output_shape[1] * origin_rgb_output_shape[2])) // origin_rgb_output_shape[2],
#     max_flatten_index % origin_rgb_output_shape[2]
# )
# RM_CI = CIs['RGB_convs_2'][original_index[0]][0, 0]
# RM_CI = RM_CI.reshape(RM_CI.shape[0]//5, 5, RM_CI.shape[1]//5, 5, 3)
# RM_CI = RM_CI.permute(0,2,1,3,4)
# RM_CI = RM_CI.reshape(RM_CI.shape[0], RM_CI.shape[1], 25, 3).mean(dim=-2).unsqueeze(-2).repeat(1,1,25,1)
# RM_CI = RM_CI.reshape(RM_CI.shape[0], RM_CI.shape[1], 5, 5, 3).permute(0,2,1,3,4)
# RM_CI = RM_CI.reshape(RM_CI.shape[0]*5, RM_CI.shape[2]*5, 3)
# fig, ax = plt.subplots()
# ax.imshow(RM_CI.detach().numpy())
# ax.set_axis_off()
# fig.savefig('Linear_color.png', bbox_inches='tight', pad_inches=0)

# gray_x = model.Gray_convs(model.gray_transform(test_img.unsqueeze(0)))[0]
# origin_gray_output_shape = gray_x.shape
# gray_x = gray_x.reshape(-1)
# gray_w_x = gray_x * model.fc1[0].weight[:, rgb_x.shape[0]:]
# max_flatten_index = torch.topk(gray_w_x[label_to_idx[label]], k=1)
# max_flatten_index = max_flatten_index.indices
# original_index = (
#     max_flatten_index // (origin_gray_output_shape[1] * origin_gray_output_shape[2]),
#     (max_flatten_index % (origin_gray_output_shape[1] * origin_gray_output_shape[2])) // origin_gray_output_shape[2],
#     max_flatten_index % origin_gray_output_shape[2]
# )
# RM_CI = CIs['Gray_convs_2'][original_index[0]][0, 0]
# fig, ax = plt.subplots()
# ax.imshow(RM_CI.detach().numpy(), cmap='gray')
# ax.set_axis_off()
# fig.savefig('Linear_gray.png', bbox_inches='tight', pad_inches=0)
# result = torch.concat([rgb_w_x, gray_w_x], dim=-1)
# result = result[label_to_idx[label]]


plt.close('all')
