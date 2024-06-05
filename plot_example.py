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
	checkpoint_filename = '0605_RGB_SFMCNN_best_plwg1y0a'
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

FMs = {}
if arch['args']['in_channels'] == 1:
	FMs[0] = model.convs[0][0].weight.reshape(-1, *arch['args']['Conv2d_kernel'][0], 1)
	print(f'FM[0] shape: {FMs[0].shape}')
	FMs[1] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)
	print(f'FM[1] shape: {FMs[1].shape}')
	FMs[2] = model.convs[2][0].weight.reshape(-1, int(model.convs[2][0].weight.shape[1]**0.5), int(model.convs[2][0].weight.shape[1]**0.5), 1)
	print(f'FM[2] shape: {FMs[2].shape}')
	FMs[3] = model.convs[3][0].weight.reshape(-1, int(model.convs[3][0].weight.shape[1]**0.5), int(model.convs[3][0].weight.shape[1]**0.5), 1)
	print(f'FM[3] shape: {FMs[3].shape}')
else:
	# kernel_size = arch['args']['Conv2d_kernel'][0]
	# weights = torch.concat([model.RGB_conv2d[0].weights, model.RGB_conv2d[0].black_block, model.RGB_conv2d[0].white_block])
	# weights = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
	# weights = weights.repeat(1,1,*kernel_size)
	# FMs['RGB_Conv2d'] = weights
	# print(f'FM[RGB_Conv2d] shape: {FMs["RGB_Conv2d"].shape}')

	# FMs['Gray_Conv2d'] = model.GRAY_conv2d[0].weight.reshape(arch['args']['channels'][0][1],1,*kernel_size)
	# print(f'FM[Gray_Conv2d] shape: {FMs["Gray_Conv2d"].shape}')

	# print(model.convs[0][0].weight.shape)
	# FMs[1] = model.convs[0][0].weight.reshape(-1, int(model.convs[0][0].weight.shape[1]**0.5), int(model.convs[0][0].weight.shape[1]**0.5), 1)
	# print(f'FM[1] shape: {FMs[1].shape}')

	# FMs[2] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)
	# print(f'FM[2] shape: {FMs[2].shape}')

	# 平行架構
	kernel_size = arch['args']['Conv2d_kernel'][0]
	weights = torch.concat([model.RGB_convs[0][0].weights, model.RGB_convs[0][0].black_block, model.RGB_convs[0][0].white_block])
	weights = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
	weights = weights.repeat(1,1,*kernel_size)
	FMs['RGB_convs_0'] = weights
	print(f'FM[RGB_convs_0] shape: {FMs["RGB_convs_0"].shape}')

	FMs['RGB_convs_1'] = model.RGB_convs[2][0].weight.reshape(-1, 2, 15, 1)
	print(f'FM[RGB_convs_1] shape: {FMs["RGB_convs_1"].shape}')

	FMs['RGB_convs_2'] = model.RGB_convs[3][0].weight.reshape(-1, int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), 1)
	print(f'FM[RGB_convs_2] shape: {FMs["RGB_convs_2"].shape}')

	FMs['Gray_convs_0'] = model.Gray_convs[0][0].weight.reshape(arch['args']['channels'][1][0],1,*kernel_size)
	print(f'FM[Gray_convs_0] shape: {FMs["Gray_convs_0"].shape}')

	FMs['Gray_convs_1'] = model.Gray_convs[2][0].weight.reshape(-1, 7, 10, 1)
	print(f'FM[Gray_convs_1] shape: {FMs["Gray_convs_1"].shape}')

	FMs['Gray_convs_2'] = model.Gray_convs[3][0].weight.reshape(-1, int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), 1)
	print(f'FM[Gray_convs_2] shape: {FMs["Gray_convs_2"].shape}')

layers = {}
if arch['args']['in_channels'] == 1:
	layers[0] = nn.Sequential(model.convs[0][:2])
	layers[1] = nn.Sequential(*(list(model.convs[0]) + list([model.convs[1][:2]])))
	layers[2] = nn.Sequential(*(list(model.convs[:2]) + list([model.convs[2][:2]])))
	layers[3] = nn.Sequential(*(list(model.convs[:3]) + list([model.convs[3][:2]])))
else:
	# layers['RGB_Conv2d'] = model.RGB_conv2d[:2]
	# layers['Gray_Conv2d'] = model.GRAY_conv2d[:2]

	# def forward(image):
	# 	with torch.no_grad():
	# 		rgb_output = model.RGB_conv2d(image)
	# 		gray_output = model.GRAY_conv2d(model.gray_transform(image))
	# 		output = torch.concat(([rgb_output, gray_output]), dim=1)
	# 		output = model.SFM(output)
	# 		output = model.convs[0][:2](output)
	# 	return output
	# layers[1] = forward

	# def forward(image):
	# 	with torch.no_grad():
	# 		rgb_output = model.RGB_conv2d(image)
	# 		gray_output = model.GRAY_conv2d(model.gray_transform(image))
	# 		output = torch.concat(([rgb_output, gray_output]), dim=1)
	# 		output = model.SFM(output)
	# 		output = nn.Sequential(
	#             *model.convs[0],
	#             model.convs[1][:2]
	#         )(output)
	# 	return output
	# layers[2] = forward

	layers['RGB_convs_0'] = model.RGB_convs[0]
	layers['RGB_convs_1'] = nn.Sequential(*(list(model.RGB_convs[:2]) + list([model.RGB_convs[2][:2]])))
	layers['RGB_convs_2'] = nn.Sequential(*(list(model.RGB_convs[:3]) + list([model.RGB_convs[3][:2]])))

	layers['Gray_convs_0'] = model.Gray_convs[0]
	layers['Gray_convs_1'] = nn.Sequential(*(list(model.Gray_convs[:2]) + list([model.Gray_convs[2][:2]])))
	layers['Gray_convs_2'] = nn.Sequential(*(list(model.Gray_convs[:3]) + list([model.Gray_convs[3][:2]])))

CIs = {}
kernel_size=arch['args']['Conv2d_kernel'][0]
stride = (arch['args']['strides'][0], arch['args']['strides'][0]) 
if arch['args']['in_channels'] == 1:
	CIs[0], CI_idx, CI_values = get_ci(images, layers[0], kernel_size=kernel_size, stride=stride)
	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	CIs[3], CI_idx, CI_values = get_ci(images, layers[3], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:3]), dim=0))
else:
	# CIs["RGB_Conv2d"], CI_idx, CI_values = get_ci(images, layers['RGB_Conv2d'], kernel_size, stride = stride)
	# CIs["Gray_Conv2d"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers['Gray_Conv2d'], kernel_size, stride = stride)
	# CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	# CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))

	CIs["RGB_convs_0"], CI_idx, CI_values = get_ci(images, layers['RGB_convs_0'], kernel_size, stride = stride)
	CIs['RGB_convs_0'] = CIs["RGB_convs_0"].permute(0,1,4,2,3).reshape(*CIs['RGB_convs_0'].shape[:2], arch['args']['in_channels'], -1).mean(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1,1,*CIs['RGB_convs_0'].shape[2:4], 1)
	CIs["RGB_convs_1"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_1"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs['RGB_convs_1'] = CIs["RGB_convs_1"].permute(0,1,4,2,3).reshape(*CIs['RGB_convs_1'].shape[:2], arch['args']['in_channels'], -1).mean(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1,1,*CIs['RGB_convs_1'].shape[2:4], 1)
	CIs["RGB_convs_2"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_2"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	CIs['RGB_convs_2'] = CIs["RGB_convs_2"].permute(0,1,4,2,3).reshape(*CIs['RGB_convs_2'].shape[:2], arch['args']['in_channels'], -1).mean(dim=-1).unsqueeze(-2).unsqueeze(-2).repeat(1,1,*CIs['RGB_convs_2'].shape[2:4], 1)

	CIs["Gray_convs_0"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers['Gray_convs_0'], kernel_size, stride = stride)
	CIs["Gray_convs_1"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_1"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs["Gray_convs_2"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_2"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))


# def get_RM_CI(layer_num, test_img, plot_shape = None, RM_CI_save_path = None):
# 	RM = layers[layer_num](test_img.unsqueeze(0))[0]
# 	if plot_map == None:
# 		plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
# 	print(f"{layer_num}_RM: {RM.shape}")
# 	RM_H, RM_W = RM.shape[1], RM.shape[2]
# 	FM_H, FM_W = FMs[layer_num].shape[2], FMs[layer_num].shape[3]
# 	RM_FM = FMs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].permute(0,2,3,1).reshape(RM_H,RM_W,FM_H,FM_W,arch['args']['in_channels'])
# 	CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
# 	RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
# 	RM_CIs[layer_num] = RM_CI
# 	if RM_CI_save_path != None:
# 		plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
# 		# plot_map(RM_FM.detach().numpy(), path=RM_CI_save_path + f'{layer_num}_RM_FM')
# 		plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')
# 	return RM_CI


	for test_id in range(images.shape[0]):
		
		test_img = images[test_id]

		# 隨機選擇一個圖片
		# 455
		# while True:
		# 	test_id = random.randint(0, len(images))
		# 	test_img = images[test_id]
		# 	pred = model(test_img.unsqueeze(0)).argmax()
		# 	label = labels[test_id].argmax()
		# 	print(test_id, pred, label)
		# 	if pred == label and pred == y:
		# 		break
		

		save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/example_{test_id}/'
		RM_save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/example_{test_id}/RMs/'
		RM_CI_save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/example_{test_id}/RM_CIs/'
		os.makedirs(RM_save_path, exist_ok=True)
		os.makedirs(RM_CI_save_path, exist_ok=True)


		if arch['args']['in_channels'] == 1:
			torchvision.utils.save_image(test_img, save_path + f'origin_{test_id}.png')
		else:
			plt.imsave(save_path + f'origin_{test_id}.png', test_img.permute(1,2,0).detach().numpy())

		segments = split(test_img.unsqueeze(0), kernel_size=arch['args']['Conv2d_kernel'][0], stride = (arch['args']['strides'][0], arch['args']['strides'][0]))[0]
		plot_map(segments.permute(1,2,3,4,0), vmax=1, vmin=0, path=save_path + f'origin_split_{test_id}.png')

		RM_CIs = {}

		if arch['args']['in_channels'] == 1:
			# Layer 0
			layer_num = 0
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1), path=RM_save_path + f'{layer_num}_RM')
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
			plot_map(RM_CI, vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')
			RM_CIs[layer_num] = RM_CI

			# Layer 1
			layer_num = 1
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1), path=RM_save_path + f'{layer_num}_RM')
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			plot_map(RM_CI, vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')
			RM_CIs[layer_num] = RM_CI

			# Layer 2
			layer_num = 2
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1), path=RM_save_path + f'{layer_num}_RM')
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			plot_map(RM_CI, vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')
			RM_CIs[layer_num] = RM_CI

			# Layer 3
			layer_num = 3
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1), path=RM_save_path + f'{layer_num}_RM')
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			plot_map(RM_CI, vmax=1, vmin=0, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')
			RM_CIs[layer_num] = RM_CI

		else:
			# # RGB_Conv2d
			# layer_num = 'RGB_Conv2d'
			# plot_shape = (15,5)
			# RM = layers[layer_num](test_img.unsqueeze(0))[0]
			# print(f"{layer_num}_RM: {RM.shape}")
			# RM_H, RM_W = RM.shape[1], RM.shape[2]
			# FM_H, FM_W = FMs[layer_num].shape[2], FMs[layer_num].shape[3]
			# RM_FM = FMs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].permute(0,2,3,1).reshape(RM_H,RM_W,FM_H,FM_W,arch['args']['in_channels'])
			# CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			# RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			# RM_CIs[layer_num] = RM_CI
			
			# plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			# plot_map(RM_FM.detach().numpy(), path=RM_CI_save_path + f'{layer_num}_RM_FM')
			# plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')

			# # Gray_Conv2d
			# layer_num = 'Gray_Conv2d'
			# plot_shape = (10,15)
			# RM = layers[layer_num](model.gray_transform(test_img).unsqueeze(0))[0]
			# print(f"{layer_num}_RM: {RM.shape}")
			# RM_H, RM_W = RM.shape[1], RM.shape[2]
			# CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			# RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
			# RM_CIs[layer_num] = RM_CI

			# plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			# plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')

			# # Layer 1
			# layer_num = 1
			# RM = layers[layer_num](test_img.unsqueeze(0))[0]
			# print(f"Layer{layer_num}_RM: {RM.shape}")
			# RM_H, RM_W = RM.shape[1], RM.shape[2]
			# CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			# RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			# RM_CIs[layer_num] = RM_CI

			# plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1), path=RM_save_path + f'{layer_num}_RM')
			# plot_map(RM_CI, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')

			# # Layer 2
			# layer_num = 2
			# RM = layers[layer_num](test_img.unsqueeze(0))[0]
			# print(f"Layer{layer_num}_RM: {RM.shape}")
			# RM_H, RM_W = RM.shape[1], RM.shape[2]
			# CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			# RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			# RM_CIs[layer_num] = RM_CI

			# plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5),1), path=RM_save_path + f'{layer_num}_RM')
			# plot_map(RM_CI, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')


			# 平行架構
			# RGB_convs_0
			layer_num = 'RGB_convs_0'
			plot_shape = (2,15)
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			print(f"{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			FM_H, FM_W = FMs[layer_num].shape[2], FMs[layer_num].shape[3]
			RM_FM = FMs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].permute(0,2,3,1).reshape(RM_H,RM_W,FM_H,FM_W,arch['args']['in_channels'])
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			RM_CIs[layer_num] = RM_CI
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			plot_map(RM_FM.detach().numpy(), path=RM_CI_save_path + f'{layer_num}_RM_FM')
			plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')

			# RGB_convs_1
			layer_num = 'RGB_convs_1'
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			RM_CIs[layer_num] = RM_CI
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			plot_map(RM_CI, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')

			# RGB_convs_2
			layer_num = 'RGB_convs_2'
			RM = layers[layer_num](test_img.unsqueeze(0))[0]
			plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,arch['args']['in_channels'])
			RM_CIs[layer_num] = RM_CI
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			plot_map(RM_CI, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')

			# Gray_convs_0
			layer_num = 'Gray_convs_0'
			plot_shape = (7,10)
			RM = layers[layer_num](model.gray_transform(test_img).unsqueeze(0))[0]
			print(f"{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
			print(test_id, torch.topk(RM, k=1, dim=0, largest=True))
			RM_CIs[layer_num] = RM_CI
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			plot_map(RM_CI, path=RM_CI_save_path + f'{layer_num}_RM_CI')

			# Gray_convs_1
			layer_num = 'Gray_convs_1'
			RM = layers[layer_num](model.gray_transform(test_img).unsqueeze(0))[0]
			plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
			RM_CIs[layer_num] = RM_CI
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			plot_map(RM_CI, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')

			# Gray_convs_2
			layer_num = 'Gray_convs_2'
			RM = layers[layer_num](model.gray_transform(test_img).unsqueeze(0))[0]
			plot_shape = (int(RM.shape[0] ** 0.5),int(RM.shape[0] ** 0.5))
			print(f"Layer{layer_num}_RM: {RM.shape}")
			RM_H, RM_W = RM.shape[1], RM.shape[2]
			CI_H, CI_W = CIs[layer_num].shape[2], CIs[layer_num].shape[3]
			RM_CI = CIs[layer_num][torch.topk(RM, k=1, dim=0, largest=True).indices.flatten()].reshape(RM_H,RM_W,CI_H,CI_W,1)
			RM_CIs[layer_num] = RM_CI
			plot_map(RM.permute(1,2,0).reshape(RM_H,RM_W,*plot_shape,1).detach().numpy(), path=RM_save_path + f'{layer_num}_RM')
			plot_map(RM_CI, path=RM_CI_save_path + f'Layer{layer_num}_RM_CI')

		plt.close('all')

		# kernel_size = [arch['args']['Conv2d_kernel'][0], arch['args']['Conv2d_kernel'][0] * torch.prod(torch.tensor(arch['args']['SFM_filters'][:1])),\
		# 	arch['args']['Conv2d_kernel'][0] * torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]))]
		# label = labels[test_id]

		# for kernel_size, layer_name in zip(kernel_size, layer_names):
		# 	if layer_name == 'Gray_Conv2d':
	    #         split_img = split(gray_transform(test_img).unsqueeze(0), kernel_size, stride=kernel_size)
	    #     else:
	    #         split_img = split(test_img.unsqueeze(0), kernel_size, stride=kernel_size)
	    #     split_img = split_img[0].permute(1,2,3,4,0).detach().numpy()

	    #     RM_CI = RM_CIs[layer_name]['RM_CI']
        # 	RM_CI_idx = RM_CIs[layer_name]['RM_CI_idx']

        # 	dist_arr = np.zeros((RM_CI.shape[:2]))
	    #     for i in range(RM_CI.shape[0]):
	    #         for j in range(RM_CI.shape[1]):
	    #             dist_arr[i][j] = cdist(split_img[i][j], RM_CI[i][j])

        # 	dist_row_idx = dist_arr.argmin() // dist_arr.shape[1]
	    #     dist_col_idx = dist_arr.argmin() % dist_arr.shape[1]
	    #     train_img_idx = int(RM_CI_idx[dist_row_idx][dist_col_idx] // np.multiply(*dist_arr.shape))
	    #     voting[train_labels[train_img_idx].item()] += 1

		# if max(list(voting.values())) == 1:
		# 	pred = train_labels[train_img_idx].item()
		# else:
		# 	pred = max(voting, key=voting.get)






