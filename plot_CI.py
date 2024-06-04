import torch
import torch.nn.functional as F
import numpy as np

from torchsummary import summary
from torch import nn

from config import *
from utils import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from dataloader import get_dataloader

images = []
image_paths = []
labels = []

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
checkpoint_filename = '0604_RGB_SFMCNN_best_zxd1v36g'
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
	kernel_size = arch['args']['Conv2d_kernel'][0]
	weights = torch.concat([model.RGB_conv2d[0].weights, model.RGB_conv2d[0].black_block, model.RGB_conv2d[0].white_block])
	weights = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
	weights = weights.repeat(1,1,*kernel_size)
	FMs['RGB_Conv2d'] = weights
	print(f'FM[RGB_Conv2d] shape: {FMs["RGB_Conv2d"].shape}')

	FMs['Gray_Conv2d'] = model.GRAY_conv2d[0].weight.reshape(arch['args']['channels'][0][1],1,*kernel_size)
	print(f'FM[Gray_Conv2d] shape: {FMs["Gray_Conv2d"].shape}')

	FMs[1] = model.convs[0][0].weight.reshape(-1, int(model.convs[0][0].weight.shape[1]**0.5), int(model.convs[0][0].weight.shape[1]**0.5), 1)
	print(f'FM[1] shape: {FMs[1].shape}')

	FMs[2] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)
	print(f'FM[2] shape: {FMs[2].shape}')

layers = {}
if arch['args']['in_channels'] == 1:
	layers[0] = nn.Sequential(model.convs[0][:2])
	layers[1] = nn.Sequential(*(list(model.convs[0]) + list([model.convs[1][:2]])))
	layers[2] = nn.Sequential(*(list(model.convs[:2]) + list([model.convs[2][:2]])))
	layers[3] = nn.Sequential(*(list(model.convs[:3]) + list([model.convs[3][:2]])))
else:
	layers['RGB_Conv2d'] = model.RGB_conv2d[:2]
	layers['Gray_Conv2d'] = model.GRAY_conv2d[:2]

	def forward(image):
		with torch.no_grad():
			rgb_output = model.RGB_conv2d(image)
			gray_output = model.GRAY_conv2d(model.gray_transform(image))
			output = torch.concat(([rgb_output, gray_output]), dim=1)
			output = model.SFM(output)
			output = model.convs[0][:2](output)
		return output
	layers[1] = forward

	def forward(image):
		with torch.no_grad():
			rgb_output = model.RGB_conv2d(image)
			gray_output = model.GRAY_conv2d(model.gray_transform(image))
			output = torch.concat(([rgb_output, gray_output]), dim=1)
			output = model.SFM(output)
			output = nn.Sequential(
	            *model.convs[0],
	            model.convs[1][:2]
	        )(output)
		return output
	layers[2] = forward

CIs = {}
kernel_size=arch['args']['Conv2d_kernel'][0]
stride = (arch['args']['strides'][0], arch['args']['strides'][0]) 
if arch['args']['in_channels'] == 1:
	CIs[0], CI_idx, CI_values = get_ci(images, layers[0], kernel_size=kernel_size, stride=stride)
	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	CIs[3], CI_idx, CI_values = get_ci(images, layers[3], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:3]), dim=0))
else:
	CIs["RGB_Conv2d"], CI_idx, CI_values = get_ci(images, layers['RGB_Conv2d'], kernel_size, stride = stride)
	CIs["Gray_Conv2d"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers['Gray_Conv2d'], kernel_size, stride = stride)
	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))


save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/'

print('FM saving ...')
FMs_save_path = save_path + 'FMs/'
os.makedirs(FMs_save_path, exist_ok=True)
if arch['args']['in_channels'] == 1:
	plot_map(FMs[0].reshape(int(FMs[0].shape[0]**0.5), int(FMs[0].shape[0]**0.5), *FMs[0].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_0')
	plot_map(FMs[1].reshape(int(FMs[1].shape[0]**0.5), int(FMs[1].shape[0]**0.5), *FMs[1].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_1')
	plot_map(FMs[2].reshape(int(FMs[2].shape[0]**0.5), int(FMs[2].shape[0]**0.5), *FMs[2].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_2')
	plot_map(FMs[3].reshape(int(FMs[3].shape[0]**0.5), int(FMs[3].shape[0]**0.5), *FMs[3].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_3')
else:
	plot_map(FMs['RGB_Conv2d'].permute(0,2,3,1).reshape(5, 15, *arch['args']['Conv2d_kernel'][0], arch['args']['in_channels']).detach().numpy(), path=FMs_save_path+'/FMs_RGB_Conv2d')
	plot_map(FMs['Gray_Conv2d'].permute(0,2,3,1).reshape(10, 15, *arch['args']['Conv2d_kernel'][0], 1).detach().numpy(), path=FMs_save_path+'/FMs_Gray_Conv2d')
	plot_map(FMs[1].reshape(int(FMs[1].shape[0]**0.5), int(FMs[1].shape[0]**0.5), *FMs[1].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_1')
	plot_map(FMs[2].reshape(int(FMs[2].shape[0]**0.5), int(FMs[2].shape[0]**0.5), *FMs[2].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_2')
print('FM saved')

print('CI saving ...')
CIs_save_path = save_path + 'CIs/'
os.makedirs(CIs_save_path, exist_ok=True)
if arch['args']['in_channels'] == 1:
	plot_map(CIs[0].reshape(int(CIs[0].shape[0]**0.5), int(CIs[0].shape[0]**0.5), *CIs[0].shape[2:]).detach().numpy(), vmax=1, vmin=0, path=CIs_save_path+'/CIs_0')
	plot_map(CIs[1].reshape(int(CIs[1].shape[0]**0.5), int(CIs[1].shape[0]**0.5), *CIs[1].shape[2:]).detach().numpy(), vmax=1, vmin=0, path=CIs_save_path+'/CIs_1')
	plot_map(CIs[2].reshape(int(CIs[2].shape[0]**0.5), int(CIs[2].shape[0]**0.5), *CIs[2].shape[2:]).detach().numpy(), vmax=1, vmin=0, path=CIs_save_path+'/CIs_2')
	plot_map(CIs[3].reshape(int(CIs[3].shape[0]**0.5), int(CIs[3].shape[0]**0.5), *CIs[3].shape[2:]).detach().numpy(), vmax=1, vmin=0, path=CIs_save_path+'/CIs_3')
else:
	plot_map(CIs['RGB_Conv2d'].reshape(5, 15, *CIs['RGB_Conv2d'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_RGB_Conv2d')
	plot_map(CIs['Gray_Conv2d'].reshape(10, 15, *CIs['Gray_Conv2d'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_Gray_Conv2d')
	plot_map(CIs[1].reshape(int(CIs[1].shape[0]**0.5), int(CIs[1].shape[0]**0.5), *CIs[1].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_1')
	plot_map(CIs[2].reshape(int(CIs[2].shape[0]**0.5), int(CIs[2].shape[0]**0.5), *CIs[2].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_2')

print('CI saved')

save_data = {'FMs': FMs, 'CIs': CIs}
torch.save(save_data, save_path + 'FMs_CIs.pth')


