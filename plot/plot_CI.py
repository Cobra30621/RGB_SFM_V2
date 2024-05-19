import torch
import torch.nn.functional as F
import numpy as np

from config import *
from utils import *
from SFMCNN import SFMCNN
from RGBCNN import RGB_SFMCNN

images = []
image_paths = []
labels = []

# Load Dataset
images, labels = load_data(config['dataset'])

# Load Model
checkpoint_filename = '0422_SFMCNN_best_fd198ukg'
checkpoint = torch.load(f'../pth/{config["dataset"]}_pth/{checkpoint_filename}.pth')
model = SFMCNN(**dict(config['model']['args']))
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
print("Test Accuracy: " + (correct/batch_num))

FMs = {}
if arch['in_channels'] == 1:
	FMs[0] = model.convs[0][0].weight.reshape(-1, *arch['args']['Conv2d_kernel'][0], 1)
	FMs[1] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)
	FMs[2] = model.convs[2][0].weight.reshape(-1, int(model.convs[2][0].weight.shape[1]**0.5), int(model.convs[2][0].weight.shape[1]**0.5), 1)
	FMs[3] = model.convs[3][0].weight.reshape(-1, int(model.convs[3][0].weight.shape[1]**0.5), int(model.convs[3][0].weight.shape[1]**0.5), 1)
else:
	weights = torch.concat([model.RGB_conv2d[0].weights, model.RGB_conv2d[0].black_block, model.RGB_conv2d[0].white_block])
	weights = weights.repeat(1,1,*kernel_size)
	FMs['RGB_Conv2d'] = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
	FMs['Gray_Conv2d'] = model.GRAY_conv2d[0].weight.reshape(arch['args']['channels'][0][1],1,*kernel_size)
	FMs[1] = model.convs[0][0].weight.reshape(-1, int(model.convs[0][0].weight.shape[1]**0.5), int(model.convs[0][0].weight.shape[1]**0.5), 1)
	FMs[2] = model.convs[1][0].weight.reshape(-1, int(model.convs[1][0].weight.shape[1]**0.5), int(model.convs[1][0].weight.shape[1]**0.5), 1)


layers = {}
if arch['in_channels'] == 1:
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
	        gray_output = model.GRAY_conv2d(gray_transform(image))
	        output = torch.concat(([rgb_output, gray_output]), dim=1)
	        output = model.SFM(output)
	        output = model.convs[0][:2](output)
    	return output
	layers[1] = forward

	def forward(image):
	    with torch.no_grad():
	        rgb_output = model.RGB_conv2d(image)
	        gray_output = model.GRAY_conv2d(gray_transform(image))
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
if arch['in_channels'] == 1:
	CIs[0], CI_idx, CI_values = get_ci(images, layers[0], kernel_size=kernel_size, stride=stride)
	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	CIs[3], CI_idx, CI_values = get_ci(images, layers[3], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:3]), dim=0))
else:
	CIs["RGB_Conv2d"], CI_idx, CI_values = get_ci(images, layers['RGB_Conv2d'], kernel_size, stride = stride)
	CIs["Gray_Conv2d"], CI_idx, CI_values = get_ci(images, layers['Gray_Conv2d'], kernel_size, stride = stride)
	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))


save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/'
FMs_save_path = save_path + 'FMs/'
os.makedirs(FMs_save_path, exist_ok=True)
if arch['in_channels'] == 1:
	plot_map(FMs[0].reshape(int(FMs[0].shape[0]**0.5), int(FMs[0].shape[0]**0.5), *FMs[0].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_0')
	plot_map(FMs[1].reshape(int(FMs[1].shape[0]**0.5), int(FMs[1].shape[0]**0.5), *FMs[1].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_1')
	plot_map(FMs[2].reshape(int(FMs[2].shape[0]**0.5), int(FMs[2].shape[0]**0.5), *FMs[2].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_2')
	plot_map(FMs[3].reshape(int(FMs[3].shape[0]**0.5), int(FMs[3].shape[0]**0.5), *FMs[3].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_3')
else:
	plot_map(FMs['RGB_Conv2d'].reshape(int(FMs['RGB_Conv2d'].shape[0]**0.5), int(FMs['RGB_Conv2d'].shape[0]**0.5), *FMs['RGB_Conv2d'].shape['RGB_Conv2d':]).detach().numpy(), path=FMs_save_path+'/FMs_'RGB_Conv2d'')
	plot_map(FMs[1].reshape(int(FMs[1].shape[0]**0.5), int(FMs[1].shape[0]**0.5), *FMs[1].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_1')
	plot_map(FMs[1].reshape(int(FMs[1].shape[0]**0.5), int(FMs[1].shape[0]**0.5), *FMs[1].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_1')
	plot_map(FMs[2].reshape(int(FMs[2].shape[0]**0.5), int(FMs[2].shape[0]**0.5), *FMs[2].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_2')

CIs_save_path = save_path + 'CIs/'
os.makedirs(CIs_save_path, exist_ok=True)
plot_map(CIs[0].reshape(int(CIs[0].shape[0]**0.5), int(CIs[0].shape[0]**0.5), *CIs[0].shape[1:]).detach().numpy(), path=CIs_save_path+'/CIs_0')
plot_map(CIs[1].reshape(int(CIs[1].shape[0]**0.5), int(CIs[1].shape[0]**0.5), *CIs[1].shape[1:]).detach().numpy(), path=CIs_save_path+'/CIs_1')
plot_map(CIs[2].reshape(int(CIs[2].shape[0]**0.5), int(CIs[2].shape[0]**0.5), *CIs[2].shape[1:]).detach().numpy(), path=CIs_save_path+'/CIs_2')
plot_map(CIs[3].reshape(int(CIs[3].shape[0]**0.5), int(CIs[3].shape[0]**0.5), *CIs[3].shape[1:]).detach().numpy(), path=CIs_save_path+'/CIs_3')

save_data = {'FMs': FMs, 'CIs': CIs}
torch.save(save_data, save_path + 'FMs_CIs.pt')


