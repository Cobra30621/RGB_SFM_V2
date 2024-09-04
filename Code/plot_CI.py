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

'''
	產生FM、CI的可解釋性圖片
'''

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
checkpoint_filename = 'RGB_SFMCNN_best'
checkpoint = torch.load(f'../pth/{config["dataset"]}_pth/{checkpoint_filename}.pth' , weights_only=True)
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

# 讀取每層卷積的weight並轉換成Reshape成矩陣(get FM)
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
	weights = torch.concat([model.RGB_convs[0][0].weights, model.RGB_convs[0][0].black_block, model.RGB_convs[0][0].white_block])
	weights = weights.reshape(arch['args']['channels'][0][0],arch['args']['in_channels'],1,1)
	weights = weights.repeat(1,1,*kernel_size)
	FMs['RGB_convs_0'] = weights
	print(f'FM[RGB_convs_0] shape: {FMs["RGB_convs_0"].shape}')

	FMs['RGB_convs_1'] = model.RGB_convs[2][0].weight.reshape(-1, 5, 6, 1)
	print(f'FM[RGB_convs_1] shape: {FMs["RGB_convs_1"].shape}')

	FMs['RGB_convs_2'] = model.RGB_convs[3][0].weight.reshape(-1, int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), int(model.RGB_convs[3][0].weight.shape[1] ** 0.5), 1)
	print(f'FM[RGB_convs_2] shape: {FMs["RGB_convs_2"].shape}')
	

	FMs['Gray_convs_0'] = model.Gray_convs[0][0].weight.reshape(arch['args']['channels'][1][0],1,*kernel_size)
	print(f'FM[Gray_convs_0] shape: {FMs["Gray_convs_0"].shape}')
	FMs['Gray_convs_1'] = model.Gray_convs[2][0].weight.reshape(-1, 7, 10, 1)
	print(f'FM[Gray_convs_1] shape: {FMs["Gray_convs_1"].shape}')
	FMs['Gray_convs_2'] = model.Gray_convs[3][0].weight.reshape(-1, int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), int(model.Gray_convs[3][0].weight.shape[1] ** 0.5), 1)
	print(f'FM[Gray_convs_2] shape: {FMs["Gray_convs_2"].shape}')


# 讀取每一層架構(為後面的CI做準備)
layers = {}
if arch['args']['in_channels'] == 1:
	layers[0] = nn.Sequential(model.convs[0][:2])
	layers[1] = nn.Sequential(*(list(model.convs[0]) + list([model.convs[1][:2]])))
	layers[2] = nn.Sequential(*(list(model.convs[:2]) + list([model.convs[2][:2]])))
	layers[3] = nn.Sequential(*(list(model.convs[:3]) + list([model.convs[3][:2]])))
else:
	layers['RGB_convs_0'] = model.RGB_convs[0]
	layers['RGB_convs_1'] = nn.Sequential(*(list(model.RGB_convs[:2]) + list([model.RGB_convs[2][:2]])))
	layers['RGB_convs_2'] = nn.Sequential(*(list(model.RGB_convs[:3]) + list([model.RGB_convs[3][:2]])))
	
	layers['Gray_convs_0'] = model.Gray_convs[0]
	layers['Gray_convs_1'] = nn.Sequential(*(list(model.Gray_convs[:2]) + list([model.Gray_convs[2][:2]])))
	layers['Gray_convs_2'] = nn.Sequential(*(list(model.Gray_convs[:3]) + list([model.Gray_convs[3][:2]])))

# 獲得每一層的CI
CIs = {}
kernel_size=arch['args']['Conv2d_kernel'][0]
stride = (arch['args']['strides'][0], arch['args']['strides'][0]) 
if arch['args']['in_channels'] == 1:
	CIs[0], CI_idx, CI_values = get_ci(images, layers[0], kernel_size=kernel_size, stride=stride)
	CIs[1], CI_idx, CI_values = get_ci(images, layers[1], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs[2], CI_idx, CI_values = get_ci(images, layers[2], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	CIs[3], CI_idx, CI_values = get_ci(images, layers[3], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:3]), dim=0))
else:
	CIs["RGB_convs_0"], CI_idx, CI_values = get_ci(images, layers['RGB_convs_0'], kernel_size, stride = stride)
	CIs["RGB_convs_1"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_1"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs["RGB_convs_2"], CI_idx, CI_values = get_ci(images, layers["RGB_convs_2"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))
	
	CIs["Gray_convs_0"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers['Gray_convs_0'], kernel_size, stride = stride)
	CIs["Gray_convs_1"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_1"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:1]), dim=0))
	CIs["Gray_convs_2"], CI_idx, CI_values = get_ci(model.gray_transform(images), layers["Gray_convs_2"], kernel_size=kernel_size, stride=stride, sfm_filter=torch.prod(torch.tensor(arch['args']['SFM_filters'][:2]), dim=0))

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
	plot_map(FMs['RGB_convs_0'].permute(0,2,3,1).reshape(5, 6, *arch['args']['Conv2d_kernel'][0], arch['args']['in_channels']).detach().numpy(), path=FMs_save_path+'/FMs_RGB_convs_0')
	plot_map(FMs['RGB_convs_1'].reshape(int(FMs['RGB_convs_1'].shape[0]**0.5), int(FMs['RGB_convs_1'].shape[0]**0.5), *FMs['RGB_convs_1'].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_RGB_convs_1')
	plot_map(FMs['RGB_convs_2'].reshape(int(FMs['RGB_convs_2'].shape[0]**0.5), int(FMs['RGB_convs_2'].shape[0]**0.5), *FMs['RGB_convs_2'].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_RGB_convs_2')
	
	plot_map(FMs['Gray_convs_0'].permute(0,2,3,1).reshape(7, 10, *arch['args']['Conv2d_kernel'][0], 1).detach().numpy(), path=FMs_save_path+'/FMs_Gray_convs_0')
	plot_map(FMs['Gray_convs_1'].reshape(int(FMs['Gray_convs_1'].shape[0]**0.5), int(FMs['Gray_convs_1'].shape[0]**0.5), *FMs['Gray_convs_1'].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_Gray_convs_1')
	plot_map(FMs['Gray_convs_2'].reshape(int(FMs['Gray_convs_2'].shape[0]**0.5), int(FMs['Gray_convs_2'].shape[0]**0.5), *FMs['Gray_convs_2'].shape[1:]).detach().numpy(), path=FMs_save_path+'/FMs_Gray_convs_2')
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
	plot_map(CIs['RGB_convs_0'].reshape(5, 6, *CIs['RGB_convs_0'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_RGB_convs_0')

	# origin CI代表為原始的CI、沒有origin的CI指的是將CI取平均代表色形成色塊
	plot_map(CIs['RGB_convs_1'].reshape(15, 15, *CIs['RGB_convs_1'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_RGB_convs_1_origin')
	CI = CIs['RGB_convs_1'].detach()
	# 將RM_CI取每個小圖的代表色塊後合併成為新的RM_CI
	CI = CI.reshape(*CI.shape[:2], CI.shape[2] // 5, 5, CI.shape[3]//5, 5, 3)
	CI = CI.permute(0,1,2,4,3,5,6)
	origin_CI_shape = CI.shape
	CI = CI.reshape(*CI.shape[:4], -1, 3).mean(dim=-2).unsqueeze(-2).repeat(1,1,1,1,25,1)
	CI = CI.reshape(*origin_CI_shape[:4], 5, 5, 3)
	CI = CI.permute(0,1,2,4,3,5,6)
	CI = CI.reshape(*CIs['RGB_convs_1'].shape)
	plot_map(CI.reshape(15, 15, *CIs['RGB_convs_1'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_RGB_convs_1')
	plt.imshow(CI.reshape(15, 15, *CIs['RGB_convs_1'].shape[2:]).detach().numpy()[0, 0])
	
	plot_map(CIs['RGB_convs_2'].reshape(int(CIs['RGB_convs_2'].shape[0]**0.5), int(CIs['RGB_convs_2'].shape[0]**0.5), *CIs['RGB_convs_2'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_RGB_convs_2_origin')
	CI = CIs['RGB_convs_2'].detach()
	CI = CI.reshape(*CI.shape[:2], CI.shape[2] // 5, 5, CI.shape[3]//5, 5, 3)
	CI = CI.permute(0,1,2,4,3,5,6)
	origin_CI_shape = CI.shape
	CI = CI.reshape(*CI.shape[:4], -1, 3).mean(dim=-2).unsqueeze(-2).repeat(1,1,1,1,25,1)
	CI = CI.reshape(*origin_CI_shape[:4], 5, 5, 3)
	CI = CI.permute(0,1,2,4,3,5,6)
	CI = CI.reshape(*CIs['RGB_convs_2'].shape)
	plot_map(CIs['RGB_convs_2'].reshape(int(CIs['RGB_convs_2'].shape[0]**0.5), int(CIs['RGB_convs_2'].shape[0]**0.5), *CIs['RGB_convs_2'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_RGB_convs_2')

	plot_map(CIs['Gray_convs_0'].reshape(7, 10, *CIs['Gray_convs_0'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_Gray_convs_0', cmap='gray')
	plot_map(CIs['Gray_convs_1'].reshape(25, 25, *CIs['Gray_convs_1'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_Gray_convs_1', cmap='gray')
	plot_map(CIs['Gray_convs_2'].reshape(int(CIs['Gray_convs_2'].shape[0]**0.5), int(CIs['Gray_convs_2'].shape[0]**0.5), *CIs['Gray_convs_2'].shape[2:]).detach().numpy(), path=CIs_save_path+'/CIs_Gray_convs_2', cmap='gray')

print('CI saved')

