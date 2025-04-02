from diabetic_retinopathy_handler import preprocess_retinal_tensor_batch, check_then_preprocess_images
from load_tools import load_model_and_data
from models.RGB_SFMCNN_V2 import  get_CI_target_layers
from utils import *

'''
	產生FM、CI的可解釋性圖片
'''

# 讀取模型與資料
checkpoint_filename = 'RGB_SFMCNN_V2_best'
model, train_dataloader, test_dataloader, images, labels = load_model_and_data(checkpoint_filename)


save_path = f'./detect/{config["dataset"]}_{checkpoint_filename}/'


# 特徵圖儲存子資料夾
FMs_save_path = save_path + 'FMs/'
os.makedirs(FMs_save_path, exist_ok=True)  # 若資料夾不存在則建立

# 定義繪製單層特徵圖的方法
def plot_FM(conv, num_filter, filter_shape, channel, path, layer_name):
    """
    將指定卷積層的 weight tensor reshape 成 5 維張量後可視化並儲存。

    參數:
        conv: 該層的卷積層
        num_filter: tuple，表示要重塑成的濾波器排版數 (rows, cols)
        filter_shape: tuple，表示每個濾波器的尺寸 (height, width)
        channel: 通道數，1 表示灰階，3 表示 RGB
        path: 儲存圖片的路徑
        layer_name: 圖片命名用的圖層名稱
    """
    # 將 weight 轉成指定形狀後轉成 numpy 格式
    rm = conv.weight.view(*num_filter, *filter_shape, channel).detach().numpy()
    print(f"plot FM {layer_name}, shape: { rm.shape}")

    # 呼叫視覺化函式
    plot_map(rm, path=path + f'/FMs_{layer_name}')


# 定義共用邏輯的分支處理函式（RGB 或 Gray 分支）
def process_branch(branch_conv, first_num_filter, channel_settings, layer_name, path, is_rgb=False):
    """
    根據指定分支的卷積層，依序處理三層並可視化其特徵圖。

    參數:
        branch_conv: 該分支（RGB 或 Gray）的卷積層序列
        first_num_filter: 第一層的
        channel_settings: 對應各層的通道設定 (來自 arch['args']['channels'])
        layer_name: 用於儲存時的圖層命名前綴
        path: 圖像儲存路徑
        is_rgb: 是否為 RGB 分支（用來處理第一層特殊處理）
    """
    for i in range(len(branch_conv)):
        conv = branch_conv[i][0]  # 取出每層的卷積層模組

        # 特殊處理 RGB 第一層
        if is_rgb and i == 0:
            channel = 3
        else:
            channel = 1

        # 特殊處理 第一層
        if i == 0:
            num_filter = first_num_filter
        else:
            num_filter = channel_settings[i - 1]

        filter_shape = channel_settings[i]

        # 呼叫視覺化函式
        plot_FM(conv, num_filter, filter_shape, channel, path, f"{layer_name}_{i}")


print('FM saving')

# 取得通道設定與 kernel size
channels = arch['args']['channels']  # [RGB_channels, Gray_channels]
kernel_size = arch['args']['Conv2d_kernel'][0]  # 用於第一層 gray conv 的濾波器數量

# 執行 RGB 分支的特徵圖儲存
process_branch(model.RGB_convs, (1,1), channels[0], "RGB_convs", FMs_save_path, is_rgb=True)

# 執行灰階分支的特徵圖儲存
process_branch(model.Gray_convs, kernel_size, channels[1], "Gray_convs", FMs_save_path)

print('FM saved')


# 讀取每一層架構(為後面的CI做準備)
rgb_layers, gray_layers = get_CI_target_layers(model)

print(f"rgb_layers {rgb_layers}")


# 使用影像前處理，如果有需要(視網膜資料集)
images = check_then_preprocess_images(images)

# 獲得每一層的CIs
CIs, CI_values = get_CIs(model, rgb_layers, gray_layers, images)


print('CI saving ...')
CIs_save_path = save_path + 'CIs/'
os.makedirs(CIs_save_path, exist_ok=True)

RGB_CI_figs = {}


RGB_CI_figs['RGB_convs_0'] = plot_map(CIs['RGB_convs_0'].reshape(5, 6, *CIs['RGB_convs_0'].shape[2:]).detach().numpy(),
                                      path=CIs_save_path + '/CIs_RGB_convs_0')

RGB_CI_figs['RGB_convs_0_heatmap'] = plot_heatmap(CI_values['RGB_convs_0'], CIs_save_path + '/CI_values_RGB_convs_0', 6, 5)

# origin CI代表為原始的CI、沒有origin的CI指的是將CI取平均代表色形成色塊
RGB_CI_figs['RGB_convs_1'] = plot_map(CIs['RGB_convs_1'].reshape(15, 15, *CIs['RGB_convs_1'].shape[2:]).detach().numpy(),
                                      path=CIs_save_path + '/CIs_RGB_convs_1_origin')
CI = CIs['RGB_convs_1'].detach()
# 將RM_CI取每個小圖的代表色塊後合併成為新的RM_CI
CI = CI.reshape(*CI.shape[:2], CI.shape[2] // 5, 5, CI.shape[3] // 5, 5, 3)
CI = CI.permute(0, 1, 2, 4, 3, 5, 6)
origin_CI_shape = CI.shape
CI = CI.reshape(*CI.shape[:4], -1, 3).mean(dim=-2).unsqueeze(-2).repeat(1, 1, 1, 1, 25, 1)
CI = CI.reshape(*origin_CI_shape[:4], 5, 5, 3)
CI = CI.permute(0, 1, 2, 4, 3, 5, 6)
CI = CI.reshape(*CIs['RGB_convs_1'].shape)
plot_map(CI.reshape(15, 15, *CIs['RGB_convs_1'].shape[2:]).detach().numpy(),
         path=CIs_save_path + '/CIs_RGB_convs_1')
plt.imshow(CI.reshape(15, 15, *CIs['RGB_convs_1'].shape[2:]).detach().numpy()[0, 0])

RGB_CI_figs['RGB_convs_1_heatmap'] = plot_heatmap(CI_values['RGB_convs_1'], CIs_save_path + '/CI_values_RGB_convs_1')

RGB_CI_figs['RGB_convs_2'] = plot_map(
    CIs['RGB_convs_2'].reshape(int(CIs['RGB_convs_2'].shape[0] ** 0.5), int(CIs['RGB_convs_2'].shape[0] ** 0.5),
                               *CIs['RGB_convs_2'].shape[2:]).detach().numpy(),
    path=CIs_save_path + '/CIs_RGB_convs_2_origin')
CI = CIs['RGB_convs_2'].detach()
CI = CI.reshape(*CI.shape[:2], CI.shape[2] // 5, 5, CI.shape[3] // 5, 5, 3)
CI = CI.permute(0, 1, 2, 4, 3, 5, 6)
origin_CI_shape = CI.shape
CI = CI.reshape(*CI.shape[:4], -1, 3).mean(dim=-2).unsqueeze(-2).repeat(1, 1, 1, 1, 25, 1)
CI = CI.reshape(*origin_CI_shape[:4], 5, 5, 3)
CI = CI.permute(0, 1, 2, 4, 3, 5, 6)
CI = CI.reshape(*CIs['RGB_convs_2'].shape)
plot_map(
    CIs['RGB_convs_2'].reshape(int(CIs['RGB_convs_2'].shape[0] ** 0.5), int(CIs['RGB_convs_2'].shape[0] ** 0.5),
                               *CIs['RGB_convs_2'].shape[2:]).detach().numpy(),
    path=CIs_save_path + '/CIs_RGB_convs_2')

RGB_CI_figs['RGB_convs_2_heatmap'] = plot_heatmap(CI_values['RGB_convs_2'], CIs_save_path + '/CI_values_RGB_convs_2', 25, 25)

plot_combine_images(RGB_CI_figs, CIs_save_path + '/RGB_combine')


plot_map(CIs['Gray_convs_0'].reshape(7, 10, *CIs['Gray_convs_0'].shape[2:]).detach().numpy(),
         path=CIs_save_path + '/CIs_Gray_convs_0', cmap='gray')
plot_heatmap(CI_values['Gray_convs_0'], CIs_save_path + '/CI_values_Gray_convs_0', 7, 10)
plot_map(CIs['Gray_convs_1'].reshape(25, 25, *CIs['Gray_convs_1'].shape[2:]).detach().numpy(),
         path=CIs_save_path + '/CIs_Gray_convs_1', cmap='gray')
plot_heatmap(CI_values['Gray_convs_1'], CIs_save_path + '/CI_values_Gray_convs_1', 25, 25)
plot_map(
    CIs['Gray_convs_2'].reshape(int(CIs['Gray_convs_2'].shape[0] ** 0.5), int(CIs['Gray_convs_2'].shape[0] ** 0.5),
                                *CIs['Gray_convs_2'].shape[2:]).detach().numpy(),
    path=CIs_save_path + '/CIs_Gray_convs_2', cmap='gray')

plot_heatmap(CI_values['Gray_convs_2'], CIs_save_path + '/CI_values_Gray_convs_2', 35, 35)

print('CI saved')

