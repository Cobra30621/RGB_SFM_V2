# 匯入所需模組與工具函式
from diabetic_retinopathy_handler import check_then_preprocess_images
from load_tools import load_model_and_data
from plot_graph_method import plot_combine_images, plot_heatmap, plot_map
from ci_getter import *
from typing import Tuple, List, Dict
import torch
import matplotlib

"""
產生 Feature Map（FM）與 Critical Input（CI） 的可視化圖片
"""

# 1️⃣ 讀取模型與資料
checkpoint_filename = config["load_model_name"]
test_data = False # 測試模型準確度
model, train_dataloader, test_dataloader, images, labels = load_model_and_data(checkpoint_filename, test_data=test_data)
mode = arch['args']['mode'] # 模式

# 2️⃣ 建立儲存目錄
save_path = f'./detect/{config["dataset"]}/{checkpoint_filename}/'
FMs_save_path = save_path + 'FMs/'
CIs_save_path = save_path + 'CIs/'
os.makedirs(FMs_save_path, exist_ok=True)
os.makedirs(CIs_save_path, exist_ok=True)

# 3️⃣ 定義：繪製單一層的 FM 圖
def plot_FM(
    conv: torch.nn.Module,
    num_filter: Tuple[int, int],
    filter_shape: Tuple[int, int],
    channel: int,
    path: str,
    layer_name: str
) -> None:
    """
    將卷積層的權重視覺化成特徵圖（Feature Map）並儲存。

    參數:
        conv: 卷積層模組（包含 .weight）
        num_filter: 濾波器排列方式 (rows, cols)
        filter_shape: 濾波器尺寸 (height, width)
        channel: 濾波器的通道數（1 表示灰階，3 表示 RGB）
        path: 輸出圖片儲存資料夾
        layer_name: 圖片檔名用的層名稱
    """
    print(f"conv.weight: {conv.weight.shape}")
    rm = conv.weight.view(*num_filter, *filter_shape, channel).detach().numpy()
    print(f"plot FM {layer_name}, shape: {rm.shape}")
    plot_map(rm, path=path + f'/FMs_{layer_name}')

# 4️⃣ 定義：繪製分支中所有 FM
def plot_FM_branch(
    branch_conv: torch.nn.Sequential,
    first_filter_shape: Tuple[int, int],
    channel_settings: List[Tuple[int, int]],
    layer_name: str,
    path: str,
    is_rgb: bool = False
) -> None:
    """
    繪製指定分支（RGB 或 Gray）的所有卷積層權重特徵圖。

    參數:
        branch_conv: 卷積層序列模組（例如 model.RGB_convs）
        first_filter_shape: 第一層濾波器的 shape
        channel_settings: 每層濾波器的尺寸設定 [(h, w), ...]
        layer_name: 層名稱前綴（例如 "RGB_convs"）
        path: 輸出圖片儲存資料夾
        is_rgb: 是否為 RGB 分支（第一層為 RGB）
    """

    for i in range(len(branch_conv)):
        conv = branch_conv[i][0]
        # RGB 第一層的 channel 為 3, 其他為 0
        channel = 3 if is_rgb and i == 0 else 1
        num_filter = channel_settings[i]
        # 第一層的 filter_shape 為 first_filter_shape，其餘為前一層的 channel_settings
        filter_shape = first_filter_shape if i == 0 else channel_settings[i - 1]
        plot_FM(conv, num_filter, filter_shape, channel, path, f"{layer_name}_{i}")

# 5️⃣ 儲存 Feature Maps
print('FM saving ...')
channels = arch['args']['channels']                # [RGB通道設定, Gray通道設定]
kernel_size = arch['args']['Conv2d_kernel'][0]     # 第一層 Gray 的 num_filter (如 70)


# FM 只有在第一層後 kernal 都是 1 ，才能運作
# plot_FM_branch(model.RGB_convs,  kernel_size, channels[0], "RGB_convs", FMs_save_path, is_rgb=True)
# plot_FM_branch(model.Gray_convs, kernel_size, channels[1], "Gray_convs", FMs_save_path)
# print('FM saved.')


# 7️⃣ 根據資料集決定是否進行預處理（如視網膜影像）
images = check_then_preprocess_images(images)

# 8️⃣ 計算各層的 CI 與對應值
force_regenerate=False
CIs, CI_values = load_or_generate_CIs(model, images, force_regenerate=force_regenerate, save_path= f'./detect/{config["dataset"]}/{checkpoint_filename}')


# 9️⃣ 定義：繪製單層 CI
def plot_CI(
    CI: torch.Tensor,
    channel: Tuple[int, int],
    path: str,
    layer_name: str
) -> "matplotlib.figure.Figure":
    """
    將 Critical Input (CI) reshape 成可視化格式並繪圖。

    參數:
        CI: 該層所有濾波器的 CI tensor，shape=[n, h, w, c]
        channel: 該層的濾波器排列方式 (rows, cols)
        path: 儲存圖片的資料夾
        layer_name: 圖片檔名用的層名稱

    回傳:
        fig: matplotlib 繪製完成的圖像對象
    """

    reshape_CI = CI.reshape(channel[0], channel[1], *CI.shape[2:]).detach().numpy()
    fig = plot_map(reshape_CI, path=path + f'/CIs_{layer_name}')
    return fig

# 🔟 定義：繪製整個分支的 CI（含熱圖與合併圖）
def plot_CI_branch(
    CIs: Dict[str, torch.Tensor],
    CI_values: Dict[str, torch.Tensor],
    layer_count: int,
    channels: List[Tuple[int, int]],
    branch_name: str,
    path: str
) -> None:
    """
    繪製整個分支的 CI 圖與對應的 CI 激活值熱圖，並合併展示。

    參數:
        CIs: 每層的 CI tensor，key 為層名
        CI_values: 每層的 CI 最大值 tensor，key 為層名
        layer_count: 該分支的層數
        channels: 每層濾波器排列設定 [(rows, cols), ...]
        branch_name: 分支名稱（如 RGB_convs）
        path: 輸出圖片的儲存資料夾
    """

    CI_figs = {}
    for layer_index in range(layer_count):
        layer_name = f"{branch_name}_{layer_index}"
        CI = CIs[layer_name]
        channel = channels[layer_index]
        print(f"plot CI {layer_name}, CIs: {CI.shape}, channel: {channel}")

        # 畫 CI 圖
        CI_fig = plot_CI(CI, channel, path, layer_name)
        CI_figs[layer_name] = CI_fig

        # 畫熱圖
        heatmap_fig = plot_heatmap(CI_values[layer_name], f"{path}/CI_values_{layer_name}", *channel)
        CI_figs[f"{layer_name}_heatmap"] = heatmap_fig

    # 將所有圖合併輸出
    plot_combine_images(CI_figs, path + f'/{branch_name}_combine')


layer_count = len(arch["args"]["Conv2d_kernel"])
print(f"layer count {layer_count}")
if mode in ['rgb', 'both']:
    plot_CI_branch(CIs, CI_values, layer_count, channels[0], "RGB_convs", CIs_save_path)

if mode in ['gray', 'both']:
    plot_CI_branch(CIs, CI_values, layer_count, channels[1], "Gray_convs", CIs_save_path)
print('CI saved.')
