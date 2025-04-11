from config import arch
from monitor.calculate_stats import calculate_RM, get_stats
from monitor.metrics import calculate_layer_metrics
import torch

def get_layer_stats(model, layer, images, is_gray=False):
    """
    獲取指定層的統計數據。

    參數:
    model: 用於處理圖像的模型。
    layers: 模型的層列表。
    layer_num: 要獲取統計數據的層的編號。
    images: 輸入的圖像數據。
    is_gray: 是否將圖像轉換為灰度圖像的布林值，預設為 False。

    返回:
    包含計算指標的字典。
    """
    if is_gray:
        input_images = model.gray_transform(images)
    else:
        input_images = images

    raw = calculate_RM(layer, input_images)
    channel_stats, overall_stats = get_stats(raw)

    # 計算指標
    metrics_results = calculate_layer_metrics({**overall_stats, **channel_stats})

    return {**metrics_results}


def get_all_layers_stats(model, rgb_layers, gray_layers, images, keep_tensor=False, without_RGBConv0 = False):
    """
    獲取所有層的統計數據。

    參數:
    model: 用於處理圖像的模型。
    layers: 模型的層列表。
    images: 輸入的圖像數據。
    keep_tensor: 若為 False，回傳結果中的 tensor 會轉換成 list。

    返回:
    包含所有層統計數據的字典，以及每層統計數據的平均值。
    """
    layer_stats = {}

    use_gray = arch['args']['use_gray']  # 使否使用輪廓層
    for key, layer in rgb_layers.items():
        # 可以跳過 RGBConv0 (由於其中的參數不需要訓練)
        if without_RGBConv0 and key == "RGB_convs_0":
            continue
        layer_stats[key] = get_layer_stats(model, layer, images, is_gray=False)

    if use_gray:
        for key, layer in gray_layers.items():
            layer_stats[key] = get_layer_stats(model, layer, images, is_gray=True)

    overall_stats = {}
    for key in layer_stats[next(iter(layer_stats))]:
        overall_stats[key] = torch.stack([layer_stat[key] for layer_stat in layer_stats.values()]).mean()

    if not keep_tensor:
        # 遍歷並轉換 tensor 為 list
        for layer_num, stats in layer_stats.items():
            for key, layer in stats.items():
                if isinstance(layer, torch.Tensor):
                    layer_stats[layer_num][key] = layer.tolist()

        for key, layer in overall_stats.items():
            if isinstance(layer, torch.Tensor):
                overall_stats[key] = layer.tolist()

    return layer_stats, overall_stats
