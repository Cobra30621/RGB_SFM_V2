from monitor.calculate_stats import calculate_RM, get_stats
from monitor.metrics import calculate_layer_metrics


def get_layer_stats(model, layers, layer_num, images, is_gray=False):
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

    raw = calculate_RM(layers, layer_num, input_images)
    channel_stats, overall_stats = get_stats(raw)

    # 計算指標
    metrics_results = calculate_layer_metrics({**overall_stats, **channel_stats})

    return {**metrics_results}


def get_all_layers_stats(model, layers, layers_infos, images):
    """
    獲取所有層的統計數據。

    參數:
    model: 用於處理圖像的模型。
    layers: 模型的層列表。
    layers_infos: 包含每層信息的列表，每個信息包含層編號和是否為灰度圖像的標記。
    images: 輸入的圖像數據。

    返回:
    包含所有層統計數據的字典，以及每層統計數據的平均值。
    """
    # 使用示例
    layer_stats = {}
    for layer_info in layers_infos:
        layer_num = layer_info["layer_num"]
        is_gray = layer_info["is_gray"]

        layer_stats[layer_num] = get_layer_stats(model, layers, layer_num, images, is_gray)

    # 計算 overall_stats 為 layer_stats 的平均值
    overall_stats = {}
    for key in layer_stats[next(iter(layer_stats))]:  # 取第一個層的鍵
        overall_stats[key] = sum(layer_stat[key] for layer_stat in layer_stats.values()) / len(layer_stats)

    return layer_stats, overall_stats