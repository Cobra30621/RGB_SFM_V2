import math
import torch.nn.functional as F

from config import *

import os
from functools import reduce
import operator
from models.RGB_SFMCNN_V2 import get_basic_target_layers



def multiply_tuples(a, b):
    return tuple(operator.mul(x, y) for x, y in zip(a, b))

def get_accumulated_kernel(kernels, end_idx):
    return reduce(multiply_tuples, kernels[:end_idx + 1])

def get_accumulated_stride(strides, end_idx):
    return reduce(operator.mul, strides[:end_idx + 1], 1)

def split(input, kernel_size=(5, 5), stride=(5, 5)):
    batch, channel, h, w = input.shape
    output_height = math.floor((h - (kernel_size[0] - 1) - 1) / stride[0] + 1)
    output_width = math.floor((w - (kernel_size[1] - 1) - 1) / stride[1] + 1)
    input = input.clone().detach()

    segments = F.unfold(input, kernel_size=kernel_size, stride=stride)
    segments = segments.reshape(batch, channel, *kernel_size, -1).permute(0, 1, 4, 2, 3)
    segments = segments.reshape(batch, channel, output_height, output_width, *kernel_size)
    return segments


def get_ci(input, layer, kernel_size=(5, 5), stride=(5, 5), sfm_filter=(1, 1)):
    segments = split(input, kernel_size, stride)
    combine_h = segments.shape[2] // sfm_filter[0]
    combine_w = segments.shape[3] // sfm_filter[1]
    ci_h = segments.shape[4] * sfm_filter[0]
    ci_w = segments.shape[5] * sfm_filter[1]

    segments = segments.reshape(-1, input.shape[1], combine_h, sfm_filter[0],
                                 combine_w, sfm_filter[1], segments.shape[4], segments.shape[5])
    segments = segments.permute(0, 2, 4, 3, 6, 5, 7, 1)
    segments = segments.reshape(-1, ci_h, ci_w, input.shape[1])

    print(f"combine_h, combine_w, ci_h, ci_w: {combine_h}, {combine_w}, {ci_h}, {ci_w}")
    print(f"segments shape: {segments.shape}")

    with torch.no_grad():
        outputs = layer(input)
        n_filters = outputs.shape[1]
        outputs = outputs.permute(0, 2, 3, 1).reshape(-1, n_filters)

    values, indices = torch.topk(outputs, k=1, dim=0)
    indices = indices.squeeze(0)
    values = values.squeeze(0)

    CI = segments[indices]
    CI = CI.unsqueeze(1)
    CI_values = values.unsqueeze(1)

    for i in range(CI_values.shape[0]):
        if CI_values[i][0] == 0:
            CI[i] = 1

    return CI, indices.unsqueeze(1), CI_values

def get_CIs(model, images):
    # 使否使用輪廓層
    mode = arch['args']['mode']  # 模式
    use_gray = mode in ['gray', 'both']

    rgb_layers, gray_layers = get_basic_target_layers(model, use_gray)
    CIs, CI_values = {}, {}

    mode_layers = [("RGB_convs", rgb_layers)]
    if use_gray:
        mode_layers.append(("Gray_convs", gray_layers))

    for mode, layers in mode_layers:
        for idx, (layer_name, layer) in enumerate(layers.items()):
            full_layer_name = f"{mode}_{idx}"
            print(full_layer_name)

            kernel_size = get_accumulated_kernel(arch['args']['Conv2d_kernel'], idx)
            stride_acc = get_accumulated_stride(arch['args']['strides'], idx)
            stride = (stride_acc, stride_acc)

            sfm_filter = torch.prod(torch.tensor(arch['args']['SFM_filters'][:idx]), dim=0) if idx > 0 else (1, 1)
            input_img = images if "RGB" in mode else model.gray_transform(images)

            CIs[full_layer_name], CI_idx, CI_values[full_layer_name] = get_ci(
                input_img, layer, kernel_size, stride, sfm_filter
            )

    return CIs, CI_values





def load_or_generate_CIs(model, images, force_regenerate=False, save_path = "cache"):
    cache_dir = f'{save_path}/cache'
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = f"{cache_dir}/cis.pt"



    if os.path.exists(cache_path) and not force_regenerate:
        print("Loading cached CIs...")
        return torch.load(cache_path)
    else:
        print("Generating CIs...")
        CIs, CI_values = get_CIs(model, images)
        torch.save((CIs, CI_values), cache_path)
        return CIs, CI_values
