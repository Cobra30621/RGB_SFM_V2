from math import sqrt
import math
from skimage import color
from scipy.spatial.distance import euclidean


def lab_euclidean_similarity(color1, color2):
    """計算 LAB 顏色空間的歐幾里得相似度"""
    return 1 / (1 + sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2))) / 255)


def lab_manhattan_similarity(color1, color2):
    """計算 LAB 顏色空間的曼哈頓距離相似度"""
    return 1 / (1 + sum(abs(c1 - c2) for c1, c2 in zip(color1, color2)) / 255)


def lab_distance(color1, color2):
    """計算顏色在 LAB 顏色空間的距離"""
    # 將顏色轉換為浮點數
    R_1, G_1, B_1 = float(color1[0]), float(color1[1]), float(color1[2])
    R_2, G_2, B_2 = float(color2[0]), float(color2[1]), float(color2[2])

    # 計算 rmean
    rmean = (R_1 + R_2) / 2

    # 計算 RGB 差異
    delta_R = R_1 - R_2
    delta_G = G_1 - G_2
    delta_B = B_1 - B_2

    # 計算 LAB 距離
    lab_distance = math.sqrt(
        (2 + rmean / 256) * (delta_R ** 2) +
        4 * (delta_G ** 2) +
        (2 + (255 - rmean) / 256) * (delta_B ** 2)
    )

    return lab_distance / 765


def lab_cieluv_similarity(color1, color2):
    """計算 CIELUV 顏色空間的相似度"""
    # 將 RGB 顏色轉換為 [0, 1] 範圍
    color1 = [c / 255.0 for c in color1]
    color2 = [c / 255.0 for c in color2]

    # 將 RGB 顏色轉換為 CIELUV 顏色空間
    color1_cieluv = color.rgb2luv([[color1]])[0][0]
    color2_cieluv = color.rgb2luv([[color2]])[0][0]

    # 計算 CIELUV 空間的歐幾里得距離
    distance = euclidean(color1_cieluv, color2_cieluv)
    return distance


def lab_delta_e_similarity(color1, color2):
    """計算 CIELAB 顏色空間的 Delta E 相似度"""
    # 將 RGB 顏色轉換為 [0, 1] 範圍
    color1 = [c / 255.0 for c in color1]
    color2 = [c / 255.0 for c in color2]

    # 將 RGB 顏色轉換為 CIELAB 顏色空間
    color1_cielab = color.rgb2lab([[color1]])[0][0]
    color2_cielab = color.rgb2lab([[color2]])[0][0]

    print(f"1: {color1} 2: {color2}")

    # 計算 Delta E*00
    delta_e = euclidean(color1_cielab, color2_cielab)
    return  delta_e


# 測試範例顏色
color1 = (0, 0, 0)  # 紅色
color2 = (255, 255, 255)  # 綠色

print("LAB 歐幾里得相似度:", lab_euclidean_similarity(color1, color2))
print("LAB 曼哈頓距離相似度:", lab_manhattan_similarity(color1, color2))
print("LAB 距離:", lab_distance(color1, color2))
print("CIELUV 相似度:", lab_cieluv_similarity(color1, color2))
print("Delta E 相似度:", lab_delta_e_similarity(color1, color2))
