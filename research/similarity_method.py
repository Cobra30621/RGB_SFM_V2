from math import sqrt
import math


def lab_euclidean_similarity(color1, color2):
    """計算 LAB 顏色空間的歐幾裏德相似度"""
    return 1 / (1 + sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2))) / 255)


def lab_manhattan_similarity(color1, color2):
    """計算 LAB 顏色空間的曼哈頓距離相似度"""
    return 1 / (1 + sum(abs(c1 - c2) for c1, c2 in zip(color1, color2)) / 255)



def lab_distance(color1, color2):
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
