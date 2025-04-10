from config import *
from ci_getter import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from dataloader import get_dataloader
from collections import defaultdict

'''
    把圖片分割成 5*5 的小圖並且塗上的顏色形成一張圖片中有不同顏色的情況
'''
def color_image_with_66Blocks(image):
    image = image.reshape(28,28)
    global i
    # 将图像转换为Pillow图像对象
    img = Image.fromarray(image.astype(np.uint8), 'L')

    # 使用填充将图像扩展到30x30
    img_padded = ImageOps.expand(img, border=1, fill='black')

    # 将图像分割成 5x5 的小图块
    img_width, img_height = img_padded.size
    block_size = 5
    blocks = []

    for y in range(0, img_height, block_size):
        for x in range(0, img_width, block_size):
            block = img_padded.crop((x, y, x + block_size, y + block_size))
            blocks.append((x, y, block))

    # 随机为每个小图块涂上红色、蓝色或绿色
    colored_blocks = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # 红色，绿色，蓝色

    for (x, y, block) in blocks:
        color = random.choice(colors)
        color_block = ImageOps.colorize(block, black="black", white=color)
        colored_blocks.append((x, y, color_block))

    # 将小图块合并回原始图像大小
    new_img = Image.new('RGB', (img_width, img_height))

    for (x, y, color_block) in colored_blocks:
        new_img.paste(color_block, (x, y))

    # 移除填充行和列，恢复图像到28x28大小
    final_img = new_img.crop((1, 1, 29, 29))

    i+=1

    print(i)

    print(np.array(final_img).shape)
    plt.imshow(np.array(final_img))
    plt.show()
    return np.array(final_img)

'''
    把圖片的每個piexl隨機塗上30色的其中一個顏色形成一張圖片中有不同顏色的情況
'''
def color_image_with_pixels(image):
    weights = [[185, 31, 87], 
                    [208, 47, 72],
                    [221, 68, 59],
                    [233, 91, 35],
                    [230, 120, 0],
                    [244, 157, 0],
                    [241, 181, 0],
                    [238, 201, 0],
                    [210, 193, 0],
                    [168, 187, 0],
                    [88, 169, 29],
                    [0, 161, 90],
                    [0, 146, 110],
                    [0, 133, 127],
                    [0, 116, 136],
                    [0, 112, 155],
                    [0, 96, 156],
                    [0, 91, 165],
                    [26, 84, 165],
                    [83, 74, 160],
                    [112, 63, 150],
                    [129, 55, 138],
                    [143, 46, 124],
                    [173, 46, 108],
                    [255, 0, 0],
                    [0, 255, 0],
                    [0, 0, 255],
                    [128, 128, 128]]


    # 确保图像是numpy array
    image = np.array(image).reshape(28, 28)

    # 创建一个空的彩色图像数组
    colored_image = np.zeros((28, 28, 3), dtype=np.uint8)

    # 遍历每个像素
    for i in range(28):
        for j in range(28):
            if image[i, j] > 0:  # 如果是白色部分
                color = random.choice(weights)
                for c in range(3):
                    colored_image[i, j, c] = image[i, j] * color[c]
            else:  # 如果是黑色部分
                colored_image[i, j] = [0, 0, 0]
    return colored_image

def color_image_fixed_color(image, color):
    """
    将MNIST图像的白色部分涂上固定颜色。

    参数:
    image (numpy array): MNIST图像数组，大小为28x28，像素值范围为0到255。
    color (tuple): 颜色的RGB值。

    返回:
    numpy array: 彩色图像数组，大小为28x28x3。
    """
    # 确保图像是numpy array
    image = np.array(image)

    # 创建一个全零的彩色图像数组
    img_colored = np.zeros((*image.shape, 3), dtype=np.uint8)

    # 将灰度图像的白色部分乘以对应的RGB值并除以255
    for c in range(3):
        img_colored[..., c] = image * color[c]  # 将灰度值转换为相应的 RGB 值

    img_colored = img_colored.reshape(28, 28, 3)

    return img_colored

def create_balanced_dataset(original_images):
    """
    创建一个包含31个类的平衡数据集，每个类的图像数量相同。

    参数:
    mnist_dataset: MNIST数据集。

    返回:
    List: 31个类的图像和标签。
    """
    # 确定每个类别的图像数量，取最小值以平衡数据集
    min_count = min(len(images) for images in original_images.values())
    num_per_class = sum(len(images) for images in original_images.values()) // 31
    
    # 创建平衡的数据集
    dataset = []
    
    for label in range(10):
        print(f'label:{label}')
        # plt.imshow(original_images[label][0].reshape(28,28), cmap='gray')
        # plt.show()
        # plt.imshow(color_image_fixed_color(original_images[label][0], (0, 0, 255)))
        # plt.show()
        plt.imshow(color_image_with_pixels(original_images[label][0]))
        plt.show()

        # 红色图像
        dataset += [(color_image_fixed_color(img, (255, 0, 0)), label) for img in original_images[label][:num_per_class]]
        # 绿色图像
        dataset += [(color_image_fixed_color(img, (0, 255, 0)), label + 10) for img in original_images[label][num_per_class:num_per_class*2]]
        # 蓝色图像
        dataset += [(color_image_fixed_color(img, (0, 0, 255)), label + 20) for img in original_images[label][num_per_class*2:num_per_class*3]]
        # 使用color_image_with_pixels函数着色的图像
        dataset += [(color_image_with_pixels(img), 30) for img in original_images[label][num_per_class*3:]]

    return dataset