from torchvision.io import read_image, ImageReadMode
from mlxtend.data import loadlocal_mnist
from torchvision import transforms
import numpy as np

def custom_sort(filename):
    # 从文件名中提取圈圈顏色和數字部分作为排序依据
    parts = filename.split('_')
    return (parts[0], int(parts[1].split('.')[0]))

def load_data(dataset_name):
    if dataset_name == 'face_dataset':
        image_folder = r'./data/face_dataset/Train'
        for root, dirs, files in os.walk(image_folder):
            print(files)
            for i, name in enumerate(sorted(files, key=custom_sort)):
                image = read_image(os.path.join(root, name), ImageReadMode.GRAY)
                image = image / 255
                image = transforms.Resize((60,60))(image)
                images.append(image)
                image_paths.append(os.path.join(root, name))
                if name[:4] == 'face':
                    labels.append(0)
                else:
                    labels.append(1)
        images = torch.tensor(np.array(images))
        labels = torch.tensor(np.eye(2)[labels])
        indices = torch.randperm(5000)
        images=images[indices]
        labels=labels[indices]

    elif dataset_name == 'mnist':
        images, labels = loadlocal_mnist('./data/MNIST/t10k-images.idx3-ubyte', './data/MNIST/t10k-labels.idx1-ubyte')
        images = images.reshape(-1, 1, 28, 28)
        images = torch.tensor(images/255).float()
        labels = torch.tensor(np.eye(10)[labels])

    return images, labels