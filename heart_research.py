import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchsummary import summary
import os
from torchvision.utils import make_grid
import numpy as np

from config import config, arch
from dataloader import get_dataloader
from dataloader.HeartCalcificationWithoutSplit import HeartCalcificationWithoutSplit
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN

def load_model():
    models = {'SFMCNN': SFMCNN, 'RGB_SFMCNN':RGB_SFMCNN}
    checkpoint_filename = 'SFMCNN_best'
    checkpoint = torch.load(f'./pth/{config["dataset"]}_pth/{checkpoint_filename}.pth' , weights_only=True)
    model = models[arch['name']](**dict(config['model']['args']))
    model.load_state_dict(checkpoint['model_weights'])
    model.cpu()
    model.eval()
    summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']), device='cpu')
    print(model)
    return model

def load_dataloader():
    return get_dataloader(dataset='HeartCalcification_Gray', root=config['root'] + '/data/',
                          batch_size=config['batch_size'], input_size=config['input_shape'])

def predict(model, dataloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())
    return all_preds, all_labels

def plot_confusion_matrix(pred_labels, true_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def visualize_image(img, label, save_path=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(img)

    height, width = img.shape[:2]
    num_blocks_h, num_blocks_w = label.shape

    # 繪製網格線
    for i in range(1, num_blocks_h):
        plt.axhline(y=i * height / num_blocks_h, color='w', linestyle='-', linewidth=1)
    for j in range(1, num_blocks_w):
        plt.axvline(x=j * width / num_blocks_w, color='w', linestyle='-', linewidth=1)

    # 在標籤為真的格子中繪製 'O'
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            if label[i, j] == 1:
                plt.text(j * width / num_blocks_w + width / (2 * num_blocks_w),
                         i * height / num_blocks_h + height / (2 * num_blocks_h), 'O',
                         color='r', fontsize=12, ha='center', va='center')

    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        print(f"圖像已保存到: {save_path}")
    else:
        plt.show()

    plt.close()

def visualize_and_save_images(dataset, labels, save_dir, label_type='true'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, (img, label) in enumerate(dataset):
        save_path = f'data/visual_45/output_image_{idx}.png'
        visualize_image(img, label, save_path=save_path)

    for idx in range(len(dataset.images)):
        img = dataset.images[idx]
        label = labels[idx]

        save_path = os.path.join(save_dir, f'output_image_{idx}_{label_type}_label.png')
        visualize_image(img, label, save_path)


# 主程序
model = load_model()
_, test_dataloader = load_dataloader()
pred_labels, true_labels = predict(model, test_dataloader)
plot_confusion_matrix(pred_labels, true_labels)
#
# dataset = HeartCalcificationWithoutSplit('data/HeartCalcification/test', grid_size=45)
#
#
# # 添加新的可視化步驟
#  # 可視化並保存圖像，使用真實標籤
# save_dir = 'D:\Paper\Cardiac calcification\output_image\weight_15_true'
# visualize_and_save_images(test_dataloader, true_labels, save_dir, label_type='true')
#
# # 可視化並保存圖像，使用預測標籤
# save_dir = 'D:\Paper\Cardiac calcification\output_image\weight_15_pred'
# visualize_and_save_images(test_dataloader, pred_labels, save_dir, label_type='pred')