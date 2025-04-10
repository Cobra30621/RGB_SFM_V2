
from torchsummary import summary

from ci_getter import *
from models.SFMCNN import SFMCNN
from models.RGB_SFMCNN import RGB_SFMCNN
from models.RGB_SFMCNN_V2 import RGB_SFMCNN_V2
from dataloader import get_dataloader

def load_model_and_data(checkpoint_filename, test_data = False):
    """
    加載模型和數據

    Args:
        checkpoint_filename (str): 檢查點文件名稱

    Returns:
        tuple: (model, train_dataloader, test_dataloader, images, labels)
    """
    with torch.no_grad():
        # Load Dataset
        train_dataloader, test_dataloader = get_dataloader(
            dataset=config['dataset'],
            root=config['root'] + '/data/',
            batch_size=config['batch_size'],
            input_size=config['input_shape']
        )

        images, labels = torch.tensor([]), torch.tensor([])
        for batch in test_dataloader:
            imgs, lbls = batch
            images = torch.cat((images, imgs))
            labels = torch.cat((labels, lbls))
        print(images.shape, labels.shape)

        # Load Model
        models = {'SFMCNN': SFMCNN, 'RGB_SFMCNN': RGB_SFMCNN, 'RGB_SFMCNN_V2': RGB_SFMCNN_V2}
        checkpoint = torch.load(f'./pth/{config["dataset"]}_pth/{checkpoint_filename}.pth', weights_only=True)
        model = models[arch['name']](**dict(config['model']['args']))
        model.load_state_dict(checkpoint['model_weights'])
        model.cpu()
        model.eval()

        summary(model, input_size=(config['model']['args']['in_channels'], *config['input_shape']), device='cpu')
        print(model)

        # Test Model
        if test_data:
            batch_num = 1000
            pred = model(images[:batch_num])
            y = labels[:batch_num]
            correct = (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            print("Test Accuracy: " + str(correct / len(pred)))

        return model, train_dataloader, test_dataloader, images, labels