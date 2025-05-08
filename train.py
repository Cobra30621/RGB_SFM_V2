import shutil

import wandb
import numpy as np

from torch import optim, nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torchsummary import summary

from dataloader import get_dataloader
from config import *
import models
from diabetic_retinopathy_handler import preprocess_retinal_tensor_image, preprocess_retinal_tensor_batch
from file_tools import increment_path
from loss.loss_function import get_loss_function, MetricBaseLoss
from models.RGB_SFMCNN_V2 import get_feature_extraction_layers, get_basic_target_layers
from monitor.monitor_method import get_all_layers_stats



def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: nn.Module, eval_loss_fn, optimizer, scheduler, epoch, device,
          training_loss_fn, use_metric_based_loss=False):
    # best_valid_loss = float('inf')
    best_valid_acc = 0
    best_valid_loss =  float('inf')
    count = 0
    patience = config['patience']
    # 使用影像前處理
    use_preprocessed_image= config['use_preprocessed_image']
    checkpoint = {}

    # 需要計算 RM 分布指標
    need_calculate_status = arch["need_calculate_status"]
    if need_calculate_status:
        # 使否使用輪廓層
        mode = arch['args']['mode']
        use_gray = mode in ['gray', 'both']
        rgb_layers, gray_layers = get_basic_target_layers(model, use_gray=use_gray)


    with torch.autograd.set_detect_anomaly(True):
        for e in range(epoch):
            print(f"------------------------------EPOCH {e}------------------------------")
            model.train()
            progress = tqdm(enumerate(train_dataloader), desc="Loss: ", total=len(train_dataloader))
            losses = 0
            correct = 0
            size = 0
            X, y = next(iter(train_dataloader))
            for batch, (X, y) in progress:
                X = X.to(device); y= y.to(device)
                print(f"X.shape: {X.shape}")
                # 使用影像愈處理
                if use_preprocessed_image:
                    X = preprocess_retinal_tensor_batch(X, final_size=config['input_shape'])

                print(f"X.shape 2: {X.shape}")
                pred = model(X)
                print(f"pred.shape: {pred.shape}, y.shape: {y.shape}")
                # 判斷是否使用 metric-based loss
                if use_metric_based_loss:
                    loss = training_loss_fn(pred, y, model, rgb_layers, gray_layers, X)
                else:
                    loss = eval_loss_fn(pred, y)

                # 反向传播
                loss.backward()
                # 更新模型参数
                optimizer.step()
                    
                # 清零梯度
                optimizer.zero_grad()
                
                losses                  += loss.detach().item()
                size += len(X)
                
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                train_loss = losses/(batch+1)
                train_acc = correct/size
                progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(train_loss, train_acc))

            valid_acc, valid_loss, _ = eval(valid_dataloader, model, eval_loss_fn, False, device = device, use_preprocessed_image=use_preprocessed_image)
            print(f"Test Loss: {valid_loss}, Test Accuracy: {valid_acc}")



            if scheduler:
                scheduler.step(valid_loss)

            metrics = {
                "train/loss": train_loss,
                "train/epoch": e,
                "train/accuracy": train_acc,
                "train/learnrate": optimizer.param_groups[0]['lr'],
                "valid/loss": valid_loss,
                "valid/accuracy": valid_acc,
            }
            wandb.log(metrics, step=e)

            #early stopping
            if config['early_stop']:
                if valid_acc < best_valid_acc:
                    count += 1
                    if count >= patience:
                        break

            if valid_acc > best_valid_acc:
            # if valid_loss < best_valid_loss:
                count = 0
                best_valid_loss = valid_loss
                best_valid_acc = valid_acc
                cur_train_loss = train_loss
                cur_train_acc = train_acc


                del checkpoint
                checkpoint = {}
                print(f'best epoch: {e}')
                checkpoint['model_weights'] = model.state_dict()
                checkpoint['optimizer'] = optimizer.state_dict()
                checkpoint['scheduler'] = scheduler.state_dict()
                checkpoint['train_loss'] = train_loss
                checkpoint['train_acc'] = train_acc
                checkpoint['valid_loss'] = valid_loss
                checkpoint['valid_acc'] = valid_acc

                torch.save(checkpoint, f'{config["save_dir"]}/epochs{e}.pth')
                # print(model)
                
    print(model)
    # Monitor
    # Prepare monitor
    images, labels = torch.tensor([]).to(device), torch.tensor([]).to(device)
    for batch in train_dataloader:
        imgs, lbls = batch
        images = torch.cat((images, imgs.to(device)))
        labels = torch.cat((labels, lbls.to(device)))

    # 需要計算 RM 分布指標
    # need_calculate_status = arch["need_calculate_status"]
    # if need_calculate_status:
    #     use_gray = arch['args']['use_gray']  # 使否使用輪廓層
    #     rgb_layers, gray_layers = get_basic_target_layers(model, use_gray=use_gray)
    #     layer_stats, overall_stats = get_all_layers_stats(model, rgb_layers, gray_layers, images)
    #
    #     for key, value in overall_stats.items():
    #         wandb.summary[key] = value
    #
    #     wandb.summary['layers'] = layer_stats
    #     print(layer_stats)

    return cur_train_loss, cur_train_acc, best_valid_loss, best_valid_acc, checkpoint

def eval(dataloader: DataLoader, model: nn.Module, loss_fn, need_table = True, device=None, use_preprocessed_image = False):
    progress = tqdm(enumerate(dataloader), desc="Loss: ", total=len(dataloader))
    model.eval()
    losses = 0
    correct = 0
    size = 0
    table = []
    with torch.no_grad():
        for batch, (X, y) in progress:
            X = X.to(device); y= y.to(device)

            # 使用影像愈處理
            if use_preprocessed_image:
                X = preprocess_retinal_tensor_batch(X, final_size=config['input_shape'])

            pred = model(X)
            loss = loss_fn(pred, y)
            
            losses += loss.detach().item()
            size += len(X)
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

            if need_table:
                X = X.cpu()
                y = y.cpu()
                
                # sample first image in batch 
                if X[0].shape[0] == 3:
                    X = np.transpose(np.array(X[0]), (1, 2, 0))
                else:
                    X = np.array(X[0])
                table.append([wandb.Image(X), y[0], pred.argmax(1)[0], loss, (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()])

            test_loss = losses/(batch+1)
            test_acc = correct/size
            progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(test_loss, test_acc))
    return test_acc, test_loss, table

config['save_dir'] = increment_path(config['save_dir'], exist_ok = False)
Path(config['save_dir']).mkdir(parents=True, exist_ok=True)

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project=project,

    name = name,

    notes = description,
    
    tags = tags,

    group = group,
    
    # track hyperparameters and run metadata
    config=config
)

train_dataloader, test_dataloader = get_dataloader(dataset=config['dataset'], root=config['root'] + '/data/', batch_size=config['batch_size'], input_size=config['input_shape'])

model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
model = model.to(config['device'])
print(model)
summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']))


eval_loss_fn = get_loss_function(config['loss_fn'])
training_loss_fn = get_loss_function(config['training_loss_fn'])
use_metric_based_loss = config['use_metric_based_loss']

optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), lr=config['lr'], **dict(config['optimizer']['args']))
scheduler = getattr(optim.lr_scheduler, config['lr_scheduler']['name'])(optimizer, **dict(config['lr_scheduler']['args']))

shutil.copyfile(f'./models/{config["model"]["name"]}.py', f'{config["save_dir"]}/{config["model"]["name"]}.py')
shutil.copyfile(f'./config.py', f'{config["save_dir"]}/config.py')

# wandb.watch(model, loss_fn, log="all", log_freq=1)
train_loss, train_acc, valid_loss, valid_acc, checkpoint = train(train_dataloader, test_dataloader, model, eval_loss_fn,
                                                                 optimizer, scheduler, config['epoch'], device = config['device'],
                                                                 training_loss_fn=training_loss_fn, use_metric_based_loss=use_metric_based_loss)
print("Train: \n\tAccuracy: {}, Avg loss: {} \n".format(train_acc, train_loss))
print("Valid: \n\tAccuracy: {}, Avg loss: {} \n".format(valid_acc, valid_loss))

# Test model
model.load_state_dict(checkpoint['model_weights'])
model.to(device)
test_acc, test_loss, test_table = eval(test_dataloader, model, eval_loss_fn, device = config['device'], need_table=False, use_preprocessed_image=config['use_preprocessed_image'])
print("Test: \n\tAccuracy: {}, Avg loss: {} \n".format(test_acc, test_loss))

# Record result into Wandb
wandb.summary['train_accuracy'] = train_acc
wandb.summary['train_avg_loss'] = train_loss
wandb.summary['test_accuracy'] = test_acc
wandb.summary['test_avg_loss'] = test_loss
record_table = wandb.Table(columns=["Image", "Answer", "Predict", "batch_Loss", "batch_Correct"], data = test_table)
wandb.log({"Test Table": record_table})
print(f'checkpoint keys: {checkpoint.keys()}')

torch.save(checkpoint, f'{config["save_dir"]}/{config["model"]["name"]}_best.pth')
art = wandb.Artifact(f'{config["model"]["name"]}_{config["dataset"]}', type="model")
art.add_file(f'{config["save_dir"]}/{config["model"]["name"]}_best.pth')
art.add_file(f'{config["save_dir"]}/{config["model"]["name"]}.py')
art.add_file(f'{config["save_dir"]}/config.py')
wandb.log_artifact(art, aliases = ["latest"])
wandb.finish()