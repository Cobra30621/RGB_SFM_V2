import shutil

from sklearn.model_selection import KFold

import wandb
import numpy as np

from torch import optim, nn
from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader, Subset
from torchsummary import summary

from dataloader import get_dataloader
from config import *
import models
from diabetic_retinopathy_handler import preprocess_retinal_tensor_image, preprocess_retinal_tensor_batch

from loss.loss_function import get_loss_function

def train(train_dataloader: DataLoader, valid_dataloader: DataLoader, model: nn.Module, eval_loss_fn, optimizer,
          scheduler, epoch, device):

    best_valid_acc = 0
    best_valid_loss = float('inf')
    count = 0
    patience = config['patience']
    # 使用影像前處理
    use_preprocessed_image = config['use_preprocessed_image']
    checkpoint = {}


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
                X = X.to(device)
                y = y.to(device)
                print(f"X.shape: {X.shape}")
                # 使用影像愈處理
                if use_preprocessed_image:
                    X = preprocess_retinal_tensor_batch(X, final_size=config['input_shape'])

                print(f"X.shape 2: {X.shape}")
                pred = model(X)
                print(f"pred.shape: {pred.shape}, y.shape: {y.shape}")

                loss = eval_loss_fn(pred, y)

                # 反向传播
                loss.backward()
                # 更新模型参数
                optimizer.step()

                # 清零梯度
                optimizer.zero_grad()

                losses += loss.detach().item()
                size += len(X)

                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

                train_loss = losses / (batch + 1)
                train_acc = correct / size
                progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(train_loss, train_acc))

            valid_acc, valid_loss, _ = eval(valid_dataloader, model, eval_loss_fn, False, device=device,
                                            use_preprocessed_image=use_preprocessed_image)
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

            # early stopping
            if config['early_stop']:
                if valid_acc < best_valid_acc:
                    count += 1
                    if count >= patience:
                        break

            if valid_acc > best_valid_acc:
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

    return cur_train_loss, cur_train_acc, best_valid_loss, best_valid_acc, checkpoint


def eval(dataloader: DataLoader, model: nn.Module, loss_fn, need_table=True, device=None, use_preprocessed_image=False):
    progress = tqdm(enumerate(dataloader), desc="Loss: ", total=len(dataloader))
    model.eval()
    losses = 0
    correct = 0
    size = 0
    table = []
    with torch.no_grad():
        for batch, (X, y) in progress:
            X = X.to(device);
            y = y.to(device)

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
                table.append([wandb.Image(X), y[0], pred.argmax(1)[0], loss,
                              (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()])

            test_loss = losses / (batch + 1)
            test_acc = correct / size
            progress.set_description("Loss: {:.7f}, Accuracy: {:.7f}".format(test_loss, test_acc))
    return test_acc, test_loss, table



model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args']))
model = model.to(config['device'])
print(model)
summary(model, input_size = (config['model']['args']['in_channels'], *config['input_shape']))


eval_loss_fn = get_loss_function(config['loss_fn'])


optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), lr=config['lr'], **dict(config['optimizer']['args']))
scheduler = getattr(optim.lr_scheduler, config['lr_scheduler']['name'])(optimizer, **dict(config['lr_scheduler']['args']))

# 利用 get_dataloader 獲取 train_dataloader，並抽取其 dataset
train_dataloader, _ = get_dataloader(
    dataset=config['dataset'],
    root=config['root'] + '/data/',
    batch_size=config['batch_size'],
    input_size=config['input_shape']
)

train_dataset = train_dataloader.dataset

# 產生標籤列表（假設 Dataset 回傳 (X, y)，且 y 是 one-hot）
all_labels = [train_dataset[i][1].argmax().item() for i in range(len(train_dataset))]

k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
kfold_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(train_dataset)), all_labels)):
    print(f"\n===== Fold {fold+1} / {k } =====")

    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=config['batch_size'], shuffle=False)

    model = getattr(getattr(models, config['model']['name']), config['model']['name'])(**dict(config['model']['args'])).to(config['device'])
    summary(model, input_size=(config['model']['args']['in_channels'], *config['input_shape']))

    eval_loss_fn = get_loss_function(config['loss_fn'])
    optimizer = getattr(optim, config['optimizer']['name'])(model.parameters(), lr=config['lr'], **dict(config['optimizer']['args']))
    scheduler = getattr(optim.lr_scheduler, config['lr_scheduler']['name'])(optimizer, **dict(config['lr_scheduler']['args']))

    shutil.copyfile(f'./models/{config["model"]["name"]}.py', f'{config["save_dir"]}/{config["model"]["name"]}_fold{fold+1}.py')
    shutil.copyfile(f'./config.py', f'{config["save_dir"]}/config_fold{fold+1}.py')

    train_loss, train_acc, valid_loss, valid_acc, checkpoint = train(
        train_loader, val_loader, model, eval_loss_fn,
        optimizer, scheduler, config['epoch'], device=config['device']
    )

    print(f"Fold {fold+1}: Train Acc = {train_acc}, Valid Acc = {valid_acc}")
    kfold_results.append((train_acc, valid_acc))

    val_acc, val_loss, _ = eval(
        val_loader, model, eval_loss_fn,
        device=config['device'], need_table=False, use_preprocessed_image=config['use_preprocessed_image']
    )
    print(f"Fold {fold+1}: Test Acc = {val_acc}, Loss = {val_loss}")

    torch.save(checkpoint, f'{config["save_dir"]}/{config["model"]["name"]}_fold{fold+1}_best.pth')

print("\n===== K-Fold Summary =====")
for i, (train_acc, val_acc) in enumerate(kfold_results):
    print(f"Fold {i+1}: Train Acc = {train_acc:.4f}, Valid Acc = {val_acc:.4f}")

mean_train = np.mean([x[0] for x in kfold_results])
mean_valid = np.mean([x[1] for x in kfold_results])
print(f"Average Train Acc: {mean_train:.4f}, Average Valid Acc: {mean_valid:.4f}")