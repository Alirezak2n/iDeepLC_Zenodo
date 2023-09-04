import wandb
from torch import nn
import torch.cuda
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import sys

want_figure = True
over_test_set = True
training = True
save_model = True
mods = sys.argv[1]
type_mod = "diamino13_atom_stan/"
mod = type_mod+mods
last_path = './saved_models/14ptms/'+type_mod+mod.split("/")[1]+'_last.pth'
best_path = './saved_models/14ptms/'+type_mod+mod.split("/")[1]+'_best.pth'

# Loading the data
train_x = np.load('./data_matrix/14ptm/'+mod+'/train_x.npy')
train_y = np.load('./data_matrix/14ptm/'+mod+'/train_y.npy')
val_x = np.load('./data_matrix/14ptm/'+mod+'/val_x.npy')
val_y = np.load('./data_matrix/14ptm/'+mod+'/val_y.npy')
test_x = np.load('./data_matrix/14ptm/'+mod+'/test_x.npy')
test_y = np.load('./data_matrix/14ptm/'+mod+'/test_y.npy')
test_no_mod_x = np.load('./data_matrix/14ptm/'+mod+'/test_no_mod_x.npy')
test_no_mod_y = np.load('./data_matrix/14ptm/'+mod+'/test_no_mod_y.npy')

# Initialize the hyper parameters

def get_config(lr=0.001, epoch=2000, batch=256, cnn_layers=1, fc_layers=4, drop=0.2, kernel=9, fc_output=128, clip=0.25):

    config = {
        "learning_rate": lr,
        "epochs": epoch,
        "batch_size": batch,
        "cnn_layers": cnn_layers,
        "fc_layers": fc_layers,
        "dropout": drop,
        "kernel_size": kernel,
        "fc_out": fc_output,
        "clipping_size": clip
    }

    return config


# Making the pytorch dataset

class MyDataset(Dataset):
    def __init__(self, sequences, retention):
        self.sequences = sequences
        self.retention = retention

    def __len__(self):
        return len(self.retention)

    def __getitem__(self, idx):
        return (self.sequences[idx], self.retention[idx])


# Initiate Dataloaders

train_dataset = MyDataset(train_x, train_y)
val_dataset = MyDataset(val_x, val_y)
test_dataset = MyDataset(test_x, test_y)
test_no_mod_dataset = MyDataset(test_no_mod_x, test_no_mod_y)

config = get_config()
dataloader_train = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"])
dataloader_val = DataLoader(val_dataset, pin_memory=True)
dataloader_test = DataLoader(test_dataset, pin_memory=True)
dataloader_test_no_mod = DataLoader(test_no_mod_dataset, pin_memory=True)

import wandb

# wandb.init(config=config, project="Retention Prediction", entity="alirezak2", )
wandb.init(config=config, project="14ptms_revised", entity="alirezak2",name=f'{mod}', dir='D:/Wandb_temp' )


# Create Network architecture



padding = int((wandb.config['kernel_size'] - 1) / 2)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.l = nn.ModuleList()

        self.l.append(
            nn.Conv1d(in_channels=57, out_channels=100, kernel_size=(wandb.config['kernel_size'],), stride=(1,),
                      padding=(padding,)))
        # self.l.append(nn.BatchNorm1d(num_features=100))
        self.l.append(nn.ELU())
        # self.l.append(nn.Dropout(p=wandb.config['dropout'], inplace=False))

        for layer in range(wandb.config['cnn_layers']):
            self.l.append(
                nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(wandb.config['kernel_size'],), stride=(1,),
                          padding=(padding,)))
            # self.l.append(nn.BatchNorm1d(num_features=100))
            self.l.append(nn.ELU())
            self.l.append(nn.Dropout(p=wandb.config['dropout'], inplace=False))

            self.l.append(
                nn.Conv1d(in_channels=100, out_channels=100, kernel_size=(wandb.config['kernel_size'],), stride=(1,),
                          padding=(padding,)))
            # self.l.append(nn.BatchNorm1d(num_features=100))
            self.l.append(nn.ELU())
            self.l.append(nn.Dropout(p=wandb.config['dropout'], inplace=False))

        self.l.append(
            nn.Conv1d(in_channels=100, out_channels=32, kernel_size=(wandb.config['kernel_size'],), stride=(1,),
                      padding=(padding,)))
        # self.l.append(nn.BatchNorm1d(num_features=32))
        self.l.append(nn.ELU())
        self.l.append(nn.Dropout(p=wandb.config['dropout'], inplace=False))

        self.l.append(nn.Flatten())

        self.l.append(nn.Linear(in_features=32 * train_x.shape[2], out_features=wandb.config['fc_out']))
        self.l.append(nn.ELU())

        for layer in range(wandb.config['fc_layers']):
            self.l.append(nn.Linear(in_features=wandb.config['fc_out'], out_features=wandb.config['fc_out']))
            self.l.append(nn.ELU())

        self.l.append(nn.Linear(in_features=wandb.config['fc_out'], out_features=int(wandb.config['fc_out'])))
        self.l.append(nn.ReLU())
        self.l.append(nn.Linear(in_features=int(wandb.config['fc_out']), out_features=int(wandb.config['fc_out'])))
        self.l.append(nn.ReLU())
        self.l.append(nn.Linear(in_features=int(wandb.config['fc_out']), out_features=1))

    def forward(self, x):
        for layer in self.l:
            x = layer(x)

        return x



# First look at Network

print(MyNet().parameters)
total_param = sum(p.numel() for p in MyNet().parameters())
print("Total Parameters =  ", total_param)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = MyNet()
model.to(device)

for batch in dataloader_train:
    X, y = batch[0].to(device), batch[1].to(device)
    X = X.float()
    print(X.dtype)
    print('X is cuda:', X.is_cuda)
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    outputs = model(X)
    print('outputs.shape:', outputs.shape)
    break


def train(model, loader, loss):
    current_loss = 0
    model.train()

    # reading the data
    for idx, batch in enumerate(loader):
        X, y = batch
        # move data to gpu if possible
        if device.type == 'cuda':
            X, y = X.to(device), y.to(device)
        # training steps
        optimizer.zero_grad()
        X = X.float()
        y = y.float()
        output = model(X)
        loss_fn = loss(output, y.reshape(-1, 1))
        loss_fn.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config['clipping_size'])
        optimizer.step()

        current_loss += loss_fn.item() * X.size(0)
        correlation = np.corrcoef(output.cpu().detach().numpy().flat, y.cpu()).min()

        if idx % 100 == 0:  # print every 2000 mini-batches
            print('epoch: %d, train_loss: %.3f' % (epoch, loss_fn.item()), "Training correlation:", correlation)

            # showing some examples of prediction
            # print("5 Random samples: ")
            # for i in random.sample(range(0,len(output)),5):
            #    print('predict: %.3f, real: %.3f'%(output[i].item(),y[i].item()))
            # print('')

    epoch_loss = current_loss / len(loader.dataset)
    return epoch_loss


def validation(model, loader, loss):
    current_loss = 0
    model.eval()
    outputs = []
    ys = []
    # reading the data
    for idx, batch in enumerate(loader):
        X, y = batch
        # move data to gpu if possible
        if device.type == 'cuda':
            X, y = X.to(device), y.to(device)
        # validating steps

        X = X.float()
        y = y.float()
        output = model(X)
        ys.append(y.item())
        outputs.append(output.item())
        loss_fn = loss(output, y.reshape(-1, 1))

        current_loss += loss_fn.item() * X.size(0)
        # print('loss: %.3f' %(loss_fn.item()))
    # print('number of outputs: ',len(output))

    correlation = np.corrcoef(outputs, ys).min()
    epoch_loss = current_loss / len(loader.dataset)
    return epoch_loss, correlation, outputs, ys


model = MyNet()

model.to(device)

# Hyper parameters

epochs = wandb.config["epochs"]
learning_rate = wandb.config["learning_rate"]
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
wandb.config["Total_Parameters"] = total_param
wandb.watch(model)

best_epochs= {mod:mods}
best_loss = 100
best_test_cor = 0
# Training the model
if training:
    for epoch in range(epochs):
        loss_train = train(model, dataloader_train, loss_function)
        loss_valid, corr, output, y = validation(model, dataloader_val, loss_function)

        loss_test, corr_test, output_test, y_test = validation(model, dataloader_test, loss_function)
        loss_test_no_mod, corr_test_no_mod, output_test_no_mod, y_test_no_mod = validation(model,
                                                                                           dataloader_test_no_mod,
                                                                                           loss_function)
        if best_test_cor < corr_test:
            best_test_cor = corr_test
        print('\n Epoch: {}  Train Loss: {:.4f}  Validation Loss: {:.4f}  Validation Correlation: {:.4f}\n'
              .format(epoch, loss_train, loss_valid, corr))

        if loss_valid < best_loss:
            best_loss = loss_valid
            torch.save(model.state_dict(), best_path)
            best_epochs[epoch]=[loss_valid,corr]
            # torch.save(model.state_dict(), './saved_models/14ptms/{}.pth'.format(loss_valid))
        try:
            wandb.log({
                # "Epoch": epoch,
                "Train Loss Averaged": loss_train,
                "Correlation": corr,
                "Valid Loss Averaged": loss_valid,
                "Test set Correlation": corr_test,
                "No mod Test set Correlation": corr_test_no_mod,
                # "Total Parameters": total_param,
                "Best test correlation": best_test_cor
                # "Valid Acc": acc_valid
            })
        except:
            pass
    print('best_epochs: ', best_epochs)
if save_model:
    torch.save(model.state_dict(), last_path)
if not training:
    model.load_state_dict(torch.load(last_path))

if over_test_set:
    loss_test, corr_test, output_test, y_test = validation(model, dataloader_test, loss_function)
    loss_test_no_mod, corr_test_no_mod, output_test_no_mod, y_test_no_mod = validation(model,
                                                                                       dataloader_test_no_mod,
                                                                                       loss_function)
    pd.DataFrame(zip(y_test, output_test)).to_csv('./saved_models/14ptms/'+type_mod+mod.split("/")[1]+'_results.csv')  # save the results
    print('\n Test set Loss: {:.4f}  Test set Correlation: {:.4f}\n'
          .format(loss_test, corr_test))
    print('\n Test set no mod Loss: {:.4f}  Test set no mod Correlation: {:.4f}\n'
          .format(loss_test_no_mod, corr_test_no_mod))
    # y_test = [i * (61.09 - 7.45) + 7.45 for i in y_test]
    # output_test = [i * (61.09 - 7.45) + 7.45 for i in output_test]
    # y_test_no_mod = [i * (61.09 - 7.45) + 7.45 for i in y_test_no_mod]
    # output_test_no_mod = [i * (61.09 - 7.45) + 7.45 for i in output_test_no_mod]
    y_test = [i * (60)  for i in y_test]
    output_test = [i * (60)  for i in output_test]
    y_test_no_mod = [i * (60) for i in y_test_no_mod]
    output_test_no_mod = [i * (60) for i in output_test_no_mod]
    output = output_test
    y = y_test

if want_figure:
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    mae_test = mean_absolute_error(output_test, y_test)
    mae_no_mod = mean_absolute_error(output_test_no_mod, y_test_no_mod)

    ax1.scatter(y_test_no_mod, output_test_no_mod, c='r', label=mod +
                                                                ' not_encoded (MAE: {:.3f}, R: {:.3f})'.format(
                                                                    mae_no_mod, corr_test_no_mod), s=3)
    ax1.scatter(y_test, output_test, c='b', label=mod +
                                                  ' encoded (MAE: {:.3f}, R: {:.3f})'.format(mae_test, corr_test), s=3)

    plt.legend(loc='upper left')
    plt.xlabel('observed retention time')
    plt.ylabel('predicted retention time')
    plt.axis('scaled')
    ax1.plot([0, 4000], [0, 4000], ls="--", c=".5")
    plt.xlim(0, 4000)
    plt.ylim(0, 4000)
    plt.savefig('./saved_models/14ptms/'+type_mod+mod.split("/")[1]+'_last.png')
    # plt.show()


model.load_state_dict(torch.load(best_path))

if over_test_set:
    loss_test, corr_test, output_test, y_test = validation(model, dataloader_test, loss_function)
    loss_test_no_mod, corr_test_no_mod, output_test_no_mod, y_test_no_mod = validation(model,
                                                                                       dataloader_test_no_mod,
                                                                                       loss_function)
    print('\n Test set Loss: {:.4f}  Test set Correlation: {:.4f}\n'
          .format(loss_test, corr_test))
    print('\n Test set no mod Loss: {:.4f}  Test set no mod Correlation: {:.4f}\n'
          .format(loss_test_no_mod, corr_test_no_mod))
    # y_test = [i * (61.09 - 7.45) + 7.45 for i in y_test]
    # output_test = [i * (61.09 - 7.45) + 7.45 for i in output_test]
    # y_test_no_mod = [i * (61.09 - 7.45) + 7.45 for i in y_test_no_mod]
    # output_test_no_mod = [i * (61.09 - 7.45) + 7.45 for i in output_test_no_mod]
    y_test = [i * (60)  for i in y_test]
    output_test = [i * (60)  for i in output_test]
    y_test_no_mod = [i * (60) for i in y_test_no_mod]
    output_test_no_mod = [i * (60) for i in output_test_no_mod]
    output = output_test
    y = y_test

if want_figure:

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    mae_test = mean_absolute_error(output_test, y_test)
    mae_no_mod = mean_absolute_error(output_test_no_mod, y_test_no_mod)

    ax1.scatter(y_test_no_mod, output_test_no_mod, c='r', label=mod +
                                                                ' not_encoded (MAE: {:.3f}, R: {:.3f})'.format(
                                                                    mae_no_mod, corr_test_no_mod), s=3)
    ax1.scatter(y_test, output_test, c='b', label=mod +
                                                  ' encoded (MAE: {:.3f}, R: {:.3f})'.format(mae_test, corr_test), s=3)

    plt.legend(loc='upper left')
    plt.xlabel('observed retention time')
    plt.ylabel('predicted retention time')
    plt.axis('scaled')
    ax1.plot([0, 4000], [0, 4000], ls="--", c=".5")
    plt.xlim(0, 4000)
    plt.ylim(0, 4000)
    plt.savefig('./saved_models/14ptms/'+type_mod+mod.split("/")[1]+'_best.png')
    # plt.show()

# Metrics
output_arrayed = np.array(output).reshape(-1, 1)
y_arrayed = np.array(y).reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(output_arrayed, y_arrayed)

num_data = len(y)

pear = np.corrcoef(output, y)
mse = mean_squared_error(output, y)
rmse = math.sqrt(mse / num_data)
rse = math.sqrt(mse / (num_data - 2))
rsquare = linear_regressor.score(output_arrayed, y_arrayed)
mae = mean_absolute_error(output, y)

print('Pearson Correlation: \n', pear)
print('RSE=', rse)
print('R-Square=', rsquare)
print('rmse=', rmse)
print('mae=', mae)

