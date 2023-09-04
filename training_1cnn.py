import pandas as pd
import wandb
from torch import nn, onnx
import torch.cuda
import numpy as np
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
import sys

dataset = 'ProteomeTools'
# Loading the data

train_x = np.load('./data_matrix/' + dataset + '/train_x.npy')
train_y = np.load('./data_matrix/' + dataset + '/train_y.npy')
val_x = np.load('./data_matrix/' + dataset + '/val_x.npy')
val_y = np.load('./data_matrix/' + dataset + '/val_y.npy')
test_x = np.load('./data_matrix/' + dataset + '/test_x.npy')
test_y = np.load('./data_matrix/' + dataset + '/test_y.npy')
max_in_data = 1620  # for reverting data back to its actual amount
min_in_data = 0
print(dataset)
show_network = False
training = False
want_figure = True
last_path = './saved_models/' + dataset + '/last.pth'
best_path = './saved_models/' + dataset + '/best.pth'


# Initialize the hyper parameters

def get_config(lr=0.001, epoch=50, batch=256, cnn_layers=1, fc_layers=4, drop=0.2, kernel=9, fc_output=128, clip=0.25):
# def get_config(lr=0.0005, epoch=100, batch=64, cnn_layers=3, fc_layers=6, drop=0.2, kernel=7, fc_output=128, clip=0.25):
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


config = get_config()
# project_name = dataset.split("/")[0]
project_name= 'ProteomeTools_test'
dataset_type = (dataset.split("/")[1][:4] if '/' in dataset else 'minmax')
# run_name = dataset_type + '_WClip_WRelu_lr' + str(config["learning_rate"])[4:] + '_b' + \
#            f'{config["batch_size"]}' + '_WElu_dropout' + str(config["dropout"])[2:]
run_name = 'first'

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

dataloader_train = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"])
dataloader_val = DataLoader(val_dataset, pin_memory=True)
dataloader_test = DataLoader(test_dataset, pin_memory=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if training:
    wandb.init(config=config, project=project_name, entity="alirezak2", name=run_name,mode='online')
else:
    wandb.init(config=config, project=project_name, entity="alirezak2", name=run_name, mode='offline')

padding = int((wandb.config['kernel_size'] - 1) / 2)


# Create Network architecture
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.l = nn.ModuleList()

        self.l.append(
            nn.Conv1d(in_channels=51, out_channels=100, kernel_size=(wandb.config['kernel_size'],), stride=(1,),
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
def network_firstlook():
    print(MyNet().parameters)
    total_param = sum(p.numel() for p in MyNet().parameters())
    print("Total Parameters =  ", total_param)

    for batch in dataloader_train:
        X, y = batch[0].to(device), batch[1].to(device)
        X = X.float()
        print(X.dtype)
        print('X is cuda:', X.is_cuda)
        print('X.shape:', X.shape)
        print('y.shape:', y.shape)
        outputs = MyNet().to(device)(X)
        print('outputs.shape:', outputs.shape)
        # onnx.export(MyNet(), X, 'old_arch.onyx', input_names=['Encoded Sequence'],
        #             output_names=['RT Prediction'])
        break


def train(model, loader, loss, optimizer, epoch):
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
        if wandb.config['clipping_size'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), wandb.config['clipping_size'])
        optimizer.step()

        current_loss += loss_fn.item() * X.size(0)
        correlation = np.corrcoef(output.cpu().detach().numpy().flat, y.cpu()).min()

        if idx % 100 == 0:  # print every 100 mini-batches
            print('epoch: {}, train_loss: {:.4f}, "Training correlation: {:.4f}'.format(epoch, loss_fn, correlation))



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


def make_figures(model, model_path, loss, show_metrics=False):
    model.load_state_dict(torch.load(model_path))
    path_name = model_path.split("/")[-1][:-4]
    loss_test, corr_test, output_test, y_test = validation(model, dataloader_test, loss)
    pd.DataFrame(zip(y_test, output_test)).to_csv('temp.csv')  # save the results
    print('\n Test set Loss: {:.4f}  Test set Correlation: {:.4f}\n'
          .format(loss_test, corr_test))
    y_test = [i * (max_in_data - min_in_data) + min_in_data for i in y_test]
    output_test = [i * (max_in_data - min_in_data) + min_in_data for i in output_test]
    output = output_test
    y = y_test
    maximum_number = max(output)

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    mae_test = mean_absolute_error(output_test, y_test)


    ax1.scatter(y_test, output, c='b',
                label='{}: {} (MAE: {:.3f}, R: {:.3f}) \n {}'.format(dataset, path_name, mae_test, corr_test, run_name),
                s=3)

    plt.legend(loc='upper left')
    plt.xlabel('observed retention time')
    plt.ylabel('predicted retention time')
    plt.axis('scaled')
    ax1.plot([0, maximum_number], [0, maximum_number], ls="--", c=".5")
    plt.xlim(0, maximum_number)
    plt.ylim(0, maximum_number)
    # plt.show()
    plt.savefig(model_path[:-8]+'/'+f'{dataset}-{path_name}.png')


    if show_metrics:
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

        print("Run name: ", path_name)
        print('Pearson Correlation: \n', pear)
        print('RSE=', rse)
        print('R-Square=', rsquare)
        print('rmse=', rmse)
        print('mae=', mae)

def test():
    testx_temp = np.load('testx_temp.npy')
    testy_temp = np.load('testy_temp.npy')
    test_temp_dataset = MyDataset(testx_temp, testy_temp)
    dataloader_test_temp = DataLoader(test_temp_dataset, pin_memory=True)

    model = MyNet()
    model.to(device)

    model.load_state_dict(torch.load(best_path))

    model.eval()
    for idx, batch in enumerate(dataloader_test_temp):
        X, y = batch
        # move data to gpu if possible
        if device.type == 'cuda':
            X, y = X.to(device), y.to(device)
        X = X.float()
        y = y.float()
        output = model(X)
        print(output)


def main():
    model = MyNet()
    model.to(device)
    if show_network:
        network_firstlook()
    total_param = sum(p.numel() for p in MyNet().parameters())

    # Hyper parameters
    epochs = wandb.config["epochs"]
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), wandb.config["learning_rate"])
    wandb.config["Total_Parameters"] = total_param
    wandb.watch(model)

    best_epochs = {}
    best_loss = 1000000
    best_loss_test = 100000
    best_test_cor = 0

    if training:
        bad_cor = 0
        for epoch in range(epochs):
            loss_train = train(model, dataloader_train, loss_function, optimizer, epoch)
            loss_valid, corr, output, y = validation(model, dataloader_val, loss_function)
            loss_test, corr_test, output_test, y_test = validation(model, dataloader_test, loss_function)

            if best_test_cor < corr_test:
                best_test_cor = corr_test
            print('\n Epoch: {}  Train Loss: {:.4f}  Validation Loss: {:.4f}  Validation Correlation: {:.4f}\n'
                  .format(epoch, loss_train, loss_valid, corr))

            if loss_valid < best_loss:
                best_loss = loss_valid
                torch.save(model.state_dict(), best_path)
                best_epochs[epoch] = [loss_valid, corr]

            if loss_test < best_loss_test:
                best_loss_test = loss_test

            if not corr > 0.3:
                bad_cor += 1
            if bad_cor > 5:
                break

            wandb.log({
                "Train Loss Averaged": loss_train,
                "Valid Loss Averaged": loss_valid,
                "Test loss Averaged": loss_test,
                "Correlation": corr,
                "Test set Correlation": corr_test,
                #"Total Parameters": total_param,
                "Best test correlation": best_test_cor,
                "Best Loss": best_loss,
                "Best Loss Test": best_loss_test
            })
        print('best_epochs: ', best_epochs, "\n", "best_test_corr: ", best_test_cor)
        torch.save(model.state_dict(), last_path)
    if want_figure:
        make_figures(model, last_path, loss_function)
        make_figures(model, best_path, loss_function)


if __name__ == '__main__':
    main()
    # network_firstlook()