import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Tuple
import numpy as np


def train(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        device: torch.device,
        clip_value: float = 0.25
) -> float:
    """
    Train the model for one epoch
    :param model: The model to be trained.
    :param dataloader: The DataLoader providing the training data.
    :param loss_fn: The loss function to use.
    :param optimizer: The optimizer to use.
    :param epoch: The current epoch number for logging.
    :param device: The device to train on (GPU or CPU).
    :param clip_value: Gradient clipping value (default: 0.0).
    :return:The average loss for the epoch.
    """

    current_loss = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = GradScaler()

    torch.backends.cudnn.benchmark = True

    for idx, (inputs, targets) in enumerate(dataloader):
        if device.type == 'cuda':
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        outputs = model(inputs.float())
        loss = loss_fn(outputs, targets.float().view(-1, 1))

        scaler.scale(loss).backward()

        if clip_value > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        if (idx + 1) % 2 == 0 or (idx + 1) == len(dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        current_loss += loss.item() * inputs.size(0)

        if idx % 100 == 0:
            correlation = np.corrcoef(outputs.cpu().detach().numpy().flat, targets.cpu())
            print(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss.item():.4f}, Correlation: {correlation.min().item():.4f}")

    avg_loss = current_loss / len(dataloader.dataset)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float, list, list]:
    """

    :param model: The model to be trained.
    :param dataloader: The DataLoader providing the validation data.
    :param loss_fn: The loss function to use.
    :param device: The device to train on (GPU or CPU).
    :return: The average loss and correlation on the validation set.
    """

    model.eval()
    current_loss = 0.0
    all_outputs, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in dataloader:
            if device.type == 'cuda':
                inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            outputs = model(inputs.float())
            loss = loss_fn(outputs, targets.float().view(-1, 1))

            current_loss += loss.item() * inputs.size(0)
            all_outputs.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    correlation = np.corrcoef(all_outputs, all_targets).min()
    avg_loss = current_loss / len(dataloader.dataset)
    return avg_loss, correlation, all_outputs, all_targets


def main(
    model: nn.Module,
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    dataloader_test: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    clip_value: float = 0.0,
    model_save_path: str = './model.pth'
) -> None:
    """
    Main function to run the training and validation loops.

    Parameters:
        model (nn.Module): The model to be trained and validated.
        dataloader_train (DataLoader): DataLoader for the training data.
        dataloader_val (DataLoader): DataLoader for the validation data.
        dataloader_test (DataLoader): DataLoader for the test data.
        epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        device (torch.device): The device to run training/validation on.
        clip_value (float): Gradient clipping value (default: 0.0).
        model_save_path (str): Path to save the trained model (default: './model.pth').
    """
    loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)

    best_model = None
    best_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train(model, dataloader_train, loss_fn, optimizer, epoch, device, clip_value)
        val_loss, val_corr = validate(model, dataloader_val, loss_fn, device)

        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Correlation: {val_corr:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = deepcopy(model.state_dict())

        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Validation Loss": val_loss,
            "Validation Correlation": val_corr,
        })

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), model_save_path)
    print(f"Best model saved to {model_save_path}")

    test_loss, test_corr = validate(model, dataloader_test, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}, Test Correlation: {test_corr:.4f}")


if __name__ == "__main__":
    # Example usage
    # Replace `YourModel`, `dataloader_train`, `dataloader_val`, `dataloader_test` with your specific instances
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YourModel()
    dataloader_train = ...
    dataloader_val = ...
    dataloader_test = ...
    epochs = 100
    learning_rate = 1e-3
    clip_value = 0.25

    main(
        model,
        dataloader_train,
        dataloader_val,
        dataloader_test,
        epochs,
        learning_rate,
        device,
        clip_value
    )