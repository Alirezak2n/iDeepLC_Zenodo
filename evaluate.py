import torch
from torch import nn, optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from copy import deepcopy
from typing import Tuple, Optional
import numpy as np


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Tuple[float, float, list, list]:
    """
    Validate the model on a given dataset.
    :param model: The trained model.
    :param dataloader: The DataLoader providing the validation/test data.
    :param loss_fn: The loss function to use.
    :param device: The device to train on (GPU or CPU).
    :return: Average loss, correlation coefficient, predictions, and ground truth values.
    """

    model.eval()
    total_loss = 0.0
    outputs, targets = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            if device.type == 'cuda':
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs_batch = model(inputs.float())
            loss = loss_fn(outputs_batch, labels.float().view(-1, 1))

            total_loss += loss.item() * inputs.size(0)
            outputs.extend(outputs_batch.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader.dataset)
    correlation = np.corrcoef(outputs, targets)[0, 1]

    return avg_loss, correlation, outputs, targets


def evaluate_model(
    model: nn.Module,
    dataloader_test: DataLoader,
    dataloader_extra: Optional[DataLoader],  # Handles different evaluation types
    loss_fn: nn.Module,
    device: torch.device,
    model_path: str,
    eval_type: str,
    save_results: bool = True
):
    """
    Load a trained model and evaluate it on test datasets.

    :param model: The trained model.
    :param dataloader_test: Test dataset loader.
    :param dataloader_extra: Additional dataset loader based on `eval_type` (e.g., test_no_mod, test_g).
    :param loss_fn: Loss function.
    :param device: Computation device.
    :param model_path: Path to the trained model.
    :param eval_type: Type of evaluation ('ptm', 'aa_glycine', etc.).
    :param save_results: If True, saves the evaluation results.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))

    loss_test, corr_test, output_test, y_test = validate(model, dataloader_test, loss_fn, device)
    print(f'Test Loss: {loss_test:.4f}, Correlation: {corr_test:.4f}')
    if dataloader_extra is not None:
        loss_extra, corr_extra, output_extra, y_extra = validate(model, dataloader_extra, loss_fn, device)
        print(f'{eval_type} Test Loss: {loss_extra:.4f}, Correlation: {corr_extra:.4f}')

    if save_results:
        filename = model_path.replace('.pth', '_results.csv')
        if dataloader_extra is not None:
            # Stack three columns
            data_to_save = np.column_stack((y_test, output_test, output_extra))
            header = "y_test,output_test,output_extra"
        else:
            # Stack only two columns (avoid NoneType values)
            data_to_save = np.column_stack((y_test, output_test))
            header = "y_test,output_test"

        np.savetxt(filename, data_to_save, delimiter=',', header=header, fmt='%.6f')
        print(f"Results saved to {filename}")

