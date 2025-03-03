from pathlib import Path

import torch
import wandb
import argparse
from torch import nn, optim
from copy import deepcopy

from config import get_config
from data_initialize import data_initialize
from model import MyNet
from train import train
from evaluate import validate, evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate a Deep Learning Model")
    parser.add_argument("--eval_type", type=str, required=True, choices=["20datasets", "ptm", "aa_glycine"],
                        help="Specify the evaluation type(20datasets, ptm, aa_glycine).")
    parser.add_argument("--dataset_name", type=str, required=True, help="Specify the dataset name.")
    parser.add_argument("--test_aa", type=str, default=None, help="Specify the test amino acid (if applicable).")
    parser.add_argument("--save_results", action="store_false", help="Save evaluation results.")
    args = parser.parse_args()

    # Load configuration
    config = get_config(epoch=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize data
    data_loaders = data_initialize(args.eval_type, dataset_name=args.dataset_name, test_aa=args.test_aa,
                                   batch_size=config["batch_size"])

    if args.eval_type == "ptm":
        dataloader_train, dataloader_val, dataloader_test, dataloader_test_no_mod, x_shape = data_loaders
        dataloader_test_extra = dataloader_test_no_mod  # Use the additional dataset
    elif args.eval_type == "aa_glycine":
        dataloader_train, dataloader_val, dataloader_test, dataloader_test_g, x_shape = data_loaders
        dataloader_test_extra = dataloader_test_g  # Use the additional dataset
    else:
        dataloader_train, dataloader_val, dataloader_test, x_shape = data_loaders
        dataloader_test_extra = None  # No extra dataset in this case

    # Initialize model
    model = MyNet(x_shape=x_shape, config=config).to(device)

    # Weights & Biases setup
    wandb.init(config=config, project="deep_learning_model",entity="alirezak2", name=args.dataset_name, mode="online")

    # Define loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    best_loss = float("inf")
    best_model_state = None

    # Training loop
    for epoch in range(config["epochs"]):
        loss_train = train(model, dataloader_train, loss_function, optimizer, epoch, device, config["clipping_size"])
        loss_valid, corr_valid, _, _ = validate(model, dataloader_val, loss_function, device)

        print(f"Epoch {epoch}: Train Loss {loss_train:.4f}, Val Loss {loss_valid:.4f}, Correlation {corr_valid:.4f}")

        if loss_valid < best_loss:
            best_loss = loss_valid
            best_model_state = model.state_dict()

        wandb.log({"Train Loss": loss_train, "Validation Loss": loss_valid, "Validation Correlation": corr_valid})

    # Save best model
    best_model_path = f"Zenodo/iDeepLC/saved_model/{args.dataset_name}_best.pth"
    Path(best_model_path.rsplit('/',1)[0]).mkdir(parents=True, exist_ok=True)
    torch.save(best_model_state, best_model_path)

    # Evaluate on the test set
    evaluate_model(model, dataloader_test, dataloader_test_extra, loss_function, device, best_model_path, args.eval_type, args.save_results)

    wandb.finish()


if __name__ == "__main__":
    main()
