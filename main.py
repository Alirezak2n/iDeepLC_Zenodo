import datetime
from pathlib import Path

import torch
import wandb
import argparse
from torch import nn, optim
from copy import deepcopy
from figure import make_figures
from config import get_config
from data_initialize import data_initialize
from model import MyNet
from train import train
from evaluate import validate, evaluate_model


def get_model_save_path(eval_type, dataset_name, test_aa=None):
    """
    Determines the correct directory and filename for saving the model based on eval_type.
    Appends a timestamp to the filename to prevent overwriting.

    Args:
        eval_type (str): Type of evaluation (20datasets, ptm, aa_glycine).
        dataset_name (str): The dataset name.
        test_aa (str, optional): The amino acid (for modified_glycine_evaluation).

    Returns:
        tuple: (model_save_path, model_dir)
    """
    timestamp = datetime.datetime.now().strftime("%m%d")

    if eval_type == "20datasets":
        model_dir = Path(f"../saved_model/20_datasets_evaluation/{dataset_name}_{timestamp}")
        pretrained_path = Path(f"../saved_model/20_datasets_evaluation/{dataset_name}/best.pth")
        model_name = f"best.pth"
    elif eval_type == "aa_glycine":
        model_dir = Path(f"../saved_model/modified_glycine_evaluation/{dataset_name}_{timestamp}")
        pretrained_path = Path(f"../saved_model/modified_glycine_evaluation/{dataset_name}/{test_aa}_best.pth")
        model_name = f"{test_aa}_best.pth"
    elif eval_type == "ptm":
        model_dir = Path("../saved_model/PTM_evaluation")
        pretrained_path = Path(f"../saved_model/PTM_evaluation/{dataset_name}_best.pth")
        model_name = f"{dataset_name}_best_{timestamp}.pth"
    else:
        raise ValueError(f"Unknown eval_type: {eval_type}")
    return model_dir / model_name, model_dir, pretrained_path


def main():
    parser = argparse.ArgumentParser(description="Train and Evaluate a Deep Learning Model")
    parser.add_argument("--eval_type", type=str, required=True, choices=["20datasets", "ptm", "aa_glycine"],
                        help="Specify the evaluation type(20datasets, ptm, aa_glycine).")
    parser.add_argument("--dataset_name", type=str, required=True, help="Specify the dataset name.")
    parser.add_argument("--test_aa", type=str, default=None, help="Specify the test amino acid (if applicable).")
    parser.add_argument("--save_results", action="store_true", help="Save evaluation results.")
    parser.add_argument("--train", action="store_true", help="Train a new model. If not set, it will only evaluate.")
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

    print(x_shape)

    # Initialize model
    model = MyNet(x_shape=x_shape, config=config).to(device)
    # Get model save path
    best_model_path, model_dir, pretrained_model_path = get_model_save_path(args.eval_type, args.dataset_name, args.test_aa)

    # Define loss function and optimizer
    loss_function = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    if args.train:
        # Weights & Biases setup
        wandb.init(config=config, project=args.eval_type,entity="alirezak2", name=args.dataset_name, mode="online")

        model_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

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
        torch.save(best_model_state, best_model_path)
        model_to_use = str(best_model_path)
        wandb.finish()
    else:
        # Load pre-trained model if training is skipped
        pretrained_model = str(pretrained_model_path)
        if not pretrained_model:
            raise FileNotFoundError(f"No pre-trained model found in {model_dir}. Please enable training first.")

        model.load_state_dict(torch.load(pretrained_model, map_location=device), strict=False)
        print(f"Loaded pre-trained model: {pretrained_model}")
        model_to_use = pretrained_model

    # Evaluate on the test set
    eval_results = evaluate_model(model, dataloader_test, dataloader_test_extra, loss_function, device, model_to_use,
                   args.eval_type, args.save_results)

    # Generate Figures
    make_figures(model, model_to_use, loss_function, dataloader_test, args.eval_type, dataloader_test_extra,
                 args.save_results, eval_results)


if __name__ == "__main__":
    main()
