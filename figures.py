import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from evaluate import validate


def make_figures(
    model: torch.nn.Module,
    model_path: str,
    loss_fn: torch.nn.Module,
    dataloader_test: DataLoader,
    eval_type: str,
    dataloader_extra: DataLoader = None,
    save_results: bool = False
):
    """
    Generate figures based on evaluation type.

    :param model: Trained PyTorch model.
    :param model_path: Path to the trained model.
    :param loss_fn: Loss function for evaluation.
    :param dataloader_test: DataLoader for the test set.
    :param eval_type: Evaluation type (`20datasets`, `ptm`, `aa_glycine`).
    :param dataloader_extra: Additional test DataLoader (for `ptm` and `aa_glycine`).
    :param save_results: Whether to save results as a CSV file.
    """
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    # Validate the model on the primary test set
    loss_test, corr_test, output_test, y_test = validate(model, dataloader_test, loss_fn, torch.device("cpu"))

    print(f"\nTest Set Loss: {loss_test:.4f}, Correlation: {corr_test:.4f}\n")

    # Handle extra dataset for specific evaluation types
    if eval_type in ["ptm", "aa_glycine"] and dataloader_extra is not None:
        loss_extra, corr_extra, output_extra, y_extra = validate(model, dataloader_extra, loss_fn, torch.device("cpu"))
        print(f"\nExtra Test Set Loss: {loss_extra:.4f}, Correlation: {corr_extra:.4f}\n")
    else:
        output_extra, y_extra = None, None

    # Save results
    if save_results:
        filename = model_path.replace(".pth", "_results.csv")

        # If there is an extra dataset, include it in results
        if output_extra is not None:
            data_to_save = np.column_stack((y_test, output_test, output_extra))
            header = "y_test,output_test,output_extra"
        else:
            data_to_save = np.column_stack((y_test, output_test))
            header = "y_test,output_test"

        np.savetxt(filename, data_to_save, delimiter=",", header=header, fmt="%.6f")
        print(f"Results saved to {filename}")

    # Generate and save figures
    if eval_type == "20datasets":
        plot_20datasets(y_test, output_test, model_path)
    elif eval_type == "ptm":
        plot_ptm(y_test, output_test, y_extra, output_extra, model_path)
    elif eval_type == "aa_glycine":
        plot_aa_glycine(model_path)
    else:
        print("Unknown evaluation type, skipping figure generation.")


def plot_20datasets(y_test, output_test, model_path):
    """Generate scatter plot for 20datasets evaluation."""
    mae_test = mean_absolute_error(y_test, output_test)
    max_value = max(output_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, output_test, c="b", label=f"MAE: {mae_test:.3f}, R: {np.corrcoef(y_test, output_test)[0, 1]:.3f}", s=3)
    plt.legend(loc="upper left")
    plt.xlabel("Observed Retention Time")
    plt.ylabel("Predicted Retention Time")
    plt.axis("scaled")
    ax.plot([0, max_value], [0, max_value], ls="--", c=".5")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.savefig(model_path.replace(".pth", "_20datasets.png"), dpi=300)
    print(f"Figure saved: {model_path.replace('.pth', '_20datasets.png')}")


def plot_ptm(y_test, output_test, y_test_no_mod, output_test_no_mod, model_path):
    """Generate scatter plot for PTM evaluation."""
    mae_test = mean_absolute_error(y_test, output_test)
    mae_no_mod = mean_absolute_error(y_test_no_mod, output_test_no_mod)
    max_value = max(output_test)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test_no_mod, output_test_no_mod, c="r", label=f"Not Encoded (MAE: {mae_no_mod:.3f})", s=3)
    ax.scatter(y_test, output_test, c="b", label=f"Encoded (MAE: {mae_test:.3f})", s=3)
    plt.legend(loc="upper left")
    plt.xlabel("Observed Retention Time")
    plt.ylabel("Predicted Retention Time")
    plt.axis("scaled")
    ax.plot([0, max_value], [0, max_value], ls="--", c=".5")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    plt.savefig(model_path.replace(".pth", "_ptm.png"), dpi=300)
    print(f"Figure saved: {model_path.replace('.pth', '_ptm.png')}")


def plot_aa_glycine(model_path):
    """Generate boxplot for AA glycine evaluation."""
    aas = ["A", "C", "D", "E", "F", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    paths = ["saved_models/aa_cv/diamino1_atom_stan_mollog_newMeta/" + aa + "_best_results.csv" for aa in aas]

    data = []
    for aa, path in zip(aas, paths):
        try:
            df = pd.read_csv(path)
            mae = mean_absolute_error(df["y_test"], df["output_test"])
            mae_glyc = mean_absolute_error(df["y_test"], df["output_extra"])
            data.append([aa, mae, "AA"])
            data.append([aa, mae_glyc, "Glycine"])
        except FileNotFoundError:
            print(f"File not found: {path}")

    df_aa = pd.DataFrame(data, columns=["Amino Acid", "Relative MAE", "Type"])
    plt.figure(figsize=(18, 6))
    sns.boxplot(x="Amino Acid", y="Relative MAE", hue="Type", data=df_aa, palette=["grey", "#276ba6"])
    plt.xlabel("Amino Acid", fontsize=12)
    plt.ylabel("Relative MAE", fontsize=12)
    plt.legend(title="Encoding Type")
    plt.savefig(model_path.replace(".pth", "_aa_glycine.png"), dpi=300)
    print(f"Figure saved: {model_path.replace('.pth', '_aa_glycine.png')}")
