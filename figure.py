import os
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader
from evaluate import validate, evaluate_model


def make_figures(
    model: torch.nn.Module,
    model_path: str,
    loss_fn: torch.nn.Module,
    dataloader_test: DataLoader,
    eval_type: str,
    dataloader_extra: DataLoader = None,
    save_results: bool = False,
    eval_results: tuple = None
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
    :param eval_results: Evaluation results from `evaluate_model`.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set correct device

    # If eval_results is provided, use it instead of re-running evaluation
    if eval_results:
        loss_test, corr_test, output_test, y_test = eval_results
    else:
        # If eval_results is not provided, fall back to running evaluate_model (not recommended)
        loss_test, corr_test, output_test, y_test = evaluate_model(model, dataloader_test, dataloader_extra,
                                                                   loss_fn, torch.device("cpu"), model_path,
                                                                   eval_type, save_results)

    # Handle extra dataset for specific evaluation types
    output_extra, y_extra = None, None
    if eval_type in ["ptm", "aa_glycine"] and dataloader_extra is not None:
        loss_extra, corr_extra, output_extra, y_extra = validate(model, dataloader_extra, loss_fn, device)
        print(f"\nExtra Test Set Loss: {loss_extra:.4f}, Correlation: {corr_extra:.4f}\n")


    # Generate figures based on eval_type
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
    dataset_name = Path(model_path).parent.name
    plt.title(f"Dataset: {dataset_name}")  # Add dataset name as title
    plt.axis("scaled")
    ax.plot([0, max_value], [0, max_value], ls="--", c=".5")
    plt.xlim(0, max_value)
    plt.ylim(0, max_value)
    save_path = model_path.replace("best.pth", "scatter_plot.png")
    plt.savefig(save_path, dpi=300)
    # plt.show()


def plot_ptm(y_test, output_test, y_test_no_mod, output_test_no_mod, model_path):
    """Generate scatter plot for PTM evaluation."""
    mae_test = mean_absolute_error(y_test, output_test) * 60
    mae_no_mod = mean_absolute_error(y_test_no_mod, output_test_no_mod) * 60
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
    save_path = model_path.replace(".pth", "_ptm.png")
    plt.savefig(save_path, dpi=300)
    # plt.show()



def plot_aa_glycine(model_path):
    """Generate AA glycine evaluation figure."""
    aas = ["A", "C", "D", "E", "F", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    general_path = os.path.dirname(model_path) + os.sep
    paths = [general_path + aa + "_best_results.csv" for aa in aas]
    dataset_name = os.path.basename(os.path.dirname(model_path))

    aa_colors = {
        "R": "C2", "H": "C2", "K": "C2",
        "E": "C3", "D": "C3",
        "S": "C1", "T": "C1", "N": "C1", "Q": "C1",
        "C": "C5", "G": "C5", "P": "C5",
        "A": "C0", "I": "C0", "L": "C0", "M": "C0",
        "F": "C0", "W": "C0", "Y": "C0", "V": "C0"
    }

    # Number of training peptides
    dataset_name_data = dataset_name.split("_")[0]
    num_train_pep = {
        aa: pd.read_csv(f"../data/modified_glycine_evaluation/{dataset_name_data}_{aa}_train.csv").shape[0]
        for aa in aas
    }
    max_train_set = round(max(num_train_pep.values()), -3)

    plt.figure(figsize=(6, 6.5))
    rs, r_gs = [], []

    print('AA: Encoded, Not Encoded')
    for aa in aas:
        file_path = os.path.join(general_path, f"{aa}_best_results.csv")

        try:
            df = pd.read_csv(file_path)

            # Rename incorrect column if necessary
            if "# y_test" in df.columns:
                df.rename(columns={"# y_test": "y_test"}, inplace=True)

            r = sum(abs(df["output_test"] - df["y_test"])) / len(df.index)
            r_glyc = sum(abs(df["output_test_g"] - df["y_test"])) / len(df.index)

            print(aa, ':', r, r_glyc)
            rs.append(r)
            r_gs.append(r_glyc)

            col_point = aa_colors.get(aa, "grey")

            # Scatter plot
            plt.scatter(r, r_glyc,
                        s=(num_train_pep[aa] / (max_train_set * 0.03)) ** 2,
                        facecolors=col_point,
                        linewidths=2,
                        alpha=0.1,
                        edgecolors=col_point)

            plt.scatter(r, r_glyc,
                        s=(num_train_pep[aa] / (max_train_set * 0.03)) ** 2,
                        facecolors="none",
                        linewidths=2,
                        edgecolors=col_point)

            # Annotate with amino acid labels
            plt.annotate(aa, xy=(r, r_glyc), fontsize=8, verticalalignment='center', horizontalalignment='center')

        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except KeyError as e:
            print(f"Missing column in {file_path}: {e}")

    # Set limits and plot diagonal reference line
    min_val = min(min(rs), min(r_gs))
    max_val = max(max(rs), max(r_gs))
    range_val = max_val - min_val

    plt.plot([min_val - 0.05 * range_val, max_val + 0.05 * range_val],
             [min_val - 0.05 * range_val, max_val + 0.05 * range_val],
             c="grey", linestyle="--", linewidth=0.5, zorder=0)

    plt.xlim(min_val - 0.05 * range_val, max_val + 0.05 * range_val)
    plt.ylim(min_val - 0.05 * range_val, max_val + 0.05 * range_val)

    # Legend for number of training peptides
    percentages = [0.25, 0.5, 0.75, 1]
    for num_train in [int(max_train_set * perc) for perc in percentages]:
        plt.scatter([], [], c='k', alpha=0.3, s=(num_train / (max_train_set * 0.03)) ** 2,
                    label=str(num_train))

    plt.legend(scatterpoints=1, frameon=False, labelspacing=1, title='Number of training peptides')

    # Labels and title
    plt.xlabel("MAE encoding the amino acid (min)")
    plt.ylabel("MAE encoding the amino acid as glycine (min)")

    # Save and show plot
    save_path = os.path.join(general_path, f"{dataset_name}_best.png")
    plt.savefig(save_path, dpi=300)
    # plt.show()
