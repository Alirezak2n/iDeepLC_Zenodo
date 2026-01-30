# ideeplc_eval_pretrained.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from config import get_config
from data_initialize import data_initialize
from model import MyNet
from evaluate import evaluate_model


# ============================
# Lists
# ============================

DATASETS_20 = [
    'arabidopsis', 'atlantis', 'swath', 'helahf', 'hela1h', 'hela2h',
    'lunahilic', 'lunasilica', 'heladeeprt', 'pancreas', 'plasma1h', 'plasma2h',
    'proteometoolsptm', 'proteometools', 'diahf', 'scx', 'yeastdeeprt', 'xbridge', 'yeast2h', 'yeast1h'
]

PTMS_14 = [
    "Acetyl", "Carbamidomethyl", "Crotonyl", "Deamidated", "Dimethyl", "Formyl", "Malonyl",
    "Methyl", "Nitro", "Oxidation", "Phospho", "Propionyl", "Succinyl", "Trimethyl",
]


# ============================
# Metrics
# ============================

def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmae(y_true, y_pred, ref_low, ref_high) -> float:
    denom = max(1e-12, (ref_high - ref_low))
    return mae(y_true, y_pred) / denom


def corrs(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    pr = pearsonr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    sr = spearmanr(y_true, y_pred)[0] if len(y_true) > 2 else np.nan
    return float(pr), float(sr)


# ============================
# Paths
# ============================

def mode_to_eval_type(mode: str) -> str:
    # Your project uses "ptm" for the 14ptm evaluation
    if mode == "20datasets":
        return "20datasets"
    if mode == "14ptm":
        return "ptm"
    raise ValueError(f"Unknown mode: {mode}")


def pretrained_model_path(mode: str, name: str) -> Path:
    """
    Mirrors your main.py pretrained path logic.
    """
    if mode == "20datasets":
        return Path(f"../saved_model/20_datasets_evaluation/{name}/best.pth")
    elif mode == "14ptm":
        return Path(f"../saved_model/PTM_evaluation/{name}_best.pth")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def out_dir_for(mode: str, base_out: Path, name: str) -> Path:
    """
    Similar layout to AlphaPeptDeep/Chronologer:
      - 20datasets: base_out/<name>/
      - 14ptm:      base_out/14ptm/<name>/
    """
    if mode == "20datasets":
        out_dir = base_out / name
    else:
        out_dir = base_out / "14ptm" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ============================
# Run one
# ============================

def run_one(mode: str, name: str, base_out: Path, device: torch.device, save_model_side_csv: bool) -> dict:
    eval_type = mode_to_eval_type(mode)
    model_path = pretrained_model_path(mode, name)

    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path.resolve()}")

    # Config is used for batch_size + model construction
    config = get_config(epoch=10)
    batch_size = config["batch_size"]

    # Data init (reuse your existing logic)
    data_loaders = data_initialize(eval_type, dataset_name=name, test_aa=None, batch_size=batch_size)

    dataloader_test_extra = None
    if eval_type == "ptm":
        # as in your main.py
        dataloader_train, dataloader_val, dataloader_test, dataloader_test_no_mod, x_shape = data_loaders
        dataloader_test_extra = dataloader_test_no_mod
    else:
        dataloader_train, dataloader_val, dataloader_test, x_shape = data_loaders

    # Model
    model = MyNet(x_shape=x_shape, config=config).to(device)

    # Evaluate (this loads weights internally again; keeping it identical to your pipeline)
    loss_fn = torch.nn.L1Loss()

    loss_test, corr_test, y_pred, y_true = evaluate_model(
        model=model,
        dataloader_test=dataloader_test,
        dataloader_extra=dataloader_test_extra,
        loss_fn=loss_fn,
        device=device,
        model_path=str(model_path),
        eval_type=eval_type,
        save_results=save_model_side_csv,  # your existing behavior if you want it
    )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    # rMAE reference from TEST distribution (since we do not reliably have raw train RTs here)
    ref_low = float(np.percentile(y_true_f, 5)) if len(y_true_f) else np.nan
    ref_high = float(np.percentile(y_true_f, 95)) if len(y_true_f) else np.nan

    out_dir = out_dir_for(mode, base_out, name)
    pd.DataFrame({"rt_true": y_true_f, "rt_pred": y_pred_f}).to_csv(out_dir / "test_predictions.csv", index=False)

    metrics = {
        "mode": mode,
        "name": name,
        "model_path": str(model_path.resolve()),
        "n_test_total": int(len(y_true)),
        "n_test_scored": int(len(y_true_f)),
        "loss_test": float(loss_test),
        "corr_test_validate": float(corr_test),
        "MAE": mae(y_true_f, y_pred_f) if len(y_true_f) else np.nan,
        "rMAE": rmae(y_true_f, y_pred_f, ref_low, ref_high) if len(y_true_f) else np.nan,
        "Pearson": corrs(y_true_f, y_pred_f)[0] if len(y_true_f) else np.nan,
        "Spearman": corrs(y_true_f, y_pred_f)[1] if len(y_true_f) else np.nan,
    }
    return metrics


# ============================
# Main
# ============================

def main():
    parser = argparse.ArgumentParser(description="Evaluate pretrained iDeepLC models on 20datasets or 14ptm (ptm).")
    parser.add_argument("--mode", required=True, choices=["20datasets", "14ptm"])
    parser.add_argument("--name", default=None, help="Single dataset/PTM name (optional if --all).")
    parser.add_argument("--all", action="store_true", help="Run the full suite for the selected mode.")
    parser.add_argument("--out_dir", default="ideeplc_pretrained_out", help="Base output folder.")
    parser.add_argument(
        "--save_model_side_csv",
        action="store_true",
        help="Also let evaluate_model() write its CSV next to the model file (your current behavior).",
    )
    args = parser.parse_args()

    if not args.all and not args.name:
        raise ValueError("Provide --name <dataset/PTM> or use --all.")

    names = DATASETS_20 if args.mode == "20datasets" else PTMS_14
    if args.all:
        run_names = names
    else:
        if args.name not in names:
            raise ValueError(f"{args.name} is not in the {args.mode} list.")
        run_names = [args.name]

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for name in run_names:
        print(f"[iDeepLC pretrained] {args.mode} -> {name}")
        rows.append(run_one(args.mode, name, base_out, device, args.save_model_side_csv))

    summary = pd.DataFrame(rows)
    if args.mode == "14ptm":
        summary_path = base_out / "14ptm" / "benchmark_metrics.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        summary_path = base_out / "benchmark_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
