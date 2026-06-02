# iDeepLC: A deep learning-based retention time predictor for unseen modified peptides with a novel encoding system

> **Zenodo version 2.** This release adds a 14-PTM benchmark (Prosit 2020 modification analysis data), pretrained-model evaluation outputs with a consolidated metrics table, a DIA-NN-to-iDeepLC preparation and prediction-injection pipeline, and additional retrained model checkpoints.

## Overview

iDeepLC is a deep learning-based tool for retention time prediction in proteomics. It supports multiple evaluation types, including **20 datasets evaluation**, **modified glycine evaluation**, and **PTM evaluation** (14 PTMs).

The repository provides tools for **training models**, **evaluating retention time predictions**, **generating figures** for analysis, and **applying pretrained models** to new data.

## Repository Structure

```
Zenodo/
│── data/                              # Input datasets for evaluation
│   ├── 20_datasets_evaluation/        # Data for the 20 dataset evaluation (20 subfolders)
│   ├── PTM_evaluation/                # Data for PTM evaluation (14 PTM subfolders)
│   ├── modified_glycine_evaluation/   # Data for modified glycine evaluation
│   ├── 14ptm_raw/                     # Raw Prosit 2020 PTM modanalysis CSVs (train/valid/test per PTM)
│   │   └── enriched/                  # Enriched versions of the 14 PTM CSVs
│   ├── structure_feature/             # Amino acid and PTM structure information
│
│── iDeepLC_Zenodo/                    # Main implementation
│   ├── config.py                      # Model configuration (epochs, batch size, etc.)
│   ├── data_initialize.py             # Data loading and preprocessing
│   ├── model.py                       # Model architecture (MyNet)
│   ├── train.py                       # Training functions
│   ├── evaluate.py                    # Model evaluation functions
│   ├── figure.py                      # Functions to generate evaluation figures
│   ├── utilities.py                   # Input data generator
│   ├── main.py                        # Main script to train/evaluate models

│   ├── prepare_ideeplc_input.py       # Prepare iDeepLC input from a DIA-NN Parquet report
│   ├── ideeplc_rt_to_parquet.py       # Inject iDeepLC predictions back into a filtered Parquet
│   ├── filter_fasta_accessions.py     # Filter a FASTA file to accessions present in result tables
│   ├── plots_generator.ipynb          # Notebook for generating manuscript figures
│   ├── ideeplc_pretrained_out/        # Pretrained-model evaluation outputs
│   │   ├── <dataset>/                 # Per-dataset test_predictions.csv (20 datasets)
│   │   ├── 14ptm/<PTM>/               # Per-PTM prediction outputs
│   │   └── benchmark_metrics.csv      # Consolidated metrics (MAE, rMAE, Pearson, Spearman, ...)
│   ├── MyNet_styled.svg               # Network architecture diagram
│   ├── requirements.txt               # Required dependencies
│   ├── README.md                      # GitHub documentation
│   ├── LICENSE                        # License file
│   └── wandb/                         # Weights & Biases run logs (if used)
│
│── saved_model/                       # Pretrained / retrained models
│   ├── 20_datasets_evaluation/        # Models per dataset (incl. retrained _0204 / _0205 variants)
│   ├── PTM_evaluation/                # PTM evaluation models
│   ├── PTM_evaluation_DeepLC/         # DeepLC PTM output
│   ├── modified_glycine_evaluation/   # Models per dataset for glycine evaluation
│   ├── modified_glycine_evaluation_DeepLC/  # DeepLC glycine output
│
│── README.md                          # Project documentation
```

---

## How to Use

### 1. Generating figures (manuscript plots)

Use the provided Jupyter Notebook:

```sh
cd iDeepLC_Zenodo
jupyter notebook plots_generator.ipynb
```

This generates **all the figures presented in the manuscript**.

---

### 2. Training & evaluation

The `main.py` script lets you **train models**, **evaluate them**, and **generate figures**.

#### Run training + evaluation

Choose one of the three evaluation types (`20datasets`, `ptm`, `aa_glycine`) and one dataset based on the evaluation type. For `aa_glycine` you must also choose an amino acid.

```sh
python main.py --eval_type 20datasets --dataset_name arabidopsis --train
python main.py --eval_type aa_glycine --dataset_name arabidopsis --test_aa A --train
python main.py --eval_type ptm --dataset_name Acetyl --train
python main.py --eval_type ptm --dataset_name Acetyl --train --save_results
```

- For `aa_glycine`, figures are generated only after **all amino acids** are processed.

#### Run only evaluation + figure generation

To **use pretrained models** for evaluation and figure generation:

```sh
python main.py --eval_type 20datasets --dataset_name arabidopsis
```

This loads the pretrained model (for example `saved_model/20_datasets_evaluation/arabidopsis/best.pth`), evaluates it, and **generates figures**.

---

### 3. Batch evaluation of pretrained models (new in v2)

`run_all_.py` evaluates pretrained models across the full suite for a given mode and writes per-dataset predictions plus a consolidated metrics table to `ideeplc_pretrained_out/`.

```sh
# Evaluate all 20 datasets
python run_all_.py --mode 20datasets --all

# Evaluate a single dataset
python run_all_.py --mode 20datasets --name arabidopsis

# Evaluate all 14 PTMs
python run_all_.py --mode 14ptm --all
```

Outputs:
- `ideeplc_pretrained_out/<name>/test_predictions.csv` per dataset or PTM.
- `ideeplc_pretrained_out/benchmark_metrics.csv` with `MAE`, `rMAE`, `Pearson`, `Spearman`, and related fields.

The 14 PTMs supported are: Acetyl, Carbamidomethyl, Crotonyl, Deamidated, Dimethyl, Formyl, Malonyl, Methyl, Nitro, Oxidation, Phospho, Propionyl, Succinyl, Trimethyl.

---

### 4. Applying iDeepLC to a DIA-NN report (new in v2)

A three-step pipeline lets you predict retention times for peptides in a DIA-NN Parquet report and write the predictions back.

```sh
# 1. Prepare iDeepLC input from a DIA-NN Parquet report
python prepare_ideeplc_input.py \
    --in_parquet report.parquet \
    --out_filtered_parquet filtered.parquet \
    --out_ideeplc_csv ideeplc_input.csv

# 2. Run iDeepLC prediction on ideeplc_input.csv (using main.py / run_all_.py)

# 3. Inject the predictions back into the filtered Parquet
python ideeplc_rt_to_parquet.py \
    --in_filtered_parquet filtered.parquet \
    --ideeplc_output_csv ideeplc_predictions.csv \
    --out_parquet report_with_ideeplc_rt.parquet
```

`filter_fasta_accessions.py` filters a FASTA file to only the UniProt accessions present in result tables:

```sh
python filter_fasta_accessions.py --help
```

---

## Dependencies

To install dependencies:

```sh
pip install -r iDeepLC_Zenodo/requirements.txt
```

---

## Citation

If you use **iDeepLC** in your research, please cite our paper:

**iDeepLC: A deep learning-based retention time predictor for unseen modified peptides with a novel encoding system**
Alireza Nameni, Arthur Declercq, Ralf Gabriels, Robbe Devreese, Lennart Martens, Sven Degroeve, and Robbin Bouwmeester
2025
DOI: *(to be added)*
