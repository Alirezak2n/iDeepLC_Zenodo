#!/usr/bin/env python3
from __future__ import annotations

import argparse, os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

RT_CANDIDATES = ["RT","iRT","PredictedRT","Predicted.RT","rt","irt"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_filtered_parquet", required=True)
    ap.add_argument("--ideeplc_output_csv", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--key_col_name", default="ideeplc_key")
    ap.add_argument("--rt_col", default="")
    ap.add_argument("--pred_col", default="predictions")  # your output column
    args = ap.parse_args()

    in_pq = os.path.expanduser(args.in_filtered_parquet)
    out_pq = os.path.expanduser(args.out_parquet)
    pred_csv = os.path.expanduser(args.ideeplc_output_csv)

    table = pq.read_table(in_pq)
    schema = table.schema

    if args.key_col_name not in schema.names:
        raise RuntimeError(f"Key column '{args.key_col_name}' not found in parquet. Re-run the prepare script.")

    # Detect RT col if not provided
    rt_col = args.rt_col.strip()
    if not rt_col:
        for c in RT_CANDIDATES:
            if c in schema.names:
                rt_col = c
                break
    if not rt_col:
        raise RuntimeError("Could not detect RT column to replace; pass --rt_col explicitly.")

    pred_df = pd.read_csv(pred_csv, sep=None, engine="python")  # auto-detect delimiter
    # Must contain either 'key' or (seq, modifications) to build key
    if "key" in pred_df.columns:
        key_series = pred_df["key"].astype(str)
    elif "seq" in pred_df.columns and "modifications" in pred_df.columns:
        key_series = pred_df["seq"].astype(str) + "||" + pred_df["modifications"].astype(str)
    else:
        raise RuntimeError("Predictions CSV must contain either 'key' or ('seq' and 'modifications').")

    if args.pred_col not in pred_df.columns:
        raise RuntimeError(f"Predictions CSV missing '{args.pred_col}'. Columns: {list(pred_df.columns)}")

    pred_map = pd.DataFrame({"ideeplc_key": key_series, "rt_pred": pred_df[args.pred_col]})
    pred_map = pred_map.dropna(subset=["rt_pred"]).drop_duplicates(subset=["ideeplc_key"])

    # Map onto parquet rows
    keys = table.column(args.key_col_name).to_pandas().astype(str)
    merged = pd.DataFrame({"ideeplc_key": keys}).merge(pred_map, on="ideeplc_key", how="left")

    if merged["rt_pred"].isna().any():
        missing = int(merged["rt_pred"].isna().sum())
        # show a few unmatched keys for debugging
        example = merged.loc[merged["rt_pred"].isna(), "ideeplc_key"].head(10).tolist()
        raise RuntimeError(f"{missing} rows did not match predictions. Example keys: {example}")

    # Replace RT column in Arrow table
    old_rt = table.column(rt_col)
    old_type = old_rt.type

    new_rt = merged["rt_pred"].to_numpy(dtype=np.float64)
    if pa.types.is_float32(old_type):
        new_rt = new_rt.astype(np.float32)
        new_arr = pa.array(new_rt, type=pa.float32())
    elif pa.types.is_float64(old_type):
        new_arr = pa.array(new_rt, type=pa.float64())
    else:
        # fallback: store float64
        new_arr = pa.array(new_rt, type=pa.float64())

    idx = schema.get_field_index(rt_col)
    out_table = table.set_column(idx, rt_col, new_arr)

    os.makedirs(os.path.dirname(out_pq) or ".", exist_ok=True)
    pq.write_table(out_table, out_pq)

    print(f"Replaced RT column '{rt_col}' for {len(new_rt):,} rows.")
    print(f"Output written: {out_pq}")

if __name__ == "__main__":
    main()