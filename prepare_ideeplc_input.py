#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

UNIMOD_ID_TO_NAME: Dict[int, str] = {
    58: "Propionyl",
    36: "Dimethyl",
    37: "Trimethyl",
    1:  "Acetyl",
    34: "Methyl",
    35: "Oxidation",
    21: "Phospho",
    1289: "Butyryl",
    2114: "Lactyl",
    64: "Succinyl",
    4 : "Carbamidomethyl",
}

MODSEQ_CANDIDATES = [
    "Modified.Sequence","ModifiedSequence","modified_sequence","ModifiedPeptide","modified_peptide","Peptide","Sequence"
]

def find_col(schema: pa.Schema, candidates: List[str]) -> Optional[str]:
    names = set(schema.names)
    for c in candidates:
        if c in names:
            return c
    return None

def parse_modseq_to_seq_mods(modseq: str) -> Tuple[str, str]:
    s = str(modseq)
    s = re.sub(r"\d+$", "", s)  # drop trailing charge digits (e.g. ...ETR2)
    mod_pat = re.compile(r"\(([^)]+)\)")

    stripped = mod_pat.sub("", s)
    stripped = re.sub(r"[^A-Z]", "", stripped)

    mods: List[Tuple[int, str]] = []
    aa_index = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if "A" <= ch <= "Z":
            aa_index += 1
            i += 1
            continue
        if ch == "(":
            j = s.find(")", i + 1)
            if j == -1:
                break
            token = s[i + 1 : j].strip()
            m = re.search(r"UniMod:(\d+)", token)
            if m:
                uid = int(m.group(1))
                name = UNIMOD_ID_TO_NAME.get(uid, f"UniMod:{uid}")
            else:
                name = token
            pos = 0 if aa_index == 0 else aa_index
            mods.append((pos, name))
            i = j + 1
            continue
        i += 1

    mod_str = "|".join([f"{pos}|{name}" for pos, name in mods])
    return stripped, mod_str

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_filtered_parquet", required=True)
    ap.add_argument("--out_ideeplc_csv", required=True)
    ap.add_argument("--modified_col", default="")
    ap.add_argument("--key_col_name", default="ideeplc_key")
    args = ap.parse_args()

    in_path = os.path.expanduser(args.in_parquet)
    out_pq = os.path.expanduser(args.out_filtered_parquet)
    out_csv = os.path.expanduser(args.out_ideeplc_csv)

    table = pq.read_table(in_path)
    mod_col = args.modified_col.strip() or find_col(table.schema, MODSEQ_CANDIDATES)
    if not mod_col:
        raise RuntimeError("Could not detect modified sequence column")

    modseq = table.column(mod_col).to_pandas().astype(str)

    # remove double-mod-on-same-site rows: contains ')(UniMod'
    bad = modseq.str.contains(r"\)\(UniMod", regex=True, na=False)
    filtered = table.filter(pa.array((~bad).to_numpy()))

    f_modseq = filtered.column(mod_col).to_pandas().astype(str)

    seqs, mods, keys = [], [], []
    for s in f_modseq.tolist():
        seq, modstr = parse_modseq_to_seq_mods(s)
        if not seq:
            continue
        seqs.append(seq)
        mods.append(modstr)
        keys.append(f"{seq}||{modstr}")

    # Attach key column to parquet (row-aligned)
    key_arr = pa.array(keys)
    out_table = filtered.append_column(args.key_col_name, key_arr)

    os.makedirs(os.path.dirname(out_pq) or ".", exist_ok=True)
    pq.write_table(out_table, out_pq)

    # Deduplicate for iDeepLC
    df = pd.DataFrame({"seq": seqs, "modifications": mods})
    df["key"] = df["seq"] + "||" + df["modifications"]
    df = df.drop_duplicates(subset=["key"]).reset_index(drop=True)
    df["tr"] = 0.0  # dummy for prediction

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df[["seq","modifications","tr","key"]].to_csv(out_csv, index=False)

    print(f"Filtered parquet: {out_pq}")
    print(f"Removed rows (double-mod): {int(bad.sum())}")
    print(f"Unique iDeepLC rows: {len(df):,}")
    print(f"iDeepLC input CSV: {out_csv}")

if __name__ == "__main__":
    main()