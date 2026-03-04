#!/usr/bin/env python3
from __future__ import annotations

import argparse, os, re, sqlite3, csv
from typing import Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
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
    "Modified.Sequence","ModifiedSequence","modified_sequence","ModifiedPeptide",
    "modified_peptide","Peptide","Sequence"
]

def find_col(schema: pa.Schema, candidates: List[str]) -> Optional[str]:
    names = set(schema.names)
    for c in candidates:
        if c in names:
            return c
    return None

_mod_pat = re.compile(r"\(([^)]+)\)")

def parse_modseq_to_seq_mods(modseq: Optional[str]) -> Tuple[str, str]:
    if modseq is None:
        return "", ""
    s = str(modseq)
    s = re.sub(r"\d+$", "", s)  # drop trailing charge digits (e.g. ...ETR2)

    stripped = _mod_pat.sub("", s)
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
    ap.add_argument("--batch_size", type=int, default=200_000)
    ap.add_argument("--sqlite_tmp", default="")  # optional override
    args = ap.parse_args()

    in_path = os.path.expanduser(args.in_parquet)
    out_pq = os.path.expanduser(args.out_filtered_parquet)
    out_csv = os.path.expanduser(args.out_ideeplc_csv)

    os.makedirs(os.path.dirname(out_pq) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    dataset = ds.dataset(in_path, format="parquet")
    mod_col = args.modified_col.strip() or find_col(dataset.schema, MODSEQ_CANDIDATES)
    if not mod_col:
        raise RuntimeError("Could not detect modified sequence column")

    # Prepare output parquet schema (input schema + key column)
    out_schema = dataset.schema.append(pa.field(args.key_col_name, pa.string()))

    # SQLite for bounded-memory deduplication for iDeepLC
    sqlite_path = os.path.expanduser(args.sqlite_tmp.strip()) if args.sqlite_tmp.strip() else out_csv + ".sqlite"
    if os.path.exists(sqlite_path):
        os.remove(sqlite_path)
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE ideeplc (
            key TEXT PRIMARY KEY,
            seq TEXT NOT NULL,
            modifications TEXT NOT NULL
        )
    """)
    conn.commit()

    removed_rows = 0
    total_written = 0

    writer: Optional[pq.ParquetWriter] = None
    try:
        for batch in dataset.to_batches(batch_size=args.batch_size):
            tbl = pa.Table.from_batches([batch])

            # Arrow-native regex check: contains ')(UniMod'
            modseq_arr = pc.cast(tbl[mod_col], pa.string())
            bad = pc.match_substring_regex(modseq_arr, r"\)\(UniMod")
            removed_rows += int(pc.sum(pc.cast(bad, pa.int64())).as_py())

            good = pc.invert(bad)
            filtered = tbl.filter(good)
            if filtered.num_rows == 0:
                continue

            # Build key column (row-aligned). Keep null key if sequence parses empty.
            keys: List[Optional[str]] = []
            modseq_list = pc.cast(filtered[mod_col], pa.string()).to_pylist()

            to_insert = []
            for s in modseq_list:
                seq, modstr = parse_modseq_to_seq_mods(s)
                if not seq:
                    keys.append(None)
                    continue
                k = f"{seq}||{modstr}"
                keys.append(k)
                to_insert.append((k, seq, modstr))

            # Deduplicate for iDeepLC in SQLite (bounded RAM)
            if to_insert:
                cur.executemany("INSERT OR IGNORE INTO ideeplc(key, seq, modifications) VALUES (?, ?, ?)", to_insert)
                conn.commit()

            key_arr = pa.array(keys, type=pa.string())
            out_batch = filtered.append_column(args.key_col_name, key_arr)

            if writer is None:
                writer = pq.ParquetWriter(out_pq, out_schema)
            writer.write_table(out_batch)
            total_written += out_batch.num_rows
    finally:
        if writer is not None:
            writer.close()
        conn.commit()

        # Export iDeepLC CSV without loading everything into pandas
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seq", "modifications", "tr", "key"])
            for seq, mods, key in cur.execute("SELECT seq, modifications, key FROM ideeplc ORDER BY key"):
                w.writerow([seq, mods, 0.0, key])

        # Count unique rows
        unique_count = cur.execute("SELECT COUNT(*) FROM ideeplc").fetchone()[0]
        conn.close()

    print(f"Filtered parquet: {out_pq}")
    print(f"Removed rows (double-mod): {removed_rows}")
    print(f"Written rows (filtered parquet): {total_written:,}")
    print(f"Unique iDeepLC rows: {unique_count:,}")
    print(f"iDeepLC input CSV: {out_csv}")
    print(f"SQLite tmp (optional to delete): {sqlite_path}")

if __name__ == "__main__":
    main()