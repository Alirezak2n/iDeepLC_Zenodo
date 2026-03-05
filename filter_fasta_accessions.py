#!/usr/bin/env python3
"""
Extract UniProt accessions from the 'Protein Accessions' column in exactly 3 files
and filter a FASTA file to only records whose header starts with '>sp|'
and whose accession (between the first and second '|') is in that set.

Input tables supported: CSV / TSV / TXT (delimiter auto-detected), Excel (.xlsx/.xls)
FASTA parsing: manual, based on '>sp|' boundaries as requested.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Set, Iterable, Optional

import pandas as pd


# UniProt accessions:
# - 6-char (e.g., O00165, Q16777, P36578)
# - 10-char newer (e.g., A0A024RBG1)
UNIPROT_ACC_RE = re.compile(
    r"\b([A-NR-Z][0-9][A-Z0-9]{3}[0-9]|[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9][A-Z0-9]{8})\b"
)


def read_table_any(path: Path) -> pd.DataFrame:
    """Read CSV/TSV/TXT (delimiter inferred) or Excel."""
    suf = path.suffix.lower()
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    # auto-detect delimiter for text
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallbacks
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path, sep=",")


def extract_accessions(values: Iterable[object]) -> Set[str]:
    """Extract all UniProt accessions from mixed cells, including semicolon-separated lists."""
    accs: Set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == "nan":
            continue
        for m in UNIPROT_ACC_RE.finditer(s):
            accs.add(m.group(1))
    return accs


def parse_sp_fasta_records(fasta_path: Path):
    """
    Generator yielding (accession, record_lines) for records starting with '>sp|'.
    Accession is taken as text between first and second '|'.
    Record ends at next '>sp|' or EOF.
    Non-sp headers (e.g., '>tr|', '>') are ignored entirely.
    """
    current_acc: Optional[str] = None
    current_lines: list[str] = []

    def flush():
        nonlocal current_acc, current_lines
        if current_acc is not None and current_lines:
            yield current_acc, current_lines
        current_acc = None
        current_lines = []

    with fasta_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">sp|"):
                # new record boundary
                if current_acc is not None:
                    # yield previous
                    for item in flush():
                        yield item

                # parse accession between first and second |
                # format: >sp|O00165|HAX1_HUMAN ...
                parts = line.split("|", 2)
                # parts[0] = ">sp", parts[1] = accession, parts[2] = rest...
                if len(parts) >= 3:
                    current_acc = parts[1].strip()
                    current_lines = [line]
                else:
                    # malformed; ignore this header and do not start a record
                    current_acc = None
                    current_lines = []
            else:
                # sequence line: only keep if we're currently inside an sp record
                if current_acc is not None:
                    current_lines.append(line)

    # flush last record
    if current_acc is not None and current_lines:
        yield current_acc, current_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter a FASTA to proteins listed in 3 tables.")
    ap.add_argument("--inputs", nargs=3, required=True,
                    help="Exactly three input files (CSV/TSV/TXT/Excel).")
    ap.add_argument("--fasta", required=True, help="Input FASTA file.")
    ap.add_argument("--out_fasta", default="filtered_sp.fasta",
                    help="Output FASTA (default: filtered_sp.fasta).")
    ap.add_argument("--out_accessions", default="accessions.txt",
                    help="Write extracted accessions to this file (default: accessions.txt).")
    ap.add_argument("--column", default="Protein Accessions",
                    help="Column name (default: 'Protein Accessions').")
    args = ap.parse_args()

    # 1) Extract accessions from the three tables
    wanted: Set[str] = set()
    for p_str in args.inputs:
        p = Path(p_str)
        df = read_table_any(p)
        if args.column not in df.columns:
            raise KeyError(
                f"Column '{args.column}' not found in {p}. "
                f"Available columns: {list(df.columns)}"
            )
        wanted |= extract_accessions(df[args.column].values)

    if not wanted:
        raise RuntimeError("No accessions extracted from input files.")

    Path(args.out_accessions).write_text("\n".join(sorted(wanted)) + "\n", encoding="utf-8")

    # 2) Stream FASTA and write only matching sp records
    fasta_path = Path(args.fasta)
    out_path = Path(args.out_fasta)

    total_sp = 0
    kept = 0

    with out_path.open("w", encoding="utf-8") as out:
        for acc, lines in parse_sp_fasta_records(fasta_path):
            total_sp += 1
            if acc in wanted:
                out.writelines(lines)
                # ensure newline separation (FASTA typically already has it)
                if lines and not lines[-1].endswith("\n"):
                    out.write("\n")
                kept += 1

    print(f"Extracted accessions: {len(wanted)}")
    print(f"FASTA sp-records scanned: {total_sp}")
    print(f"FASTA records written:    {kept}")
    print(f"Output FASTA:             {out_path}")
    print(f"Output accession list:    {args.out_accessions}")


if __name__ == "__main__":
    main()