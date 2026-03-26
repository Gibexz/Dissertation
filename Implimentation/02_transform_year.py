# 02_transform_year.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


STATUS_KEEP = {"Analyzed", "Modified"}
MIN_SCORE = 7.0


def transform_year(df: pd.DataFrame) -> pd.DataFrame:
    # Parse published timestamp
    df["published"] = pd.to_datetime(df["published"], errors="coerce", utc=True)

    # Basic integrity
    df["cve_id"] = df["cve_id"].astype(str).str.strip()
    df = df.dropna(subset=["cve_id", "published", "vulnStatus", "baseScore"])

    # Actionable statuses only
    df = df[df["vulnStatus"].isin(STATUS_KEEP)]

    # High/Critical threshold
    df["baseScore"] = df["baseScore"].astype(float)
    df = df[df["baseScore"] >= MIN_SCORE]

    # Keep only necessary columns (optional)
    df = df[["cve_id", "published", "vulnStatus", "baseScore", "baseSeverity", "cvssVersion"]]

    # Year-level uniqueness (safe, typically no effect, but prevents accidental duplication inside a file)
    df = df.sort_values(["cve_id", "published"]).drop_duplicates(subset=["cve_id"], keep="first")

    return df


def main():
    ap = argparse.ArgumentParser(description="Transform/clean one extracted yearly NVD table.")
    ap.add_argument("--input", required=True, help="Path to extracted CSV (from 01_extract_year.py).")
    ap.add_argument("--output", required=True, help="Output CSV path for transformed yearly table.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    clean = transform_year(df)

    clean.to_csv(out_path, index=False)
    print(f"[TRANSFORM] Wrote {len(clean):,} rows -> {out_path}")


if __name__ == "__main__":
    main()