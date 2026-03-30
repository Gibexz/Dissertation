# 03_merge_and_build_series.py
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def merge_years(transformed_dir: Path, pattern: str = "transformed_*.csv") -> pd.DataFrame:
    files = sorted(transformed_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {pattern} in {transformed_dir}")

    parts = []
    for f in files:
        parts.append(pd.read_csv(f, parse_dates=["published"]))
        print(f"[MERGE] Loaded {f.name}")

    df_all = pd.concat(parts, ignore_index=True)

    # Cross-year integrity: ensure one row per CVE ID (ETL duplicate safeguard)
    df_all["published"] = pd.to_datetime(df_all["published"], errors="coerce", utc=True)
    df_all = df_all.sort_values(["cve_id", "published"]).drop_duplicates(subset=["cve_id"], keep="first")

    return df_all


def build_weekly_ycrit(df_all: pd.DataFrame) -> pd.DataFrame:
    s = (
        df_all.set_index("published")
             .sort_index()
             .resample("W-MON")["cve_id"]
             .nunique()
             .rename("ycrit")
             .asfreq("W-MON", fill_value=0)
    )
    return s.reset_index().rename(columns={"published": "week"})


def main():
    ap = argparse.ArgumentParser(description="Merge transformed yearly datasets and build weekly Ycrit series.")
    ap.add_argument("--input_dir", required=True, help="Directory containing transformed_YYYY.csv files.")
    ap.add_argument("--output_dir", required=True, help="Directory to write merged dataset and weekly series.")
    ap.add_argument("--pattern", default="transformed_*.csv", help="Glob pattern for transformed yearly files.")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    merged = merge_years(in_dir, args.pattern)
    merged_out = out_dir / "nvd_cleaned_merged_2019_2023.csv"
    merged.to_csv(merged_out, index=False)
    print(f"[MERGE] Wrote merged cleaned dataset: {merged_out} ({len(merged):,} rows)")

    weekly = build_weekly_ycrit(merged)
    weekly_out = out_dir / "ycrit_weekly_w_mon.csv"
    weekly.to_csv(weekly_out, index=False)
    print(f"[SERIES] Wrote weekly series: {weekly_out} ({len(weekly):,} weeks)")


if __name__ == "__main__":
    main()