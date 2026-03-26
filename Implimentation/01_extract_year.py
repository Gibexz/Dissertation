# 01_extract_year.py
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, List

import pandas as pd


def open_json(path: Path) -> Dict[str, Any]:
    """Open .json or .json.gz and return parsed dict."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], keys: List[str], default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def extract_best_cvss_v3(metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Return (baseScore, baseSeverity, version) using:
    - prefer cvssMetricV31
    - else cvssMetricV30
    If multiple metric entries exist, select the maximum baseScore (conservative).
    """
    def best_from(metric_list: Any, version: str):
        if not isinstance(metric_list, list) or not metric_list:
            return (None, None, None)

        best_score = None
        best_sev = None
        for m in metric_list:
            cvss_data = safe_get(m, ["cvssData"], {}) or {}
            score = cvss_data.get("baseScore")
            sev = cvss_data.get("baseSeverity")
            if isinstance(score, (int, float)):
                score = float(score)
                if best_score is None or score > best_score:
                    best_score = score
                    best_sev = sev if isinstance(sev, str) else None

        if best_score is None:
            return (None, None, None)
        return (best_score, best_sev, version)

    score, sev, ver = best_from(metrics.get("cvssMetricV31"), "3.1")
    if score is not None:
        return score, sev, ver

    score, sev, ver = best_from(metrics.get("cvssMetricV30"), "3.0")
    if score is not None:
        return score, sev, ver

    return None, None, None


def iter_rows(nvd_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    NVD JSON 2.x structure:
    top-level: {"vulnerabilities": [{"cve": {...}}, ...]}
    """
    vulns = nvd_json.get("vulnerabilities", [])
    if not isinstance(vulns, list):
        return

    for item in vulns:
        if not isinstance(item, dict):
            continue
        cve = item.get("cve", {})
        if not isinstance(cve, dict):
            continue

        metrics = cve.get("metrics", {})
        metrics = metrics if isinstance(metrics, dict) else {}

        base_score, base_sev, cvss_ver = extract_best_cvss_v3(metrics)

        yield {
            "cve_id": cve.get("id"),
            "published": cve.get("published"),
            "vulnStatus": cve.get("vulnStatus"),
            "baseScore": base_score,
            "baseSeverity": base_sev,
            "cvssVersion": cvss_ver,
        }


def extract_year_file(input_path: Path) -> pd.DataFrame:
    raw = open_json(input_path)
    df = pd.DataFrame(list(iter_rows(raw)))
    return df


def main():
    ap = argparse.ArgumentParser(description="Extract raw fields from one NVD yearly JSON feed.")
    ap.add_argument("--input", required=True, help="Path to a single yearly NVD JSON(.gz) file.")
    ap.add_argument("--output", required=True, help="Output CSV path for extracted (unfiltered) table.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = extract_year_file(in_path)
    df.to_csv(out_path, index=False)
    print(f"[EXTRACT] Wrote {len(df):,} rows -> {out_path}")


if __name__ == "__main__":
    main()