#!/usr/bin/env python3
"""Backfill MLflow nested trial runs from trial_summary.json."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.mlflow_study import backfill_study_from_summary
from dpo.train.paths import EXPERIMENT_NAME, MLRUNS_DIR
from dpo.train.study_report import write_study_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLflow runs from trial_summary.json")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--parent-run-name",
        type=str,
        default=None,
        help="Attach nested runs to an existing parent run by name (default: new review parent)",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Do not regenerate study_report.html",
    )
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Not found: {summary_path}")

    if not args.skip_html:
        report = write_study_report(summary_path)
        print(f"HTML report: file://{report.resolve()}")

    parent_id = backfill_study_from_summary(
        summary_path,
        parent_run_name=args.parent_run_name,
    )
    uri = f"file://{MLRUNS_DIR.resolve()}"
    print(f"MLflow tracking: {uri}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Parent run id: {parent_id}")
    print(f"\nmlflow ui --backend-store-uri {uri} --port 5001")
    print("Open the review parent run → expand trial-* children → Compare runs")


if __name__ == "__main__":
    main()
