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
from dpo.train.study_report import write_study_report as write_study_report_v1
from dpo.train.study_report_v2 import write_study_report as write_study_report_v2


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLflow runs from trial_summary.json")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--parent-run-name",
        type=str,
        default=None,
        help="Parent run name (default: new optuna-parallel-review-* parent)",
    )
    parser.add_argument(
        "--replace-parent",
        action="store_true",
        help="Delete existing parent+children with --parent-run-name, then backfill fresh",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Do not regenerate HTML reports",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use study_report_v2.py → study_report_v2.html; MLflow parent name gets -v2 suffix",
    )
    args = parser.parse_args()
    summary_path = args.summary.resolve()
    if not summary_path.is_file():
        raise SystemExit(f"Not found: {summary_path}")

    run_dir = summary_path.parent
    report_paths: list[Path] = []
    if not args.skip_html:
        if args.v2:
            report = write_study_report_v2(
                summary_path, output_path=run_dir / "study_report_v2.html"
            )
            report_paths.append(report)
        else:
            report = write_study_report_v1(summary_path)
            report_paths.append(report)
        print(f"HTML report: file://{report_paths[0].resolve()}")
    elif args.v2:
        candidate = run_dir / "study_report_v2.html"
        if candidate.is_file():
            report_paths.append(candidate)
    else:
        candidate = run_dir / "study_report.html"
        if candidate.is_file():
            report_paths.append(candidate)

    parent_id = backfill_study_from_summary(
        summary_path,
        parent_run_name=args.parent_run_name,
        parent_name_suffix="v2" if args.v2 and not args.parent_run_name else "",
        report_paths=report_paths or None,
        replace_parent=args.replace_parent,
    )
    uri = f"file://{MLRUNS_DIR.resolve()}"
    print(f"MLflow tracking: {uri}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Parent run id: {parent_id}")
    print(f"\nmlflow ui --backend-store-uri {uri} --port 5001")
    print("Open the review parent run → expand trial-* children → Compare runs")


if __name__ == "__main__":
    main()
