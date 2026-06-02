#!/usr/bin/env python3
"""Regenerate trial_summary.json + study_report.html with resolved hyperparams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.optuna_parallel import enrich_summary_dict
from dpo.train.study_report import write_study_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich summary params and regenerate HTML")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="optuna-run-* directory containing trial_summary.json",
    )
    parser.add_argument(
        "--backfill-mlflow",
        action="store_true",
        help=(
            "Creates NEW duplicate MLflow review parents (optuna-parallel-review-*). "
            "Prefer updating disk only; use only if you explicitly want extra runs."
        ),
    )
    args = parser.parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "trial_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = enrich_summary_dict(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = write_study_report(summary_path)
    print(f"Updated {summary_path}")
    print(f"HTML: file://{report.resolve()}")

    if args.backfill_mlflow:
        from dpo.train.mlflow_study import backfill_study_from_summary

        parent_id = backfill_study_from_summary(summary_path, report_paths=[report])
        print(f"MLflow review parent: {parent_id}")


if __name__ == "__main__":
    main()
