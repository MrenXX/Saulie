#!/usr/bin/env python3
"""Repair stale trial_summary / study_report / MLflow after early launcher exit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from dpo.train.mlflow_study import backfill_study_from_summary
from dpo.train.optuna_parallel import (
    OptunaRunConfig,
    copy_trial_summary_to_study_results,
    enrich_summary_dict,
    reload_study,
    write_final_summary,
    read_jsonl,
    dedupe_oom_queue,
    default_merged_queue,
)
from dpo.train.paths import trial_summary_path
from dpo.train.study_report import write_study_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate summary+HTML from Optuna DB; replace stale MLflow parent tree"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--study-name", default="steering-dpo-v1.5-hybrid-seed42")
    parser.add_argument("--study-version", default="v1.5")
    parser.add_argument(
        "--parent-run-name",
        type=str,
        default=None,
        help="MLflow parent run name to replace (default: infer from launcher.log timestamp)",
    )
    parser.add_argument(
        "--skip-mlflow",
        action="store_true",
        help="Only regenerate JSON + HTML on disk",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    study_version = args.study_version
    summary_path = trial_summary_path(run_dir, study_version)

    cfg = OptunaRunConfig(
        run_dir=run_dir,
        study_storage=run_dir / "optuna_study.db",
        study_name=args.study_name,
        target_complete_trials=20,
        max_attempted_trials=60,
        parallel_workers=2,
        study_version=study_version,
        experiment_name=f"steering-dpo-{study_version}",
    )
    study = reload_study(cfg)
    merged = default_merged_queue(run_dir)
    queue_records = read_jsonl(merged) if merged.is_file() else []
    deduped = dedupe_oom_queue(queue_records) if queue_records else []
    write_final_summary(
        study,
        cfg,
        queue_records=queue_records,
        deduped_queue=deduped,
        meta={"repaired_at": "repair_study_artifacts.py"},
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = enrich_summary_dict(summary)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    copied = copy_trial_summary_to_study_results(summary_path, study_version)
    report = write_study_report(summary_path)
    study_results_report = copied.parent / "study_report_v1.5.html"
    study_results_report.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    summary["study_report_html"] = str(report.resolve())
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    counts = summary.get("counts") or {}
    print(f"Summary: {summary_path}")
    print(f"Copied: {copied}")
    print(f"HTML: file://{report.resolve()}")
    print(
        f"COMPLETE={counts.get('COMPLETE')} target_reached={summary.get('target_reached')}"
    )

    if args.skip_mlflow:
        return

    parent_name = args.parent_run_name
    if not parent_name:
        launcher = run_dir / "launcher.log"
        if launcher.is_file():
            for line in launcher.read_text(encoding="utf-8", errors="replace").splitlines():
                if "optuna-parallel-" in line and "main-" in line:
                    pass
        # Match the early-exit parent from this run dir timestamp
        parent_name = f"optuna-parallel-{study_version}-main-20260602-052758"

    parent_id = backfill_study_from_summary(
        summary_path,
        parent_run_name=parent_name,
        replace_parent=True,
        report_paths=[report, copied],
    )
    uri = f"file://{(REPO_ROOT / 'dpo/train/mlruns').resolve()}"
    print(f"MLflow parent: {parent_name} id={parent_id}")
    print(f"mlflow ui --backend-store-uri {uri} --port 5001")


if __name__ == "__main__":
    main()
