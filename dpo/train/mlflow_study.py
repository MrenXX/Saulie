"""MLflow logging for DPO Optuna studies (nested child runs per trial)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow

from dpo.train.paths import EXPERIMENT_NAME, MLRUNS_DIR

METRIC_KEYS = (
    "eval_rewards_accuracy",
    "eval_rewards_margin",
    "eval_loss",
    "train_loss",
    "eval_rewards_chosen",
    "eval_rewards_rejected",
    "eval_logps_chosen",
    "eval_logps_rejected",
    "macro_accuracy_by_source_family",
    "macro_accuracy_by_category",
    "macro_accuracy_by_source_family_category",
    "margin_vs_length_delta_corr",
    "margin_vs_abs_length_delta_corr",
    "peak_vram_allocated_gb",
    "peak_vram_reserved_gb",
    "runtime_seconds",
)


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"file://{MLRUNS_DIR.resolve()}"


def parent_run_id() -> str | None:
    return os.environ.get("MLFLOW_PARENT_RUN_ID")


def setup_mlflow() -> None:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)


def _mlflow_param_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return str(v)


def _ref_cache_hits(ua: dict) -> tuple[int | None, int | None]:
    rc = ua.get("ref_cache") or {}
    splits = rc.get("splits") or {}
    train_hit = eval_hit = None
    if "train" in splits:
        train_hit = 1 if splits["train"].get("hit") else 0
    if "eval" in splits:
        eval_hit = 1 if splits["eval"].get("hit") else 0
    return train_hit, eval_hit


def log_trial_run(
    parent_id: str,
    trial_row: dict,
    *,
    run_dir: Path | None = None,
) -> str | None:
    """Log one nested MLflow run for a trial. Returns child run id."""
    n = trial_row.get("trial_number", trial_row.get("number"))
    state = trial_row.get("state", "UNKNOWN")
    p = trial_row.get("params") or {}
    d = trial_row.get("derived") or {}
    ua = trial_row.get("user_attrs") or {}
    run_name = f"trial-{n}"

    params: dict[str, str] = {"optuna_state": state}
    for k, v in p.items():
        params[k] = _mlflow_param_value(v)
    for k, v in d.items():
        params[f"derived_{k}"] = _mlflow_param_value(v)

    if trial_row.get("failure_reason"):
        params["failure_reason"] = _mlflow_param_value(trial_row["failure_reason"])
    if trial_row.get("solo_retry"):
        params["solo_retry"] = "true"
    if trial_row.get("parallel_oom_recovered"):
        params["parallel_oom_recovered"] = "true"

    adapter = ua.get("saved_adapter_path")
    if adapter:
        params["adapter_path"] = str(adapter)

    metrics: dict[str, float] = {}
    if trial_row.get("value") is not None and state == "COMPLETE":
        metrics["eval_rewards_accuracy"] = float(trial_row["value"])
    for key in METRIC_KEYS:
        v = ua.get(key)
        if v is not None:
            try:
                metrics[key] = float(v)
            except (TypeError, ValueError):
                pass

    train_hit, eval_hit = _ref_cache_hits(ua)
    if train_hit is not None:
        metrics["ref_cache_train_hit"] = float(train_hit)
    if eval_hit is not None:
        metrics["ref_cache_eval_hit"] = float(eval_hit)

    with mlflow.start_run(
        run_name=run_name,
        nested=True,
        parent_run_id=parent_id,
    ):
        mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.set_tag("optuna_state", state)
        if run_dir:
            diag = run_dir / f"trial-{n}" / "diagnostics.json"
            if diag.is_file():
                mlflow.log_artifact(str(diag.resolve()))
        return mlflow.active_run().info.run_id


def log_trial_from_optuna(
    trial,
    *,
    state: str,
    params: dict | None = None,
    derived: dict | None = None,
    scorecard: dict | None = None,
    run_dir: Path | None = None,
    failure_reason: str | None = None,
) -> None:
    """Log from live Optuna trial (worker process)."""
    pid = parent_run_id()
    if not pid:
        return
    setup_mlflow()
    ua = dict(trial.user_attrs)
    if scorecard:
        for k, v in scorecard.items():
            if k == "ref_cache":
                ua["ref_cache"] = v
            elif v is not None and not isinstance(v, (dict, list)):
                ua[k] = v
    row = {
        "trial_number": trial.number,
        "state": state,
        "value": trial.value,
        "params": params or dict(trial.params),
        "derived": derived or ua.get("derived") or {},
        "user_attrs": ua,
        "failure_reason": failure_reason or ua.get("failure_reason"),
        "solo_retry": ua.get("solo_retry", False),
        "parallel_oom_recovered": ua.get("parallel_oom_recovered", False),
    }
    log_trial_run(pid, row, run_dir=run_dir)


def log_parent_study_summary(summary: dict, summary_path: Path, report_path: Path | None) -> None:
    """Log best-trial metrics/params and artifacts on the active parent run."""
    if mlflow.active_run() is None:
        return
    counts = summary.get("counts") or {}
    metrics: dict[str, float] = {
        "complete_count": float(counts.get("COMPLETE", 0)),
        "pruned_count": float(counts.get("PRUNED", 0)),
        "fail_count": float(counts.get("FAIL", 0)),
    }
    if summary.get("best_accuracy") is not None:
        metrics["best_accuracy"] = float(summary["best_accuracy"])
    if summary.get("best_trial") is not None:
        metrics["best_trial_number"] = float(summary["best_trial"])
    mlflow.log_metrics(metrics)
    best_params = summary.get("best_params") or {}
    if best_params:
        mlflow.log_params({f"best_{k}": _mlflow_param_value(v) for k, v in best_params.items()})
    mlflow.set_tag("run_dir", str(summary.get("run_dir", "")))
    mlflow.log_artifact(str(summary_path.resolve()))
    if report_path and report_path.is_file():
        mlflow.log_artifact(str(report_path.resolve()))


def backfill_study_from_summary(
    summary_path: Path,
    *,
    parent_run_name: str | None = None,
) -> str:
    """Create parent + nested trial runs from trial_summary.json. Returns parent run id."""
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = Path(summary.get("run_dir", summary_path.parent))
    setup_mlflow()

    if parent_run_name:
        client = mlflow.tracking.MlflowClient()
        exp = client.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            raise RuntimeError(f"Experiment not found: {EXPERIMENT_NAME}")
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{parent_run_name}'",
            max_results=1,
        )
        if not runs:
            raise RuntimeError(f"Parent run not found: {parent_run_name}")
        parent_id = runs[0].info.run_id
    else:
        basename = run_dir.name
        name = f"optuna-parallel-review-{basename}"
        with mlflow.start_run(run_name=name):
            parent_id = mlflow.active_run().info.run_id
            mlflow.log_params({
                "study_name": _mlflow_param_value(summary.get("study_name")),
                "run_dir": str(run_dir),
                "backfill": "true",
                "target_complete_trials": _mlflow_param_value(
                    summary.get("target_complete_trials")
                ),
            })
            for t in summary.get("trials") or []:
                log_trial_run(parent_id, t, run_dir=run_dir)
            report = run_dir / "study_report.html"
            log_parent_study_summary(summary, summary_path, report if report.is_file() else None)

    if parent_run_name:
        for t in summary.get("trials") or []:
            log_trial_run(parent_id, t, run_dir=run_dir)

    return parent_id
