"""MLflow logging for DPO Optuna studies (nested child runs per trial)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import mlflow

from dpo.train.paths import EXPERIMENT_NAME, MLRUNS_DIR, experiment_name_for_version

SKIP_METRIC_ATTRS = frozenset({
    "val_diagnostics_json",
    "adapter_diagnostics",
    "derived",
    "ref_cache",
    "vram",
})

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
    "hybrid_score_v1_1",
)


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI") or f"file://{MLRUNS_DIR.resolve()}"


def parent_run_id() -> str | None:
    return os.environ.get("MLFLOW_PARENT_RUN_ID")


def setup_mlflow(experiment_name: str | None = None) -> None:
    mlflow.set_tracking_uri(tracking_uri())
    name = experiment_name or os.environ.get("MLFLOW_EXPERIMENT_NAME") or EXPERIMENT_NAME
    mlflow.set_experiment(name)


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
    if ua.get("anchor_trial"):
        params["anchor_trial"] = "true"
    if ua.get("enqueued_trial"):
        params["enqueued_trial"] = "true"
    if ua.get("rescue_label"):
        params["rescue_label"] = _mlflow_param_value(ua["rescue_label"])
    if ua.get("plan_b_label"):
        params["plan_b_label"] = _mlflow_param_value(ua["plan_b_label"])
    if ua.get("stack_mode"):
        params["stack_mode"] = _mlflow_param_value(ua["stack_mode"])
    if ua.get("plan_a_trial_index") is not None:
        params["plan_a_trial_index"] = str(ua["plan_a_trial_index"])
    if ua.get("high_margin_warning"):
        params["high_margin_warning"] = "true"
    if ua.get("duplicate_of") is not None:
        params["duplicate_of"] = str(ua["duplicate_of"])

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

    for key, v in ua.items():
        if key in SKIP_METRIC_ATTRS or key in metrics:
            continue
        if isinstance(v, bool):
            continue
        try:
            metrics[key] = float(v)
        except (TypeError, ValueError):
            continue

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


def _trial_objective_value(trial, ua: dict) -> float | None:
    """Objective while trial is still running: use user_attrs, not trial.value."""
    for key in ("hybrid_score_v1_1", "eval_rewards_accuracy"):
        v = ua.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    v = getattr(trial, "value", None)
    if v is not None:
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


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
    """Log from live Optuna trial (worker process). Never raises — logging must not fail trials."""
    pid = parent_run_id()
    if not pid:
        return
    try:
        setup_mlflow(os.environ.get("MLFLOW_EXPERIMENT_NAME"))
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
            "value": _trial_objective_value(trial, ua),
            "params": params or dict(trial.params),
            "derived": derived or ua.get("derived") or {},
            "user_attrs": ua,
            "failure_reason": failure_reason or ua.get("failure_reason"),
            "solo_retry": ua.get("solo_retry", False),
            "parallel_oom_recovered": ua.get("parallel_oom_recovered", False),
        }
        log_trial_run(pid, row, run_dir=run_dir)
    except Exception as e:
        import traceback

        print(f"[mlflow] trial-{trial.number} logging skipped: {e}", flush=True)
        traceback.print_exc()


def log_parent_study_summary(
    summary: dict,
    summary_path: Path,
    report_paths: Path | list[Path] | None = None,
) -> None:
    """Log best-trial metrics/params and artifacts on the active parent run."""
    if mlflow.active_run() is None:
        return
    paths: list[Path] = []
    if report_paths is None:
        pass
    elif isinstance(report_paths, Path):
        paths = [report_paths]
    else:
        paths = list(report_paths)
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
    for report_path in paths:
        if report_path.is_file():
            mlflow.log_artifact(str(report_path.resolve()))


def _nested_child_run_ids(client: mlflow.tracking.MlflowClient, parent_id: str) -> list[str]:
    """Return run ids of direct nested children (filesystem store has no parent filter)."""
    parent = client.get_run(parent_id)
    exp_id = parent.info.experiment_id
    out: list[str] = []
    for run in client.search_runs(experiment_ids=[exp_id], max_results=500):
        if run.data.tags.get("mlflow.parentRunId") == parent_id:
            out.append(run.info.run_id)
    return out


def delete_parent_run_tree(
    client: mlflow.tracking.MlflowClient,
    experiment_id: str,
    parent_run_name: str,
) -> list[str]:
    """Delete a top-level parent and all nested trial-* children. Returns deleted run ids."""
    deleted: list[str] = []
    parents = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"attributes.run_name = '{parent_run_name}'",
        max_results=20,
    )
    parent_ids = {p.info.run_id for p in parents}
    for parent in parents:
        pid = parent.info.run_id
        for child_id in _nested_child_run_ids(client, pid):
            client.delete_run(child_id)
            deleted.append(child_id)
        client.delete_run(pid)
        deleted.append(pid)
    # Children orphaned when a parent was deleted without subtree cleanup (early launcher exit).
    for run in client.search_runs(experiment_ids=[experiment_id], max_results=500):
        parent_tag = run.data.tags.get("mlflow.parentRunId")
        if parent_tag and parent_tag not in parent_ids:
            if run.info.run_name and run.info.run_name.startswith("trial-"):
                client.delete_run(run.info.run_id)
                deleted.append(run.info.run_id)
    return deleted


def backfill_study_from_summary(
    summary_path: Path,
    *,
    parent_run_name: str | None = None,
    parent_name_suffix: str = "",
    report_paths: list[Path] | None = None,
    replace_parent: bool = False,
) -> str:
    """Create parent + nested trial runs from trial_summary.json. Returns parent run id.

    If replace_parent=True and parent_run_name is set, delete any existing parent run(s)
    with that name (and nested children) before creating a fresh parent with full trials.
    """
    summary_path = summary_path.resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = Path(summary.get("run_dir", summary_path.parent))
    exp_name = summary.get("experiment_name") or EXPERIMENT_NAME
    if summary.get("study_version"):
        exp_name = experiment_name_for_version(str(summary["study_version"]))
    setup_mlflow(exp_name)
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {exp_name}")

    if parent_run_name and replace_parent:
        removed = delete_parent_run_tree(client, exp.experiment_id, parent_run_name)
        if removed:
            print(
                f"[mlflow] deleted stale parent tree {parent_run_name!r}: "
                f"{len(removed)} run(s)",
                flush=True,
            )

    if parent_run_name and not replace_parent:
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string=f"attributes.run_name = '{parent_run_name}'",
            max_results=1,
        )
        if not runs:
            raise RuntimeError(f"Parent run not found: {parent_run_name}")
        parent_id = runs[0].info.run_id
        for t in _trials_with_resolved_params(summary):
            log_trial_run(parent_id, t, run_dir=run_dir)
        return parent_id

    if parent_run_name:
        name = parent_run_name
    else:
        basename = run_dir.name
        suffix = f"-{parent_name_suffix}" if parent_name_suffix else ""
        name = f"optuna-parallel-review{suffix}-{basename}"

    reports = report_paths
    if reports is None:
        reports = []
        for candidate in (run_dir / "study_report_v2.html", run_dir / "study_report.html"):
            if candidate.is_file():
                reports.append(candidate)

    with mlflow.start_run(run_name=name):
        parent_id = mlflow.active_run().info.run_id
        mlflow.log_params({
            "study_name": _mlflow_param_value(summary.get("study_name")),
            "run_dir": str(run_dir),
            "backfill": "true",
            "report_suffix": _mlflow_param_value(parent_name_suffix or "default"),
            "target_complete_trials": _mlflow_param_value(
                summary.get("target_complete_trials")
            ),
            "target_reached": _mlflow_param_value(summary.get("target_reached")),
        })
        mlflow.set_tag("run_dir", str(run_dir.resolve()))
        mlflow.set_tag("study_version", str(summary.get("study_version", "")))
        trials = _trials_with_resolved_params(summary)
        for t in trials:
            log_trial_run(parent_id, t, run_dir=run_dir)
        log_parent_study_summary(summary, summary_path, reports)
        print(
            f"[mlflow] backfilled {len(trials)} nested runs under {name!r}",
            flush=True,
        )

    return parent_id


def _trials_with_resolved_params(summary: dict) -> list[dict]:
    """Ensure each trial row has canonical params for MLflow backfill."""
    from dpo.train.optuna_parallel import params_for_summary_row

    study_version = str(summary.get("study_version") or "")
    out: list[dict] = []
    for row in summary.get("trials") or []:
        trial = dict(row)
        p = params_for_summary_row(trial, study_version)
        if p:
            trial["params"] = p
            derived = trial.get("derived") or {}
            if not derived and p:
                from dpo.train.optuna_parallel import derive_summary_fields

                trial["derived"] = derive_summary_fields(p)
        out.append(trial)
    return out
