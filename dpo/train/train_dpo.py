"""
DPO training for Saulie steering (V4 dataset).
  python dpo/train/train_dpo.py --dummy [--epochs 2]
  python dpo/train/train_dpo.py --optuna --parallel-workers 2 --target-complete-trials 20
  python dpo/train/train_dpo.py --optuna --optuna-smoke
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import mlflow
import numpy as np
import optuna
import torch
import transformers
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOConfig

import trl

from dpo.train.dpo_data import (
    MAX_LENGTH,
    SPLIT_SEED,
    build_datasets,
    compute_split_diagnostics,
    create_split_manifest,
    load_split_manifest_records,
    load_v4_rows,
    manifest_path_for_seed,
    manifest_sha256,
    run_mask_audit,
)
from dpo.train.dpo_report import write_report
from dpo.train.dpo_trainer_compat import (
    AssistantOnlyDPOCollator,
    AssistantOnlyDPOTrainer,
    collect_adapter_diagnostics,
    enforce_adapter_gradients,
)
from dpo.train.paths import (
    CACHE_DIR,
    DATA_PATH,
    EXPERIMENT_NAME,
    MLRUNS_DIR,
    MODEL_ID_BF16,
    OUTPUT_BASE,
    SFT_ADAPTER,
)
from train.train_sft import clear_gpu, compute_data_hash, patch_chat_template_for_assistant_loss

SEED = 42
N_OPTUNA_TRIALS = 20
BASELINE_REPORT = OUTPUT_BASE / "steering-dpo-v1.0" / "dummy-run" / "dummy_report_baseline_dual_model.json"

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_length_mode(length_mode: str) -> tuple[list[str], float | None, bool]:
    """Map PLAN_FINAL length_mode to TRL loss_type (sigmoid_norm is not WPO)."""
    if length_mode == "none":
        return ["sigmoid"], None, False
    if length_mode == "sigmoid_norm":
        from packaging.version import Version

        if Version(trl.__version__) < Version("1.0.0"):
            raise RuntimeError(
                f"TRL {trl.__version__} lacks sigmoid_norm; upgrade trl>=1.0"
            )
        return ["sigmoid_norm"], None, False
    if length_mode == "ld_0.1":
        return ["sigmoid"], 0.1, False
    if length_mode == "ld_0.2":
        return ["sigmoid"], 0.2, False
    if length_mode == "ld_0.3":
        return ["sigmoid"], 0.3, False
    if length_mode == "ld_0.5":
        return ["sigmoid"], 0.5, False
    if length_mode == "ipo":
        return ["ipo"], None, False
    raise ValueError(f"Unknown length_mode: {length_mode}")


def set_all_seeds(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_ID_BF16))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    patch_chat_template_for_assistant_loss(tokenizer)
    return tokenizer


def _load_bnb_base():
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    return AutoModelForCausalLM.from_pretrained(
        str(MODEL_ID_BF16),
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
    )


SFT_ADAPTER_NAME = "default"
DPO_ADAPTER_NAME = "dpo"


def build_dpo_peft_model(lora_r: int, lora_alpha: int, lora_dropout: float) -> PeftModel:
    """
    Three-layer stack (no merge): BnB base + frozen SFT LoRA + trainable DPO LoRA.
    TRL adds ref adapter (SFT copy). Policy forward uses default+dpo via trainer hook.
    """
    print("Loading stacked PEFT: BnB base + SFT (default) + DPO LoRA...")
    base = _load_bnb_base()
    model = PeftModel.from_pretrained(
        base,
        str(SFT_ADAPTER),
        adapter_name=SFT_ADAPTER_NAME,
        is_trainable=False,
    )
    for name, param in model.named_parameters():
        param.requires_grad = False

    dpo_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
        init_lora_weights=True,
    )
    model.add_adapter(DPO_ADAPTER_NAME, dpo_config)
    for name, param in model.named_parameters():
        if f".{DPO_ADAPTER_NAME}." in name:
            param.requires_grad = True

    print(f"  peft adapters (pre-trainer): {list(model.peft_config.keys())}")
    print(f"  policy stack: [{SFT_ADAPTER_NAME}, {DPO_ADAPTER_NAME}] (set in trainer)")
    print(f"  reference: TRL ref adapter (copy of {SFT_ADAPTER_NAME})")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params (dpo): {trainable:,}")
    return model


def save_dpo_adapter(model: PeftModel, output_dir: Path) -> None:
    """Persist only the DPO adapter; never overwrite SFT trial-17 files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if DPO_ADAPTER_NAME not in model.peft_config:
        raise ValueError(f"Missing {DPO_ADAPTER_NAME} adapter on model")
    model.save_pretrained(str(output_dir), selected_adapters=[DPO_ADAPTER_NAME])


def vram_stats() -> dict:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    return {
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
    }


def extract_reward_metrics(trainer) -> dict[str, float]:
    for entry in reversed(trainer.state.log_history):
        if "eval_rewards/accuracies" in entry:
            keys = (
                "eval_rewards/accuracies",
                "eval_rewards/margins",
                "eval_rewards/chosen",
                "eval_rewards/rejected",
                "eval_loss",
                "eval_logps/chosen",
                "eval_logps/rejected",
            )
            return {k: float(entry[k]) for k in keys if k in entry}
    return {}


def make_dpo_config(
    output_dir: Path,
    *,
    num_train_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    lr_scheduler_type: str,
    warmup_ratio: float,
    max_grad_norm: float,
    weight_decay: float,
    beta: float,
    loss_type: list[str],
    ld_alpha: float | None,
    use_weighting: bool,
    label_smoothing: float,
    neftune_noise_alpha: float,
    live_logging: bool = False,
) -> DPOConfig:
    return DPOConfig(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        weight_decay=weight_decay,
        beta=beta,
        loss_type=loss_type,
        ld_alpha=ld_alpha,
        use_weighting=use_weighting,
        label_smoothing=label_smoothing,
        max_length=MAX_LENGTH,
        truncation_mode="keep_end",
        precompute_ref_log_probs=True,
        precompute_ref_batch_size=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        optim="paged_adamw_32bit",
        neftune_noise_alpha=neftune_noise_alpha,
        logging_steps=1 if live_logging else 10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=1,
        seed=SEED,
        report_to="none",
        remove_unused_columns=False,
        disable_tqdm=not live_logging,
    )


def run_training(
    model,
    datasets,
    tokenizer,
    dpo_args: DPOConfig,
    *,
    live_logging: bool = False,
    heartbeat_log: Callable | None = None,
    trial_number: int | None = None,
    worker_id: int | None = None,
) -> tuple[AssistantOnlyDPOTrainer, object, dict]:
    from dpo.train.dpo_trainer_compat import (
        StepWatchdogCallback,
        TrainingHeartbeatCallback,
    )
    from dpo.train.ref_logprob_cache import get_last_ref_cache_meta

    collator = AssistantOnlyDPOCollator(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )
    callbacks = []
    if live_logging and heartbeat_log is not None:
        callbacks.append(TrainingHeartbeatCallback(heartbeat_log))
    if trial_number is not None:
        callbacks.append(
            StepWatchdogCallback(
                trial_number=trial_number,
                worker_id=worker_id,
                log_fn=heartbeat_log,
            )
        )

    trainer = AssistantOnlyDPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
    )
    from dpo.train.dpo_trainer_compat import require_reference_adapter

    require_reference_adapter(trainer.accelerator.unwrap_model(trainer.model))
    print(f"  peft adapters (post-trainer): {list(trainer.model.peft_config.keys())}")
    enforce_adapter_gradients(trainer.model)
    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"  trainable params after TRL init (dpo only): {trainable:,}")
    adapter_diag = collect_adapter_diagnostics(trainer.model, trainer)
    print(f"  adapter diagnostics: only_dpo_trainable={adapter_diag['only_dpo_trainable']}")
    if hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()
    train_result = trainer.train()
    metrics = extract_reward_metrics(trainer)
    if not metrics:
        trainer.evaluate()
        metrics = extract_reward_metrics(trainer)
    adapter_diag = collect_adapter_diagnostics(trainer.model, trainer)
    ref_cache = get_last_ref_cache_meta()
    return trainer, train_result, metrics, adapter_diag, ref_cache


def load_baseline_metrics() -> dict:
    if BASELINE_REPORT.exists():
        with BASELINE_REPORT.open(encoding="utf-8") as f:
            return json.load(f)
    legacy = OUTPUT_BASE / "steering-dpo-v1.0" / "dummy-run" / "dummy_report.json"
    if legacy.exists():
        with legacy.open(encoding="utf-8") as f:
            data = json.load(f)
        if data.get("vram", {}).get("peak_allocated_gb", 0) > 12:
            return data
    return {}


def compare_to_baseline(new_metrics: dict, baseline: dict) -> dict:
    keys = [
        ("eval_rewards/accuracies", "eval_rewards_accuracies"),
        ("eval_rewards/margins", "eval_rewards_margins"),
        ("eval_loss", "eval_loss"),
        ("train_loss", "train_loss"),
    ]
    comparison = {}
    for metric_key, report_key in keys:
        new_v = new_metrics.get(metric_key) or new_metrics.get(report_key)
        old_v = baseline.get(metric_key) or baseline.get(report_key)
        if new_v is None:
            continue
        entry = {"new": new_v}
        if old_v is not None:
            entry["baseline"] = old_v
            entry["delta"] = new_v - old_v
        comparison[report_key] = entry
    if baseline.get("vram"):
        comparison["vram_peak_allocated_gb"] = {
            "baseline": baseline["vram"].get("peak_allocated_gb"),
            "new": new_metrics.get("vram", {}).get("peak_allocated_gb"),
        }
    return comparison


def run_dummy(args):
    set_all_seeds(SEED)
    output_dir = OUTPUT_BASE / "steering-dpo-v1.0" / "dummy-run"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat()

    baseline = load_baseline_metrics()

    print("\n" + "=" * 60)
    print(" DPO CORRECTED DUMMY RUN (stacked SFT + DPO, no merge)")
    print("=" * 60)
    print(f" TRL version: {trl.__version__}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()

    rows = load_v4_rows()
    print("Regenerating split manifest (fixed 2-way fallback)...")
    create_split_manifest(rows, seed=SPLIT_SEED)
    manifest_path = manifest_path_for_seed(SPLIT_SEED)
    manifest_records = load_split_manifest_records(SPLIT_SEED)
    split_map = {r["id"]: r["split"] for r in manifest_records}

    tokenizer = load_tokenizer()
    mask_audit_path = output_dir / "mask_audit.json"
    mask_audit = run_mask_audit(tokenizer, rows, mask_audit_path)

    datasets, length_stats = build_datasets(tokenizer, split_map, enforce_max_length=True)
    split_diagnostics = compute_split_diagnostics(manifest_records)

    model = build_dpo_peft_model(lora_r=16, lora_alpha=32, lora_dropout=0.1)

    dpo_args = make_dpo_config(
        output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_grad_norm=0.5,
        weight_decay=0.05,
        beta=0.1,
        loss_type=["sigmoid"],
        ld_alpha=None,
        use_weighting=False,
        label_smoothing=0.0,
        neftune_noise_alpha=0.0,
    )

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    report: dict = {
        "mode": "dummy_corrected",
        "stack": "single_model_stacked_default_dpo",
        "policy_adapters": [SFT_ADAPTER_NAME, DPO_ADAPTER_NAME],
        "reference_adapter": "ref",
        "seed": SEED,
        "split_seed": SPLIT_SEED,
        "max_length": MAX_LENGTH,
        "epochs": args.epochs,
        "train_size": len(datasets["train"]),
        "val_size": len(datasets["val"]),
        "data_hash": compute_data_hash(DATA_PATH),
        "trl_version": trl.__version__,
        "split_manifest_path": str(manifest_path),
        "split_manifest_sha256": manifest_sha256(manifest_path),
        "mask_audit_path": str(mask_audit_path),
        "mask_audit": mask_audit,
        "length_stats": length_stats,
        "split_diagnostics": split_diagnostics,
        "hparams": {
            "beta": 0.1,
            "lr": 1e-5,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "batch_size": 1,
            "grad_accum": 4,
            "lr_scheduler": "cosine",
            "warmup_ratio": 0.1,
            "max_grad_norm": 0.5,
            "weight_decay": 0.05,
            "loss_type": "sigmoid",
            "ld_alpha": None,
            "use_weighting": False,
            "neftune_noise_alpha": 0.0,
        },
        "started_at": started,
    }

    trainer = None
    with mlflow.start_run(run_name=f"dummy-corrected-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({
            "stack": "single_model_stacked_default_dpo",
            "beta": 0.1, "lr": 1e-5, "lora_r": 16, "epochs": args.epochs,
            "max_length": MAX_LENGTH, "seed": SEED, "trl": trl.__version__,
        })
        try:
            trainer, train_result, metrics, adapter_diag, ref_cache = run_training(
                model, datasets, tokenizer, dpo_args
            )
            report["ref_cache"] = ref_cache
            report["train_loss"] = float(train_result.training_loss)
            report.update({k.replace("/", "_"): v for k, v in metrics.items()})
            adapter_path = output_dir / "best_adapter"
            save_dpo_adapter(trainer.model, adapter_path)
            report["saved_adapter_path"] = str(adapter_path)
            report["adapter_diagnostics"] = adapter_diag
            report["vram"] = vram_stats()
            report["comparison_vs_baseline_dual_model"] = compare_to_baseline(
                {**metrics, "train_loss": report["train_loss"], "vram": report["vram"]},
                baseline,
            )
            print("\n--- Dummy eval metrics ---")
            for k, v in sorted(metrics.items()):
                print(f"  {k}: {v}")
            print(f"\n  VRAM peak allocated: {report['vram']['peak_allocated_gb']:.2f} GB")
            print(f"  only_dpo_trainable: {adapter_diag['only_dpo_trainable']}")
        finally:
            if trainer is not None:
                del trainer
            del model
            clear_gpu()

    report["finished_at"] = datetime.now().isoformat()
    report_path = output_dir / "dummy_report.json"
    write_report(report, report_path)
    gates = report["gates"]
    print(f"\nDummy report saved: {report_path}")
    print("--- Optuna readiness gates ---")
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        print(f"  {name}: {status}")
    print(f"\noptuna_ready={report['optuna_ready']}")
    print("STOP: review dummy report before Optuna.")


_OPTUNA_DATA = None


def _ensure_optuna_data(dummy_report_path: Path | None = None):
    global _OPTUNA_DATA
    if _OPTUNA_DATA is not None:
        return _OPTUNA_DATA
    from dpo.train.dpo_diagnostics import set_run_provenance

    split_map = {r["id"]: r["split"] for r in load_split_manifest_records(SPLIT_SEED)}
    tokenizer = load_tokenizer()
    datasets, length_stats = build_datasets(tokenizer, split_map)
    mask_pass = True
    dr = dummy_report_path or (
        OUTPUT_BASE / "steering-dpo-v1.0" / "dummy-run" / "dummy_report.json"
    )
    if dr.exists():
        report = json.loads(dr.read_text(encoding="utf-8"))
        mask_pass = report.get("mask_audit", {}).get("pass", True)
    set_run_provenance(
        data_hash=compute_data_hash(DATA_PATH),
        length_stats=length_stats,
        mask_audit_pass=mask_pass,
        trl_version=trl.__version__,
        dummy_report_path=dr if dr.exists() else None,
    )
    _OPTUNA_DATA = (datasets, tokenizer)
    return _OPTUNA_DATA


def derive_trial_params(params: dict) -> dict:
    from dpo.train import optuna_parallel as op

    lora_r = params["lora_r"]
    if "batch_combo" in params and "per_device_train_batch_size" not in params:
        params = op.expand_params_for_training(params)
    batch_size = params["per_device_train_batch_size"]
    grad_accum = params["gradient_accumulation_steps"]
    loss_type, ld_alpha, use_weighting = parse_length_mode(params["length_mode"])
    effective_batch = batch_size * grad_accum
    return {
        "lora_alpha": 2 * lora_r,
        "effective_batch": effective_batch,
        "loss_type": loss_type,
        "ld_alpha": ld_alpha,
        "use_weighting": use_weighting,
    }


def run_optuna_trial(
    cfg,
    trial: optuna.Trial,
    *,
    solo_record: dict | None = None,
) -> float:
    from dpo.train import optuna_parallel as op
    from dpo.train.dpo_diagnostics import (
        apply_trial_diagnostics,
        build_trial_scorecard,
        compute_hybrid_score_v1_1,
        compute_val_diagnostics,
        get_provenance,
        log_line,
        log_trial_scorecard,
        worker_prefix,
    )
    from dpo.train.mlflow_study import log_trial_from_optuna, parent_run_id, setup_mlflow
    from dpo.train.ref_logprob_cache import reset_ref_cache_meta

    if parent_run_id():
        os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg.experiment_name)
        setup_mlflow()

    is_solo = solo_record is not None
    is_parallel_worker = cfg.worker_id is not None and not is_solo
    prefix = worker_prefix(cfg.worker_id, solo=is_solo)
    t0 = time.time()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    vram_start = op.cuda_mem_snapshot()
    stage = "init"

    if is_solo:
        params = op.expand_params_for_training(dict(solo_record["params"]))
        trial.set_user_attr("solo_retry", True)
        trial.set_user_attr("requires_solo", True)
        orig = solo_record.get("original_trial_numbers") or [
            solo_record.get("original_trial_number")
        ]
        trial.set_user_attr(
            "original_parallel_oom_trial_numbers",
            [n for n in orig if n is not None],
        )
    else:
        params = op.sample_trial_params(trial, cfg)
        op.check_duplicate_params(trial)
        op.set_sampler_trial_metadata(trial, cfg)
        if op.params_key(dict(trial.params)) == op.params_key(op.ANCHOR_TRIAL_PARAMS_V11):
            trial.set_user_attr("anchor_trial", True)

    derived = derive_trial_params(params)
    datasets, tokenizer = _ensure_optuna_data(cfg.dummy_report_path)
    mlflow_logged = False
    scorecard = None

    def _log_mlflow(state: str) -> None:
        nonlocal mlflow_logged
        if mlflow_logged or not parent_run_id():
            return
        log_trial_from_optuna(
            trial,
            state=state,
            params=params,
            derived=derived,
            scorecard=scorecard,
            run_dir=cfg.run_dir,
            failure_reason=trial.user_attrs.get("failure_reason"),
        )
        mlflow_logged = True

    model = None
    trainer = None
    collator = None
    ref_cache: dict = {}
    trial_dir = cfg.run_dir / f"trial-{trial.number}"
    objective_result: float | None = None
    train_result = None
    metrics: dict = {}
    adapter_diag: dict = {}
    val_diag: dict | None = None

    def _set_stage(new_stage: str) -> None:
        nonlocal stage
        stage = new_stage
        trial.set_user_attr("last_stage", stage)
        log_line(prefix, f"stage={stage}")

    try:
        if cfg.study_version != "v1.1" and (
            derived["effective_batch"] < 4 or derived["effective_batch"] > 16
        ):
            trial.set_user_attr(
                "failure_reason",
                f"effective_batch={derived['effective_batch']} outside [4,16]",
            )
            raise optuna.TrialPruned(
                f"effective_batch={derived['effective_batch']} outside [4,16]"
            )

        if cfg.worker_id is not None:
            trial.set_user_attr("worker_id", cfg.worker_id)

        trial_dir.mkdir(parents=True, exist_ok=True)
        log_line(prefix, f"TRIAL {trial.number} START")
        log_line(prefix, f"  beta={params['beta']} epochs={params['num_train_epochs']} lr={params['learning_rate']:.2e}")
        log_line(prefix, f"  lora_r={params['lora_r']} batch={params['per_device_train_batch_size']} accum={params['gradient_accumulation_steps']} eff={derived['effective_batch']}")
        log_line(prefix, f"  length_mode={params['length_mode']} neftune={params['neftune_noise_alpha']} sched={params['lr_scheduler_type']}")
        log_line(prefix, f"  trial_dir={trial_dir}")

        from dpo.train.dpo_trainer_compat import TrialWallTimeout
        from dpo.train.gpu_train_lock import gpu_train_lock

        lock_ctx = (
            gpu_train_lock(cfg.run_dir, cfg.worker_id)
            if cfg.worker_id is not None
            else nullcontext()
        )
        with lock_ctx:
            try:
                _set_stage("vram_wait")
                wait_reason = op.wait_for_vram(params, cfg.worker_id)
                if wait_reason:
                    trial.set_user_attr("failure_reason", wait_reason)
                    raise optuna.TrialPruned(wait_reason)

                reset_ref_cache_meta()
                _set_stage("model_build")
                model = build_dpo_peft_model(
                    params["lora_r"], derived["lora_alpha"], params["lora_dropout"]
                )
                dpo_args = make_dpo_config(
                    trial_dir,
                    num_train_epochs=params["num_train_epochs"],
                    per_device_train_batch_size=params["per_device_train_batch_size"],
                    gradient_accumulation_steps=params["gradient_accumulation_steps"],
                    learning_rate=params["learning_rate"],
                    lr_scheduler_type=params["lr_scheduler_type"],
                    warmup_ratio=0.1,
                    max_grad_norm=params["max_grad_norm"],
                    weight_decay=0.05,
                    beta=params["beta"],
                    loss_type=derived["loss_type"],
                    ld_alpha=derived["ld_alpha"],
                    use_weighting=derived["use_weighting"],
                    label_smoothing=0.0,
                    neftune_noise_alpha=params["neftune_noise_alpha"],
                    live_logging=True,
                )
                _set_stage("precompute_ref_and_train")
                log_line(
                    prefix,
                    "ref cache logged during trainer init",
                )
                trainer, train_result, metrics, adapter_diag, ref_cache = run_training(
                    model,
                    datasets,
                    tokenizer,
                    dpo_args,
                    live_logging=True,
                    heartbeat_log=lambda msg: log_line(prefix, msg),
                    trial_number=trial.number,
                    worker_id=cfg.worker_id,
                )
                for split, meta in ref_cache.get("splits", {}).items():
                    hit = "HIT" if meta.get("hit") else "MISS"
                    log_line(prefix, f"ref_cache {split}: {hit} {meta.get('path')}")
                collator = trainer.data_collator
                _set_stage("save_adapter")
                save_dpo_adapter(trainer.model, trial_dir / "best_adapter")

                vram = vram_stats()
                runtime = time.time() - t0
                adapter_path = str(trial_dir / "best_adapter")

                def _val_diag_heartbeat(i: int, n: int) -> None:
                    trial.set_user_attr("val_diag_progress", f"{i + 1}/{n}")

                _set_stage("val_diagnostics")
                val_diag = compute_val_diagnostics(
                    trainer,
                    trainer.eval_dataset or datasets["val"],
                    collator,
                    log_fn=lambda msg: log_line(prefix, msg),
                    trial_number=trial.number,
                    heartbeat_fn=_val_diag_heartbeat,
                )
                _set_stage("diagnostics_write")
                scorecard = build_trial_scorecard(
                    metrics,
                    train_loss=float(train_result.training_loss),
                    vram=vram,
                    runtime_seconds=runtime,
                    saved_adapter_path=adapter_path,
                    ref_cache=ref_cache,
                )
                diag_payload = apply_trial_diagnostics(
                    trial,
                    scorecard=scorecard,
                    val_diag=val_diag,
                    provenance=get_provenance(),
                    params=params,
                    derived=derived,
                    adapter_diag=adapter_diag,
                )
                with (trial_dir / "diagnostics.json").open("w", encoding="utf-8") as f:
                    json.dump(diag_payload, f, indent=2)

                trial.set_user_attr("vram", vram)
                trial.set_user_attr("adapter_diagnostics", adapter_diag)
                trial.set_user_attr("derived", derived)

                if is_solo:
                    trial.set_user_attr("parallel_oom_recovered", True)

                acc = metrics.get("eval_rewards/accuracies")
                if acc is None:
                    raise RuntimeError("Missing eval_rewards/accuracies")
                acc_f = float(acc)
                trial.set_user_attr("eval_rewards_accuracy", acc_f)
                objective = acc_f
                if cfg.study_version == "v1.1":
                    hybrid = compute_hybrid_score_v1_1(
                        accuracy=acc_f,
                        macro_family_category=val_diag.get(
                            "macro_accuracy_by_source_family_category"
                        )
                        if val_diag
                        else None,
                        margin=scorecard.get("eval_rewards_margin"),
                        eval_loss=scorecard.get("eval_loss"),
                        len_corr=val_diag.get("margin_vs_length_delta_corr")
                        if val_diag
                        else None,
                        abs_len_corr=val_diag.get("margin_vs_abs_length_delta_corr")
                        if val_diag
                        else None,
                    )
                    trial.set_user_attr("hybrid_score_v1_1", hybrid)
                    objective = hybrid
                log_trial_scorecard(prefix, trial.number, scorecard, val_diag)
                if cfg.study_version == "v1.1":
                    log_line(prefix, f"  hybrid_score_v1_1={objective:.4f}")
                objective_result = objective
                _set_stage("complete")
            except TrialWallTimeout as e:
                msg = str(e)
                if stage == "val_diagnostics" or "val_diagnostics_timeout" in msg:
                    trial.set_user_attr("failure_reason", "val_diagnostics_timeout")
                elif stage == "precompute_ref_and_train" or "trial_wall" in msg or "step_wall" in msg:
                    trial.set_user_attr("failure_reason", "train_wall_timeout")
                else:
                    trial.set_user_attr("failure_reason", msg[:200])
                raise optuna.TrialPruned(f"wall_timeout: {e}") from e
            finally:
                log_line(prefix, "stage=gpu_cleanup")
                if trainer is not None:
                    del trainer
                    trainer = None
                if model is not None:
                    del model
                    model = None
                clear_gpu()

        if objective_result is not None:
            _log_mlflow("COMPLETE")
            return objective_result
        raise RuntimeError("trial finished GPU section without objective")

    except torch.cuda.OutOfMemoryError as e:
        vram_oom = op.cuda_mem_snapshot()
        log_line(prefix, f"OOM at {stage}: {e}")
        if is_parallel_worker:
            qpath = op.worker_queue_path(cfg.run_dir, cfg.worker_id)
            rec = op.make_oom_record(
                trial=trial,
                params=params,
                derived=derived,
                worker_id=cfg.worker_id,
                stage=stage,
                vram_start=vram_start,
                vram_oom=vram_oom,
            )
            op.append_oom_record(qpath, rec)
            trial.set_user_attr("failure_reason", "parallel_oom_queued_for_solo")
            trial.set_user_attr("queued_for_solo_retry", True)
            trial.set_user_attr("oom_stage", stage)
            raise optuna.TrialPruned("parallel_oom_queued_for_solo") from e
        trial.set_user_attr("failure_reason", "solo_oom_intrinsic_or_too_large")
        trial.set_user_attr("oom_stage", stage)
        raise optuna.TrialPruned("solo_oom_intrinsic_or_too_large") from e
    except optuna.TrialPruned:
        _log_mlflow("PRUNED")
        raise
    except Exception:
        if not trial.user_attrs.get("failure_reason"):
            trial.set_user_attr("failure_reason", "trial_exception")
        _log_mlflow("FAIL")
        raise


def run_optuna(args):
    from dpo.train import optuna_parallel as op

    set_all_seeds(SEED)
    if args.optuna_smoke:
        args.parallel_workers = 2
        args.target_complete_trials = 2
        args.max_attempted_trials = 12

    print("\n" + "=" * 60)
    if args.optuna_worker:
        print(f" DPO OPTUNA WORKER {args.worker_id}")
    elif args.optuna_smoke:
        print(" DPO OPTUNA SMOKE TEST (2 workers, 2 complete trials)")
    elif args.parallel_workers > 0:
        print(
            f" DPO PARALLEL OPTUNA ({args.parallel_workers} workers, "
            f"target={args.target_complete_trials} complete)"
        )
    else:
        print(f" DPO OPTUNA STUDY ({args.n_trials} trials, single process)")
    print("=" * 60)

    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")

    if args.optuna_worker:
        if args.worker_id is None:
            raise SystemExit("--optuna-worker requires --worker-id")
        cfg = op.config_from_args(args)
        mlflow.set_experiment(cfg.experiment_name)
        op.run_worker_loop(cfg, run_optuna_trial)
        return

    if args.parallel_workers > 0:
        from dpo.train.mlflow_study import log_parent_study_summary

        cfg = op.config_from_args(args)
        mlflow.set_experiment(cfg.experiment_name)
        _ensure_optuna_data(cfg.dummy_report_path)
        tracking_uri = f"file://{MLRUNS_DIR.resolve()}"
        smoke_tag = "smoke" if args.optuna_smoke else "main"
        with mlflow.start_run(
            run_name=f"optuna-parallel-{cfg.study_version}-{smoke_tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        ):
            parent_id = mlflow.active_run().info.run_id
            os.environ["MLFLOW_PARENT_RUN_ID"] = parent_id
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
            mlflow.log_params({
                "parallel_workers": args.parallel_workers,
                "target_complete_trials": args.target_complete_trials,
                "max_attempted_trials": args.max_attempted_trials,
                "study_name": cfg.study_name,
                "study_version": cfg.study_version,
                "optuna_smoke": args.optuna_smoke,
                "optuna_base_seed": cfg.optuna_base_seed,
                "run_dir": str(cfg.run_dir.resolve()),
            })
            mlflow.set_tag("run_dir", str(cfg.run_dir.resolve()))
            mlflow.set_tag("study_version", cfg.study_version)
            summary_path = op.run_parallel_launcher(
                cfg,
                run_optuna_trial,
                mlflow_parent_run_id=parent_id,
                mlflow_tracking_uri=tracking_uri,
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report_path = Path(summary.get("study_report_html", cfg.run_dir / "study_report.html"))
            log_parent_study_summary(summary, summary_path, report_path)
        report_path = Path(
            json.loads(summary_path.read_text(encoding="utf-8")).get(
                "study_report_html", cfg.run_dir / "study_report.html"
            )
        )
        if args.optuna_smoke:
            print(f"\nSmoke test done. Summary: {summary_path}")
        else:
            print(f"\nOptuna study finished. Summary: {summary_path}")
        if report_path.is_file():
            print(f"Study report: file://{report_path.resolve()}")
        print(f"MLflow UI: mlflow ui --backend-store-uri {tracking_uri} --port 5001")
        return

    # Legacy single-process Optuna
    datasets, tokenizer = _ensure_optuna_data()

    def objective(trial: optuna.Trial) -> float:
        cfg = op.OptunaRunConfig(
            run_dir=OUTPUT_BASE / EXPERIMENT_NAME,
            study_storage=OUTPUT_BASE / EXPERIMENT_NAME / "optuna_study_legacy.db",
            study_name=EXPERIMENT_NAME,
            target_complete_trials=args.n_trials,
            max_attempted_trials=args.n_trials,
            parallel_workers=0,
        )
        return run_optuna_trial(cfg, trial)

    study = optuna.create_study(
        direction="maximize",
        study_name=EXPERIMENT_NAME,
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.NopPruner(),
    )
    with mlflow.start_run(run_name=f"optuna-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        study.optimize(objective, n_trials=args.n_trials)
        best = study.best_trial
        summary_path = OUTPUT_BASE / EXPERIMENT_NAME / "trial_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "best_trial": best.number,
                    "best_accuracy": best.value,
                    "best_params": best.params,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nBest trial #{best.number}: accuracy={best.value:.4f}")


def main():
    from dpo.train.optuna_parallel import add_parallel_cli_args

    parser = argparse.ArgumentParser(description="DPO training (dummy or Optuna)")
    parser.add_argument("--dummy", action="store_true", help="Run dummy training")
    parser.add_argument("--optuna", action="store_true", help="Run Optuna study")
    parser.add_argument("--epochs", type=int, default=2, help="Dummy epochs")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=N_OPTUNA_TRIALS,
        help="Legacy single-process trial count",
    )
    add_parallel_cli_args(parser)
    args = parser.parse_args()

    if args.dummy and args.optuna:
        parser.error("Use only one of --dummy or --optuna")
    if args.dummy:
        run_dummy(args)
    elif args.optuna or args.optuna_worker:
        run_optuna(args)
    else:
        parser.error("Specify --dummy or --optuna")


if __name__ == "__main__":
    main()
