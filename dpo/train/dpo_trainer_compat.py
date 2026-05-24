"""TRL DPO compatibility: assistant-only collator and pretokenized dataset support."""

from __future__ import annotations

import fcntl
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import Dataset, IterableDataset
from datasets.fingerprint import Hasher
from peft import PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, TrainerCallback, TrainingArguments
from transformers.data.data_collator import DataCollatorMixin
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.utils import is_peft_available
from trl import DPOTrainer
from trl.trainer.dpo_trainer import pad

from dpo.train.ref_logprob_cache import (
    cache_path,
    get_last_ref_cache_meta,
    load_ref_logps,
    ref_cache_key,
    save_ref_logps,
    set_last_ref_cache_meta,
)

SFT_ADAPTER_NAME = "default"
DPO_ADAPTER_NAME = "dpo"
REF_ADAPTER_NAME = "ref"
POLICY_ADAPTER_STACK = [SFT_ADAPTER_NAME, DPO_ADAPTER_NAME]


def _adapter_from_param_name(name: str) -> str | None:
    for adapter in (SFT_ADAPTER_NAME, DPO_ADAPTER_NAME, REF_ADAPTER_NAME):
        if f".{adapter}." in name or name.endswith(f".{adapter}"):
            return adapter
    return None


def enforce_adapter_gradients(model: PeftModel) -> None:
    """Activation and trainability are separate: only DPO LoRA may train."""
    for name, param in model.named_parameters():
        adapter = _adapter_from_param_name(name)
        if adapter == DPO_ADAPTER_NAME:
            param.requires_grad = True
        else:
            param.requires_grad = False


def ensure_policy_adapter_stack(model) -> None:
    """Activate frozen SFT + trainable DPO; re-enforce grad flags after PEFT activation."""
    if not is_peft_available() or not isinstance(model, PeftModel):
        return
    if not set(POLICY_ADAPTER_STACK).issubset(model.peft_config.keys()):
        return
    model.base_model.set_adapter(POLICY_ADAPTER_STACK)
    enforce_adapter_gradients(model)


def collect_adapter_diagnostics(model, trainer=None) -> dict:
    """Report active adapters, trainable counts, and non-DPO gradients if any."""
    diag: dict[str, Any] = {"active_adapters": None, "trainable_by_adapter": {}, "trainable_total": 0}
    if not is_peft_available() or not isinstance(model, PeftModel):
        return diag

    try:
        diag["active_adapters"] = list(model.active_adapters)
    except Exception:
        diag["active_adapters"] = str(getattr(model, "active_adapter", None))

    for adapter in (SFT_ADAPTER_NAME, DPO_ADAPTER_NAME, REF_ADAPTER_NAME):
        diag["trainable_by_adapter"][adapter] = 0

    non_dpo_grad_params: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        diag["trainable_total"] += param.numel()
        adapter = _adapter_from_param_name(name)
        if adapter in diag["trainable_by_adapter"]:
            diag["trainable_by_adapter"][adapter] += param.numel()
        if adapter and adapter != DPO_ADAPTER_NAME:
            non_dpo_grad_params.append(name)
        elif adapter is None and param.requires_grad:
            non_dpo_grad_params.append(name)

    if trainer is not None and hasattr(trainer, "optimizer") and trainer.optimizer is not None:
        diag["optimizer_param_groups"] = len(trainer.optimizer.param_groups)
        diag["optimizer_trainable_tensors"] = sum(
            len(g["params"]) for g in trainer.optimizer.param_groups
        )
    else:
        diag["optimizer_param_groups"] = None

    diag["non_dpo_trainable_param_names"] = non_dpo_grad_params[:20]
    diag["non_dpo_trainable_count"] = len(non_dpo_grad_params)
    diag["only_dpo_trainable"] = len(non_dpo_grad_params) == 0
    return diag


@dataclass
class AssistantOnlyDPOCollator(DataCollatorMixin):
    """Build TRL batches with per-token assistant-only completion masks."""

    pad_token_id: int
    pad_to_multiple_of: int | None = None
    return_tensors: str = "pt"

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt_chosen_ids = []
        prompt_rejected_ids = []
        chosen_attention = []
        rejected_attention = []
        chosen_completion_mask = []
        rejected_completion_mask = []

        for ex in examples:
            p = ex["prompt_ids"]
            c = ex["chosen_ids"]
            r = ex["rejected_ids"]
            cs = ex["chosen_score_mask"]
            rs = ex["rejected_score_mask"]
            prompt_chosen_ids.append(p + c)
            prompt_rejected_ids.append(p + r)
            chosen_attention.append([1] * (len(p) + len(c)))
            rejected_attention.append([1] * (len(p) + len(r)))
            chosen_completion_mask.append([0] * len(p) + cs)
            rejected_completion_mask.append([0] * len(p) + rs)

        input_ids = prompt_chosen_ids + prompt_rejected_ids
        attention_mask = chosen_attention + rejected_attention
        completion_mask = chosen_completion_mask + rejected_completion_mask

        input_ids = [torch.tensor(ids) for ids in input_ids]
        attention_mask = [torch.tensor(m, dtype=torch.long) for m in attention_mask]
        completion_mask = [torch.tensor(m, dtype=torch.long) for m in completion_mask]

        output: dict[str, Any] = {}
        output["input_ids"] = pad(
            input_ids,
            padding_value=self.pad_token_id,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["attention_mask"] = pad(
            attention_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        output["completion_mask"] = pad(
            completion_mask,
            padding_value=0,
            padding_side="right",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        if "ref_chosen_logps" in examples[0]:
            output["ref_chosen_logps"] = torch.tensor([ex["ref_chosen_logps"] for ex in examples])
        if "ref_rejected_logps" in examples[0]:
            output["ref_rejected_logps"] = torch.tensor([ex["ref_rejected_logps"] for ex in examples])
        return output


class TrainingHeartbeatCallback(TrainerCallback):
    """Periodic progress lines when stdout is redirected to a log file."""

    def __init__(self, log_fn: Callable[[str], None], interval_s: float = 60.0):
        self.log_fn = log_fn
        self.interval_s = interval_s
        self._last_log = 0.0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict | None = None,
        **kwargs,
    ) -> TrainerControl:
        now = time.time()
        if now - self._last_log < self.interval_s:
            return control
        self._last_log = now
        loss = (logs or {}).get("loss", "?")
        epoch = (logs or {}).get("epoch", state.epoch)
        self.log_fn(
            f"heartbeat step={state.global_step} epoch={epoch} loss={loss}"
        )
        return control


def _load_or_compute_ref_npz(
    trainer: DPOTrainer,
    dataset: Dataset,
    name: str,
    batch_size: int,
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return load_ref_logps(path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=trainer.data_collator,
            num_workers=trainer.args.dataloader_num_workers,
            pin_memory=trainer.args.dataloader_pin_memory,
            shuffle=False,
        )
        data_loader = trainer.accelerator.prepare(dataloader)
        ref_chosen_logps: list[torch.Tensor] = []
        ref_rejected_logps: list[torch.Tensor] = []
        for padded_batch in tqdm(
            iterable=data_loader,
            desc=f"Computing reference log probs for {name} dataset",
        ):
            ref_chosen_logp, ref_rejected_logp = trainer.compute_ref_log_probs(padded_batch)
            ref_chosen_logp, ref_rejected_logp = trainer.accelerator.gather_for_metrics(
                (ref_chosen_logp, ref_rejected_logp)
            )
            ref_chosen_logps.append(ref_chosen_logp.cpu())
            ref_rejected_logps.append(ref_rejected_logp.cpu())

        ref_chosen = torch.cat(ref_chosen_logps).float().numpy()
        ref_rejected = torch.cat(ref_rejected_logps).float().numpy()
        if trainer.accelerator.is_main_process:
            save_ref_logps(path, ref_chosen, ref_rejected)
        trainer.accelerator.wait_for_everyone()
        if path.exists():
            return load_ref_logps(path)
        raise RuntimeError(f"ref cache not written: {path}")


class AssistantOnlyDPOTrainer(DPOTrainer):
    """Skip TRL re-tokenization; policy forward stacks SFT + DPO adapters (no merge)."""

    def _precompute_ref_logps(self, dataset: Dataset, name: str, batch_size: int) -> Dataset:
        key = ref_cache_key(
            dataset_fingerprint=dataset._fingerprint,
            precompute_batch_size=batch_size,
        )
        path = cache_path(key, name)
        fingerprint = Hasher.hash((dataset._fingerprint, key))

        if path.exists():
            ref_chosen_logps, ref_rejected_logps = load_ref_logps(path)
            set_last_ref_cache_meta(
                {"hit": True, "path": str(path.resolve()), "split": name, "rows": len(ref_chosen_logps)}
            )
            print(f"  ref_cache HIT ({name}): {path}", flush=True)
        else:
            print(f"  ref_cache MISS ({name}): computing -> {path}", flush=True)
            ref_chosen_logps, ref_rejected_logps = _load_or_compute_ref_npz(
                self, dataset, name, batch_size, path
            )
            set_last_ref_cache_meta(
                {
                    "hit": False,
                    "path": str(path.resolve()),
                    "split": name,
                    "rows": len(ref_chosen_logps),
                }
            )

        dataset = dataset.add_column(name="ref_chosen_logps", column=ref_chosen_logps)
        dataset = dataset.add_column(
            name="ref_rejected_logps", column=ref_rejected_logps, new_fingerprint=fingerprint
        )
        return dataset

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase,
        args,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        first = next(iter(dataset))
        if "prompt_ids" in first and "chosen_ids" in first:
            return dataset
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)

    def _compute_loss(self, model, inputs, return_outputs=False):
        ensure_policy_adapter_stack(self.accelerator.unwrap_model(model))
        return super()._compute_loss(model, inputs, return_outputs=return_outputs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        ensure_policy_adapter_stack(self.accelerator.unwrap_model(model))
        return super().training_step(model, inputs, num_items_in_batch)
