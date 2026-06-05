#!/usr/bin/env bash
# Merge v1.5 eval slate trials with full validation (Delta-W + drift + gen smoke).
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate saulgman
cd /root/saulie

RUN="dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732"
REPORT="dpo/eval/merge_v15_slate_report.json"
TRIALS=(19 16 8 27 20 4)

python - <<'PY'
import json, subprocess, sys
from pathlib import Path

REPO = Path("/root/saulie")
RUN = REPO / "dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732"
TRIALS = [19, 16, 8, 27, 20, 4]
results = []

for n in TRIALS:
    adapter = RUN / f"trial-{n}" / "best_adapter"
    out = RUN / f"trial-{n}" / "sft_dpo_cat"
    print(f"\n======== trial {n} ========", flush=True)
    cmd = [
        sys.executable,
        str(REPO / "dpo/train/merge_sft_dpo_lora.py"),
        "--dpo-adapter", str(adapter),
        "--output", str(out),
        "--dpo-weight", "1.0",
        "--audit-forward-drift",
        "--generation-smoke",
        "--audit-fp8-deploy",
    ]
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=False)
    row = {"trial": n, "exit_code": proc.returncode}
    meta_path = out / "merge_meta.json"
    if meta_path.is_file():
        row["merge_meta"] = json.loads(meta_path.read_text())
    else:
        row["merge_meta"] = None
    results.append(row)

report_path = REPO / "dpo/eval/merge_v15_slate_report.json"
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps({"trials": results}, indent=2), encoding="utf-8")
print(f"\nWrote {report_path}", flush=True)

# Human-readable validation summary (same metrics as prior slate + FP8 deploy)
val_path = REPO / "dpo/eval/merge_v15_slate_validation.md"
lines = [
    "# v1.5 eval slate merge validation",
    "",
    "Gates: **Delta-W** (blocking) | local stack↔cat | **BnB cat↔FP8 cat** (deploy proxy)",
    "",
    "| Trial | exit | ΔW max | stack↔cat max | top1 | stack smoke | cat smoke | verdict | FP8↔BnB max | FP8 top1 | bnb cat | fp8 cat | FP8 regress |",
    "|------:|-----:|-------:|--------------:|-----:|------------:|----------:|---------|--------------:|---------:|--------:|--------:|-------------:|",
]
for row in results:
    n = row["trial"]
    ec = row["exit_code"]
    m = row.get("merge_meta") or {}
    mc = m.get("merge_correctness") or {}
    ld = m.get("local_forward_drift") or {}
    ls = ld.get("summary") or {}
    bs = m.get("local_behavior_smoke") or {}
    bss = bs.get("summary") or {}
    ed = m.get("export_decision") or {}
    dd = m.get("deploy_forward_drift") or {}
    dds = dd.get("summary") or {}
    dbs = m.get("deploy_behavior_smoke") or {}
    dbss = dbs.get("summary") or {}
    dw = mc.get("max_abs_delta_diff")
    dw_s = f"{dw:.1e}" if isinstance(dw, float) else "—"
    ld_max = ls.get("max_abs_logit_diff")
    ld_s = f"{ld_max:.2f}" if isinstance(ld_max, float) else "—"
    t1 = ls.get("top1_agreement_mean")
    t1_s = f"{t1:.2f}" if isinstance(t1, float) else "—"
    st = bss.get("stack_clean_prompts")
    ct = bss.get("cat_clean_prompts")
    pr = bss.get("prompts", 7)
    st_s = f"{st}/{pr}" if st is not None else "—"
    ct_s = f"{ct}/{pr}" if ct is not None else "—"
    fp8_max = dds.get("max_abs_logit_diff")
    fp8_max_s = f"{fp8_max:.2f}" if isinstance(fp8_max, float) else dd.get("status", "—")
    fp8_t1 = dds.get("top1_agreement_mean")
    fp8_t1_s = f"{fp8_t1:.2f}" if isinstance(fp8_t1, float) else "—"
    bnb_g = dbss.get("bnb8_cat_clean_prompts")
    fp8_g = dbss.get("fp8_cat_clean_prompts")
    bnb_g_s = f"{bnb_g}/{pr}" if bnb_g is not None else "—"
    fp8_g_s = f"{fp8_g}/{pr}" if fp8_g is not None else "—"
    reg = dbss.get("fp8_only_regression_prompts")
    reg_s = str(reg) if reg is not None else "—"
    lines.append(
        f"| {n} | {ec} | {dw_s} | {ld_s} | {t1_s} | {st_s} | {ct_s} | {ed.get('verdict', '—')} | "
        f"{fp8_max_s} | {fp8_t1_s} | {bnb_g_s} | {fp8_g_s} | {reg_s} |"
    )
lines.extend([
    "",
    "Notes:",
    "- FP8 path is HF `MODEL_ID_FP8` + cat LoRA (proxy for vLLM FP8+LoRA).",
    "- Forward drift thresholds are diagnostic only; export gated by ΔW only.",
    "- `FP8 regress` = prompts where FP8 gen tags are worse than BnB cat alone.",
])
val_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {val_path}", flush=True)
PY
