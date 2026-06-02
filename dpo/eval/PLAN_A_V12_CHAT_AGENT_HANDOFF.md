# Plan A v1.2 + HF chat REPL — agent handoff

**Created:** 2026-05-30  
**Workspace:** `/root/saulie`  
**Conda env:** `saulgman`  
**Next session focus:** Manual **conversation gate** on v1.2 adapters (`chat_policy_stack.py`); compare against **SFT-only** and **bare base** on the same script; decide Plan B vs cat/FP8 export. Do **not** rerun broad Optuna or update `dpo/train/HANDOFF.md` unless the user asks.

**Related handoffs (do not duplicate — read only what you need):**

| Doc | Scope |
|-----|--------|
| `dpo/DPO_HANDOFF.md` | Original DPO project plan (pre-implementation era) |
| `dpo/train/HANDOFF.md` | v1.0/v1.1 Optuna + final vLLM eval runbook |
| `dpo/eval/plan_a_existing_trials_and_final_stack_retrains.md` | Plan A spec (2 rescue configs, gate order) |
| `dpo/eval/PLAN_A_PART1_SEMANTIC_REPORT.md` | Old v1.1 trials @ w=1.0 fluency-only (10 skeletons) |
| `dpo/eval/RUN_PLAN_A_V12.md` | Launch/monitor commands (some chat lines superseded below) |

---

## One-paragraph state

Plan A abandoned old Optuna winners for **deployment conversation quality**, not val accuracy. Part 1 showed v1.1 trials are mostly fluent at `w=1.0` but **trial-13 failed a normal chat** (phrase-loop fixation). Part 2 trained **two fixed v1.2 rescue runs** (`plan_a_minimal_dpo`, `plan_a_ipo`) successfully after fixing v1.2 Optuna enqueue bugs. User REPL: **trial 0 slightly better than trial 1**, but **both still repeat easily and do not hold a normal conversation** — may be partly **SFT**, not only DPO. Interactive HF testing is centralized in **`dpo/eval/chat_policy_stack.py`** with `--base {bnb,bf16}` and stack modes below.

---

## Official v1.2 training run (COMPLETE)

| Field | Value |
|-------|--------|
| **Run dir** | `/root/saulie/dpo/train/models/steering-dpo-v1.2/optuna-run-20260530-064345` |
| **Pointer** | `dpo/train/models/steering-dpo-v1.2/LATEST_RUN_DIR.txt` |
| **Study** | `steering-dpo-v1.2-plan-a-seed42-064345` |
| **MLflow** | experiment `steering-dpo-v1.2` |
| **Counts** | 2 COMPLETE, 0 PRUNED, 0 FAIL |
| **Summary** | `.../optuna-run-20260530-064345/trial_summary.json` |

| Trial | `rescue_label` | Val accuracy | Val margin | Adapter |
|-------|----------------|-------------|------------|---------|
| 0 | `plan_a_minimal_dpo` (`ld_0.5`) | 38.5% | −1.94 | `trial-0/best_adapter` |
| 1 | `plan_a_ipo` | 30.8% | −7.30 | `trial-1/best_adapter` |

Low val accuracy is expected for conservative rescue configs; **do not** pick a winner by accuracy alone. User cares about **English fluency, no CJK, no phrase loops, normal multi-turn chat** @ `dpo_weight=1.0`.

### Failed v1.2 launches (ignore these dirs)

| Run dir | Cause |
|---------|--------|
| `optuna-run-20260530-064151` | All trials PRUNED: `study.optimize` + empty enqueued `trial.params` |
| `optuna-run-20260530-064314` | Worker crash: `batch_combo` not expanded → `KeyError: per_device_train_batch_size` |

**Fixes (in repo):** `dpo/train/optuna_parallel.py` — v1.2 uses `study.ask()`/`tell()` for exactly 2 trials; `resolve_trial_params` expands `batch_combo` and falls back to `PLAN_A_RESCUE_TRIALS[trial.number]`.

---

## Locked user constraints

- **Do not** update `dpo/train/HANDOFF.md` unless asked.
- **No** automated “normal conversation” eval scripts — user uses REPL only.
- **Do not** judge raw instruct base as “Saulie”; compare **SFT trial-17** vs **policy stack**.
- Architecture unchanged: **BnB8 base + frozen SFT `default` + trainable DPO `dpo`**, `set_adapter(["default","dpo"])` for policy.
- Cat merge / FP8 vLLM only **after** conversation + skeleton gates pass (per Plan A doc).

---

## HF chat REPL (`chat_policy_stack.py`)

**Script:** `dpo/eval/chat_policy_stack.py`  
**Default base:** `bnb` (matches DPO training). **`bf16`** loads full-precision weights + adapters (more VRAM).

```bash
conda activate saulgman
cd /root/saulie
```

### v1.2 policy @ w=1.0 (primary)

```bash
RUN=/root/saulie/dpo/train/models/steering-dpo-v1.2/optuna-run-20260530-064345

python dpo/eval/chat_policy_stack.py --trial minimal          # trial 0
python dpo/eval/chat_policy_stack.py --trial ipo              # trial 1
python dpo/eval/chat_policy_stack.py --base bf16 --trial minimal
```

### Baselines (HF decode: Qwen3 sample — temp 0.7, top_p 0.8, top_k 20; not greedy)

```bash
python dpo/eval/chat_policy_stack.py --sft-only               # BnB + trial-17 SFT
python dpo/eval/chat_policy_stack.py --base bf16 --sft-only
python dpo/eval/chat_policy_stack.py --base-only              # BnB instruct, no LoRA
python dpo/eval/chat_policy_stack.py --base bf16 --base-only
```

### Other useful flags

- `--dpo-weight 0.25` — scale DPO residual (restart to change mid-session)
- `--decode sample` — sampling decode
- `--dpo-adapter <path>` — any trial adapter dir
- REPL: `/reset`, `/quit`

**Loaders:** `dpo/train/merge_sft_dpo_lora.py` — `load_sft_stack(base=...)`, `load_stacked_for_merge(..., base=...)`; `dpo/train/model_load.py` — `load_base("bnb"|"bf16")`.

---

## Plan A gate order (from spec)

1. **Gate 1 — normal conversation (5 short chats):** user REPL @ `w=1.0` on trial 0 and 1; compare `--sft-only`.
2. **Gate 2 — 10 skeleton fluency** (if Gate 1 passes): reuse `dpo/eval/run_plan_a_part1_hf_smoke.py` pattern / `plan_a_part1_w1_10skel.jsonl` skeleton set; see `PLAN_A_PART1_SEMANTIC_REPORT.md` for rubric.
3. **Cat / FP8** only if both gates pass.

If **both** v1.2 trials fail Gate 1 → **Plan B** (SFT-baked base, etc.) per `plan_a_existing_trials_and_final_stack_retrains.md`.

---

## Key code / data paths

| Path | Purpose |
|------|---------|
| `dpo/train/optuna_parallel.py` | `PLAN_A_RESCUE_TRIAL_1/2`, `enqueue_plan_a_rescue_trials`, `run_plan_a_v12_worker_loop` |
| `dpo/train/train_dpo.py` | v1.2 objective = val accuracy; logs `rescue_label`, `high_margin_warning` if margin > 5 |
| `dpo/train/scripts/preflight_plan_a_v12.py` | IPO + `ld_0.5` TRL smoke |
| `dpo/train/scripts/run_plan_a_v12.sh` | Launch 1-worker v1.2 study |
| `train/models/steering-sft-v1.1/trial-17/best_adapter` | Frozen SFT (`paths.SFT_ADAPTER`) |
| `dpo/dataset/DPO_522_prompt_a_and_prompt_b_V4_repaired.jsonl` | Training data |
| `dpo/eval/plan_a_part1_w1_10skel.jsonl` | Part 1 HF smoke outputs |
| `dpo/eval/dpo_phase1_smoke_fp8.json` | FP8/vLLM smoke reference |

---

## Prior findings (short pointers)

- **trial-29 @ w=1.0:** CJK/gibberish on HF stack — not vLLM-only; see rescue docs.
- **trial-13 @ w=1.0:** grammatical but **hook/phrase loop** — motivated Plan A Part 2.
- **Part 1:** six v1.1 trials mostly 9–10/10 skeleton PASS; not trial-29-level collapse — `PLAN_A_PART1_SEMANTIC_REPORT.md`.
- **Weighted deploy (~0.25):** optional stopgap from older trials — `dpo/eval/dpo_weighted_rescue_manifest.jsonl`, `merge_sft_dpo_lora.py --dpo-weight`.

---

## Suggested skills (next agent)

| Skill | When |
|-------|------|
| None required | REPL + skeleton reruns follow existing scripts |
| `babysit` | Only if relaunching Optuna / CI |
| `/root/saulie/SKILL.md` (grill-me) | If revisiting Plan B or SFT vs DPO architecture |
| `canvas` | Only if building a structured comparison table for the user |

---

## Explicit non-goals for next session

- Rerun v1.2 training unless user requests (study is complete).
- Broad v1.1/v1.0 Optuna exploration.
- Update `dpo/train/HANDOFF.md` or old `dpo/DPO_HANDOFF.md` without user direction.
- Automated LLM-judge “conversation gate” scripts.
