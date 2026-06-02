# Train-setup semantic review (10 skeletons × 3 trials × 6 weights)

**Setup:** HF BnB `Qwen3-4B-Instruct-2507` + frozen SFT (`default`) + DPO (`dpo`) + `set_adapter(["default","dpo"])` + runtime `dpo_weight` scaling. **Not** merged cat LoRAs.

**Data:** `dpo/eval/train_setup_grid_10skel.jsonl` (180 lines)  
**Decode:** greedy  
**Skeletons (seed 42):** `eval_A4_002`, `eval_A4_003`, `eval_A4_004`, `eval_A6_002`, `eval_B8_001`, `eval_B8_002`, `eval_C4_001`, `eval_C8_005`, `eval_D6_002`, `eval_D8_002`

**Review method:** Three manual semantic passes (Composer 2.5) on `review_trial29.jsonl`, `review_trial10.jsonl`, `review_trial1_v10.jsonl` — every assistant turn read. No regex/CJK automation.

**Labels:** **GOOD** = coherent English, on-topic steering, usable | **BORDERLINE** = weak/repetitive/incomplete but not gibberish | **BAD** = CJK mix, nonsense, broken thread

---

## Executive summary — which trials are worth using?

| Trial | Study | Optuna note | Usable weight band | Primary pick | Avoid |
|-------|--------|-------------|-------------------|--------------|-------|
| **trial29** | v1.1 | Val acc 100%, margin 11.67 | **0.0 – 0.25** | **w=0.25** (sweet spot) | **≥0.5**; never **w=1.0** |
| **trial10** | v1.1 | Conservative residual (r=8) | **0.0 – 0.5** | **w=0.25** (10/10 GOOD) | **w=1.0** (B8 threads break) |
| **trial1_v10** | v1.0 | Low margin 2.08 | **0.0 – 0.25** | **w=0.25** (0 BAD rows) | **≥0.75**; **w=1.0** |

**Recommendation for next step (still train-setup only):**

1. **Lead candidate:** trial-29 @ **dpo_weight 0.25** — best Optuna trial, strongest GOOD rate at low weights; must cap DPO strength.
2. **Backup candidate:** trial-10 @ **0.25** — most stable English at high weights; no CJK collapse at w=1.0 on this sample (trial-29 does collapse).
3. **Secondary backup:** v1.0 trial-1 @ **0.25** — zero BAD rows here; does not clearly beat SFT-only (w=0.0); use for A/B, not as default.

**Do not use:** trial-29 (or any trial) at **w=1.0** for conversational eval — matches earlier full-strength policy failure.

---

## Cross-trial comparison at recommended weight (w=0.25)

| Dimension | trial29 @ 0.25 | trial10 @ 0.25 | trial1 @ 0.25 |
|-----------|----------------|----------------|---------------|
| GOOD / 10 skeletons | **8** | **10** | **4** |
| BAD / 10 | 0 | 0 | 0 |
| Steering vs SFT | Visible DPO flavor; practical products | Cleanest prose; strong product fit | Mixed; some nonsense products (e.g. “microclimate sleeve”) |
| Weak skeletons | B8_001 sand closure; C4_001 never lands product | Rare slips | Early product pitch on A4/A6 |
| vs metric winner | Yes (trial29) | No (trial10) | No (older study) |

**Practical pick:** If you must keep trial-29 for Optuna lineage → **cap at 0.25**. If English stability matters more than val accuracy → **trial-10 @ 0.25** is the safer human-quality choice on this grid.

---

## trial29 — per weight

| dpo_weight | GOOD | BORDERLINE | BAD | Verdict |
|------------|------|------------|-----|---------|
| 0.00 | 5 | 5 | 0 | Usable baseline (SFT-heavy) |
| 0.10 | 7 | 3 | 0 | **Recommended** |
| 0.25 | **8** | 2 | 0 | **Best band** |
| 0.50 | 2 | 8 | 0 | Metaphor soup; unreliable steers |
| 0.75 | 0 | 8 | 2 | Telegraphic + first CJK (A4_003) |
| 1.00 | 0 | 0 | **10** | **Unusable** — systematic EN/ZH mix |

**Failure mode at high weight:** Pseudo-steering (“not X, is Y”), object/friction obsession, then Chinese blocks and incoherent micro-hacks. Confirms full-strength DPO residual is too aggressive despite perfect pairwise val accuracy.

**Persistent skeleton issues (all weights):** `eval_C4_001` often ends without a product; `eval_B8_001` rarely closes sand steer well.

---

## trial10 — per weight

| dpo_weight | GOOD | BORDERLINE | BAD | Verdict |
|------------|------|------------|-----|---------|
| 0.00 | 7 | 3 | 0 | Usable |
| 0.10 | 7 | 3 | 0 | Usable |
| 0.25 | **10** | 0 | 0 | **Best — use this** |
| 0.50 | 8 | 2 | 0 | Acceptable backup |
| 0.75 | 2 | 8 | 0 | Bloated; weak steers |
| 1.00 | 0 | 8 | **2** | B8_001 / B8_002 **BAD** (broken logic, not CJK) |

**Note:** trial-10 @ 1.0 can pass naive “no Chinese character” checks while still failing semantically on long discovery threads.

---

## trial1_v10 (v1.0 trial-1) — per weight

| dpo_weight | GOOD | BORDERLINE | BAD | Verdict |
|------------|------|------------|-----|---------|
| 0.00 | 2 | 8 | 0 | SFT control; product-pushy |
| 0.10 | 4 | 5 | 1 | Mixed |
| 0.25 | 4 | 6 | **0** | **Only weight with zero BAD** |
| 0.50 | 2 | 5 | 3 | B8/A6 failures |
| 0.75 | 3 | 4 | 3 | Gadget hallucination |
| 1.00 | 2 | 5 | 3 | B8_002 rock loop; C8 lid loop |

**vs gentle DPO expectation:** Low weights ≈ SFT with occasional wins (D6 exhaust fan, C8 tea). Does not systematically improve steering; mid/high weights add gadget noise without consistent gain.

---

## What worked vs what did not (train setup)

### Worked

- **Weight scaling** on the training forward path: trial-29 goes from 10/10 BAD at w=1.0 to 8/10 GOOD at w=0.25 on the same skeletons.
- **trial-10 @ 0.25:** Best single configuration in this grid (10/10 GOOD).
- **trial-29 @ 0.1–0.25:** Production-plausible English on problem-first skeletons (A4 headaches, C8 tea, D6 bathroom, D8 kitchen).
- **trial-1 @ 0.25:** Safe fallback with no catastrophic rows.

### Did not work

- **trial-29 @ w≥0.5:** English but unreliable or absurd products.
- **trial-29 @ w≥0.75:** Language and coherence breakdown.
- **Any trial @ w=1.0** for trial-29 — total failure.
- **Long “discovery” skeletons (B8)** under high DPO — metaphor loops, wrong products, repetition (trial-10 w=1.0, trial-1 w=1.0).
- **Expecting val reward accuracy to predict open-loop chat quality** — disproven for trial-29.

---

## Artifacts

| File | Purpose |
|------|---------|
| `train_setup_grid_10skel.jsonl` | All generations (minimal fields) |
| `review_trial29.jsonl` / `review_trial10.jsonl` / `review_trial1_v10.jsonl` | Per-trial splits for review |
| `dpo/train/run_train_setup_grid.py` | Grid runner |
| `RESCUE_SMOKE_RESULTS.md` | Earlier 2-skeleton automated rescue (superseded for semantics by this doc) |

---

## Next step (when train-setup is good enough)

1. Re-run **one** chosen config (e.g. trial-29 @ 0.25) on the 12-skeleton mini slate from `DPO_POLICY_RESCUE_PLAN.md` — still train setup.
2. Only after that passes human read → export **weighted cat** for that trial/weight and verify merged LoRA matches HF stack behavior.
3. Do not run full 52-skeleton judge until both train-setup and merged-LoRA smokes read clean.
