# SFT (Supervised Fine-Tuning)

Qwen3-4B steering SFT — runs **before** DPO in the Saulie pipeline.

## Layout

| Path | In git? | Purpose |
|------|---------|---------|
| `train_sft.py` | Yes | Optuna HPO + MLflow SFT training (canonical) |
| `analyze_seq_len.py` | Yes | Sequence-length analysis utility |
| `sft_eval/` | Scripts yes, JSON no | SFT-specific eval harness + `llm_judge_prompt.md` |
| `SFT_DATA_V3.jsonl` | No (gitignored) | Training dataset |
| `models/` | No (gitignored) | LoRA adapters — prod baseline at `models/steering-sft-v1.1/trial-17/best_adapter` |
| `mlruns/` | No (gitignored) | MLflow tracking store |
| `eval_results/` | No (gitignored) | One-off eval JSON outputs |

## vs `dpo/eval/`

- **`sft/sft_eval/`** — SFT trial comparison, SFT judge prompt, legacy multi-adapter deploy for SFT studies.
- **`dpo/eval/`** — DPO behavioral eval, v1.5 merge gate, production deploy scripts (`vllm_scripts/`).

## How DPO uses this

```python
from sft.train_sft import patch_chat_template_for_assistant_loss, compute_data_hash, clear_gpu
```

Frozen SFT adapter path (used by merge + deploy):

```
sft/models/steering-sft-v1.1/trial-17/best_adapter
```

Defined in `dpo/train/paths.py` as `SFT_ADAPTER`.

## Training

```bash
conda activate saulgman
python sft/train_sft.py
```

Requires `Qwen3-4B-Instruct-2507` (BF16) on disk. See [`REPRODUCIBILITY.md`](../REPRODUCIBILITY.md) for full setup.

## SFT eval

```bash
bash sft/sft_eval/deploy_qwenie_eval.sh
python sft/sft_eval/eval_generate_vllm.py
```

Judge rubric: `sft/sft_eval/llm_judge_prompt.md`.
