# Supervised fine-tuning

SFT is the first training phase in the Qwen pipeline. It teaches conversational behavior from completed examples. DPO is a later phase that compares preferred and rejected continuations against a frozen SFT reference.

The target pattern is to engage with an opener, ask questions that expose useful constraints, and connect those details to one generic product category. It does not teach live inventory, current prices, or product URLs. [Runtime retrieval](../rag/README.md) supplies catalog facts separately.

## Inputs and loss

[`train_sft.py`](train_sft.py) reads JSONL with `messages` and `metadata`. Metadata includes `opening_type` and conversation length for stratification. The loader uses seed 42, reserves 10% for test, then splits validation from the remainder. Saved March studies record 400 training, 50 validation, and 50 test conversations.

The trainer sets `assistant_only_loss=True`. `patch_chat_template_for_assistant_loss` adds generation markers to the Qwen chat template so user text remains context and only assistant responses contribute to loss. It performs a small mask check before training. Packing is disabled and the sequence limit is 832 tokens.

The standard path loads the BF16 Qwen checkpoint with bitsandbytes 8-bit quantization. The script also has an FP8 loading option. FP8 is the later serving format, so evaluation references to an FP8 model do not mean all training used FP8.

## Training and tracking

The stack uses Transformers, TRL, PEFT, Optuna, and MLflow. Each trial reloads the base, trains its LoRA, evaluates validation loss, and saves a `best_adapter`. Optuna minimizes validation loss. Generated conversations assess behavioral quality separately.

The current script uses rank 16, alpha 32, dropout 0.1, and attention plus MLP projection targets. It enables gradient checkpointing and NEFTune noise. It searches learning rate, epoch count, gradient accumulation, and clipping.

This is an editable research entry point, not an immutable recipe for each historical study. Present defaults name `steering-sft-v1.2`, request 15 trials, and fix the scheduler to `constant_with_warmup`. Saved v1.1 and v1.2 results include `linear` schedulers. Saved parameters take precedence when describing an old run.

## Saved studies

- March 23, 2026: local `steering-sft-v1` adapters and MLflow records establish an earlier Qwen study. A complete standalone ranking was not recovered.
- March 25: v1.1 has 20 completed trial records. Trial 16 has the lowest recorded validation loss, `2.2194`. Trial 17 records `2.2227`.
- March 28: v1.2 has seven completed records. Trial 13 has the lowest recorded loss, `2.1843`. The parent run requested 15 trials, which must not be reported as 15 completed trials.
- March 31: a saved comparison covers the base and ten SFT candidates on 14 scenarios per model. Its old filename says `top5`, but the contents have 11 models.

Evidence: [v1.1 summary](../docs/evidence/sft-study-v1.1.json), [v1.2 summary](../docs/evidence/sft-study-v1.2.json), [selected MLflow metadata](../docs/evidence/sft-run-metadata.json), [conversation outputs](../docs/evidence/sft-conversations.json).

Local qualitative judging identified repetitive openings and uneven naturalness. Different judge notes favored different trials. Those notes explain the preference phase, but they do not form a consistent numerical leaderboard.

## Frozen DPO baseline

The later pipeline fixes this adapter:

```text
sft/models/steering-sft-v1.1/trial-17/best_adapter
```

[`dpo/train/paths.py`](../dpo/train/paths.py) defines the path. Trial 17 is the chosen baseline, not the lowest-loss v1.1 trial. A full selection scorecard explaining its choice over trial 16 was not recovered.

Its saved configuration records rank 16, alpha 32, dropout 0.1, and targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. It uses learning rate `0.00012074558410393839`, three epochs, a linear scheduler, and effective batch four. MLflow records validation loss `2.2226924896240234`.

DPO keeps this adapter frozen. Deployment exports its contribution together with the selected DPO contribution into one LoRA.

## Dataset provenance limit

Saved study metadata logs SHA-256 prefix `ced5b49dd3d0`. The current local `SFT_DATA_V3.jsonl` has 500 rows and hash prefix `958f1a0131d2`. They are different files. Neither the current script nor the filename proves which exact bytes trained trial 17.

The audit did not recover an exact hash match in the supplied dataset locations. [Data provenance](../docs/data.md) preserves both facts rather than treating the current file as the historical input.

## Run a new study

Prepare the [environment and model assets](../REPRODUCIBILITY.md), supply the dataset, and review the configuration constants in `train_sft.py`. Use a new experiment name and output location to preserve existing adapters.

```bash
conda activate saulgman
python sft/train_sft.py
```

This starts training with the current constants. It does not recreate trial 17 exactly.

## Evaluate SFT

The [SFT skeletons](sft_eval/eval_skeletons.json) and [judge rubric](sft_eval/llm_judge_prompt.md) are tracked. After restoring adapters, review paths and model lists before running the legacy scripts:

```bash
bash sft/sft_eval/deploy_qwenie_eval.sh
python sft/sft_eval/eval_generate_vllm.py
```

Use the [saved comparison](../docs/evidence/sft-conversations.json) to inspect original outputs without GPU services. Later DPO checks live in [`dpo/eval`](../dpo/eval/README.md).
