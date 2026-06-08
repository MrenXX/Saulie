# Saulie — DPO v1.5 Final Eval

Behavioral evaluation for DPO finalists: **cat-merge validation**, **vLLM FP8** generation, blind **LLM-judge** packets, and Round 1 artifacts.

**Full documentation:** [`dpo/eval/README.md`](dpo/eval/README.md)

This branch is for **eval and merge validation**. Training studies live on `main`; production serving lives on `deployment`; RAG on `rag`.

---

## Quick start

```bash
# 1. Merge gate (finalist cat adapters)
bash dpo/train/scripts/merge_v15_eval_slate.sh

# 2. Deploy vLLM FP8 (runtime LoRA)
MAX_LORA_RANK=64 bash dpo/eval/vllm_scripts/deploy_qwenie_eval.sh

# 3. Round 1 generation (checkpointed; resumes on crash)
python dpo/eval/run_v15_final_eval.py --round 1 --anonymize --skip-deploy
```

**Judge packet (blind):** `dpo/eval/generations_round1.json` + `llm_judge_prompt_dpo.md` + `DATA_CONTEXT.md`

**Deploy pick:** trial **4** (cat `sft_dpo_cat`, vLLM `MAX_LORA_RANK=32`) — see `deployment` branch for serving.
