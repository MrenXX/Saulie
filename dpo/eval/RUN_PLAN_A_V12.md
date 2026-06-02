# Plan A v1.2 — run commands

## Preflight

```bash
cd /root/saulie && conda activate saulgman
python dpo/train/scripts/preflight_plan_a_v12.py
```

## Train (sequential, 2 fixed trials)

```bash
bash dpo/train/scripts/run_plan_a_v12.sh
```

Or manually set `RUN_DIR` and launch; then babysit in another terminal:

```bash
python dpo/train/scripts/babysit_study.py --run-dir "$RUN_DIR" --target-complete 2 --poll-min 3
tail -f "$RUN_DIR/worker_0.log"
```

## MLflow

Experiment: `steering-dpo-v1.2`

```bash
mlflow ui --backend-store-uri file:///root/saulie/dpo/train/mlruns --port 5001
```

Child runs: `trial-0` = minimal DPO (`plan_a_minimal_dpo`), `trial-1` = IPO (`plan_a_ipo`) — order depends on Optuna queue.

## Manual conversation gate (your REPL)

```bash
python dpo/eval/chat_policy_stack.py \
  --dpo-adapter "$RUN_DIR/trial-0/best_adapter" \
  --dpo-weight 1.0

python dpo/eval/chat_policy_stack.py \
  --dpo-adapter "$RUN_DIR/trial-1/best_adapter" \
  --dpo-weight 1.0
```

SFT baseline for comparison:

```bash
python dpo/eval/chat_policy_stack.py --adapter-mode sft
```

Check `diagnostics.json` and MLflow for `high_margin_warning` if margin > 5.
