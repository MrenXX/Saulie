# Saulie

DPO steering training (Qwen + LoRA). See `dpo/` for code, dataset, and Optuna study results.

**Review a finished study (HTML):** open `dpo/results/optuna-run-20260523-041252/study_report.html` or regenerate:

```bash
python -m dpo.train.study_report --summary dpo/results/optuna-run-20260523-041252/trial_summary.json
```

Full training needs GPU, SFT adapter, and deps in `dpo/requirements-dpo.txt`.
