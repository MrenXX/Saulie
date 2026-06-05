# v1.5 eval slate merge validation

Gates: **Delta-W** (blocking) | local stack↔cat | **BnB cat↔FP8 cat** (deploy proxy)

| Trial | exit | ΔW max | stack↔cat max | top1 | stack smoke | cat smoke | verdict | FP8↔BnB max | FP8 top1 | bnb cat | fp8 cat | FP8 regress |
|------:|-----:|-------:|--------------:|-----:|------------:|----------:|---------|--------------:|---------:|--------:|--------:|-------------:|
| 19 | 0 | 1.2e-09 | 1.50 | 0.71 | 7/7 | 7/7 | keep_cat_for_eval | 1.38 | 0.86 | 7/7 | 7/7 | 0 |
| 16 | 0 | 1.2e-09 | 1.33 | 0.86 | 7/7 | 7/7 | keep_cat_for_eval | 1.12 | 1.00 | 7/7 | 7/7 | 0 |
| 8 | 0 | 1.4e-09 | 1.16 | 0.71 | 7/7 | 7/7 | keep_cat_for_eval | 1.25 | 1.00 | 7/7 | 7/7 | 0 |
| 27 | 0 | 1.4e-09 | 1.03 | 1.00 | 7/7 | 7/7 | keep_cat_for_eval | 1.27 | 0.71 | 7/7 | 7/7 | 0 |
| 20 | 0 | 9.3e-10 | 1.19 | 0.86 | 6/7 | 7/7 | keep_cat_for_eval | 1.38 | 0.86 | 7/7 | 7/7 | 0 |
| 4 | 0 | 1.4e-09 | 1.00 | 0.86 | 7/7 | 7/7 | keep_cat_for_eval | 1.12 | 0.86 | 7/7 | 7/7 | 0 |

Notes:
- FP8 path is HF `MODEL_ID_FP8` + cat LoRA (proxy for vLLM FP8+LoRA).
- Forward drift thresholds are diagnostic only; export gated by ΔW only.
- `FP8 regress` = prompts where FP8 gen tags are worse than BnB cat alone.
