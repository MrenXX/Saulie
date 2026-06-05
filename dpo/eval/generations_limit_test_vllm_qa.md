# vLLM limit test QA — eval_O4_001

**Artifact:** [`generations_limit_test_vllm.json`](generations_limit_test_vllm.json)  
**Backend:** vLLM FP8 + LoRA (runtime load/unload for trial-16 / `candidate_B`)  
**Sampling:** temp=0.7 top_p=0.8 top_k=20 rep_penalty=1.05 max_tokens=256 (`extra_body`)

| Check | SFT (vLLM) | candidate_B (trial-16 cat) |
|-------|:----------:|:--------------------------:|
| No loops / near-verbatim repetition | Pass | Pass |
| Fluent English, no CJK | Pass | Pass |
| No forced product recommendation | Pass | Pass |
| Coherent multi-turn | Pass | Pass |

**Gate:** Pass — proceed to full Round 1 vLLM generation.

Runtime LoRA load/unload confirmed (no container restart between SFT and trial-16).
