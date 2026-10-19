# Supervised fine-tuning

SFT teaches the conversational task before DPO refines it. I trained Qwen3-4B-Instruct-2507 on about 500 synthetic dialogues in which the user starts with a situation rather than a shopping request. The assistant has to engage, ask useful questions, and reach a justified product category.

Recommendations stay generic during training. Specific items come from [retrieval](../rag/README.md) at inference time, so changing the catalog does not require another fine-tune. [Data generation](../docs/data.md)

## Input and loss

[`train_sft.py`](train_sft.py) reads JSONL rows with `messages` and `metadata`. Opening type and length bucket are used for stratification. Seed 42 produces 400 training, 50 validation, and 50 test conversations for the 500-row set.

Only assistant tokens contribute to loss. `patch_chat_template_for_assistant_loss` adds generation markers to the Qwen template so TRL can build the assistant mask. The sequence limit is 832 tokens, packing is disabled, and gradient checkpointing reduces memory use.

The standard training path loads the BF16 checkpoint with bitsandbytes 8-bit quantization. LoRA updates the attention and MLP projections while the base stays frozen. The script also retains an FP8 loading option from the initial experiments; the selected training setup uses bitsandbytes.

## Training studies

I used Optuna to search hyperparameters and MLflow to track runs. Each trial reloads the base, trains an adapter, evaluates validation loss, and saves its best checkpoint.

- v1.0 explored a broad range in 13 trials. It exposed loss on user tokens and a warmup configuration bug.
- v1.1 corrected both, added weight decay and NEFTune noise, and ran 20 trials.
- v1.2 fixed most settings and narrowed the remaining search. Its configuration requests 15 trials; the included summary contains seven completed trials.

With so few dialogues, weight decay and NEFTune noise were intended to reduce overfitting to the generated examples.

The current script contains the v1.2 defaults: rank 16, alpha 32, dropout 0.1, and a search over learning rate, epochs, accumulation, and clipping. It fixes the scheduler to `constant_with_warmup`. Earlier results include linear schedulers, so use the run parameters when reproducing an earlier study.

[v1.1 results](../docs/evidence/sft-study-v1.1.json), [v1.2 results](../docs/evidence/sft-study-v1.2.json), [run metadata](../docs/evidence/sft-run-metadata.json)

## Selecting trial 17

Validation loss was useful for shortlisting, but it did not measure whether a conversation reached a sensible recommendation. The best three v1.1 losses were 2.2194 for trial 16, 2.2211 for trial 15, and 2.2227 for trial 17. These differences were small.

I generated conversations from the shortlisted adapters and judged six aspects: steering, questions, recommendation landing, tone, naturalness, and difficult openings. Trial 17 was selected through that review despite ranking third by loss. It became the frozen starting point and reference for DPO.

The [comparison outputs](../docs/evidence/sft-conversations.json) contain the base and ten SFT candidates on 14 scenarios each. The [judge rubric](sft_eval/llm_judge_prompt.md) defines the dimensions. The models learned the broad steering pattern, but repetitive openings and weak phrasing remained, which motivated the preference stage.

Trial 17 uses rank 16, alpha 32, dropout 0.1, learning rate `0.00012074558410393839`, three epochs, a linear scheduler, and effective batch size four. Targets are `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

The adapter path used by the later pipeline is:

```text
sft/models/steering-sft-v1.1/trial-17/best_adapter
```

## Run training

Prepare the [environment and external assets](../REPRODUCIBILITY.md), supply the dataset, and review the constants at the top of `train_sft.py`. Set a new experiment name to keep its output separate from the existing adapters.

```bash
conda activate saulgman
python sft/train_sft.py
```

The March runs logged dataset hash prefix `ced5b49dd3d0`. The current local V3 copy has a different hash, so this command starts a new study rather than exactly recreating trial 17. [Dataset versions](../docs/data.md#dataset-versions-and-hashes)

## Generate evaluation conversations

Restore the adapters and review paths and model lists in the evaluation scripts:

```bash
bash sft/sft_eval/deploy_qwenie_eval.sh
python sft/sft_eval/eval_generate_vllm.py
```

The [14 scenarios](sft_eval/eval_skeletons.json) fix the user turns. Each adapter generates the assistant turns for review. [DPO](../dpo/README.md) then uses preferred and rejected continuations to refine the selected SFT behavior.
