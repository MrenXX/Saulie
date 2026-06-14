"""
Conversational Steering SFT Training Script
=============================================
Model: Qwen3-4B-Instruct-2507 (FP8 or bf16+BnB)
Stack: Transformers + PEFT + TRL + Optuna + MLflow
Hardware: RTX 4070 12GB VRAM (WSL2)
"""

import os
import gc
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime
import traceback

# Tells Pytorch to request expandable blocks of memory from CUDA allowing for less fragmented memory between trials of training
# Needs to be imported before `import torch` 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import mlflow
import optuna
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig

from sklearn.model_selection import train_test_split
from collections import Counter

from colorama import Fore, Style, init as colorama_init

warnings.filterwarnings("ignore", category=FutureWarning)
colorama_init(autoreset=True)


# ============================================================
# 0. CONFIGURATION
# ============================================================

# --- Paths (EDIT THESE) ---
ABS_PATH        = Path(r"/root/saulie")
DATA_PATH       = ABS_PATH / "sft" / "SFT_DATA_V3.jsonl"
OUTPUT_BASE     = ABS_PATH / "sft" / "models"
MLRUNS_DIR      = ABS_PATH / "sft" / "mlruns"
EVAL_PROMPTS    = ABS_PATH / "sft" / "eval_prompts_template.json"

# --- Model ---
# Option A: FP8 (try first) - using local models
MODEL_ID_FP8    = ABS_PATH / "Qwen3-4B-Instruct-2507-FP8"
# Option B: bf16 + BnB 8-bit (fallback)
MODEL_ID_BF16   = ABS_PATH / "Qwen3-4B-Instruct-2507"

USE_QUANT_VERSION         = False   # Set to False to use BnB fallback

# --- Training constants ---
MAX_SEQ_LEN     = 832 #Longest sample is 785 tokens but we leave headroom for special tokens and make it divisible by 64  which helps GPU in memory alignment  
SEED            = 42
N_OPTUNA_TRIALS = 15
EXPERIMENT_NAME = "steering-sft-v1.2"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. HELPERS
# ============================================================

class EvalLossLoggerCallback(TrainerCallback):
    """Logs eval metrics with color for visibility."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "eval_loss" in logs:
            print(f"\n{Fore.CYAN}{'--- Eval ---':^40}{Style.RESET_ALL}")
            for k, v in logs.items():
                val = f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                print(f"  {Fore.YELLOW}{k:<28}{Style.RESET_ALL} {val}")
            print()


def compute_data_hash(path: Path) -> str:
    """SHA256 of the data file for reproducibility tracking."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def clear_gpu():
    """Free GPU memory between Optuna trials."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def patch_chat_template_for_assistant_loss(tokenizer):
    """
    Patch Qwen3-Instruct-2507 chat template to add {% generation %} /
    {% endgeneration %} tags around assistant content.

    TRL's assistant_only_loss requires these tags to build the token mask
    that restricts loss computation to assistant responses only.

    The official Qwen3-4B-Instruct-2507 template does NOT include these
    tags. This patch follows the fix published by Unsloth (commit 9ae8300).

    The patch:
      - Removes thinking/reasoning logic (not used by Instruct models)
      - Removes the multi_step_tool reverse-scan block
      - Adds {% generation %} before assistant content
      - Adds {% endgeneration %} after assistant <|im_end|>
    """
    template = tokenizer.chat_template

    if "{% generation %}" in template and "{% endgeneration %}" in template:
        print(f"  Chat template already has generation tags. No patch needed.")
        return

    if "<|im_start|>" not in template:
        raise ValueError(
            "Chat template does not look like a Qwen3 ChatML template. "
            "Cannot safely patch. Aborting."
        )

    import re
    template = re.sub(
        r'\{%-?\s*set\s+ns\s*=\s*namespace\(multi_step_tool.*?\{%-?\s*endfor\s*-?%\}',
        '',
        template,
        flags=re.DOTALL
    )

    template = re.sub(
        r"\{%-?\s*set\s+reasoning_content\s*=\s*''.*?"
        r"(?=\{%-?\s*if\s+message\.tool_calls)",
        '',
        template,
        flags=re.DOTALL
    )

    template = re.sub(
        r"\{\{-?\s*'<\|im_start\|>'\s*\+\s*message\.role\s*\+\s*'\\n'\s*\+\s*content\s*\}\}",
        "{{- '<|im_start|>' + message.role + '\\n'}}{% generation %}{{- content }}",
        template
    )

    template = re.sub(
        r"(\{\{-?\s*'<\|im_end\|>\\n'\s*\}\})\s*(\{%-?\s*elif\s+message\.role\s*==\s*[\"']tool[\"'])",
        r"\1\n    {% endgeneration %}\n  \2",
        template
    )

    tokenizer.chat_template = template

    test_messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    try:
        result = tokenizer.apply_chat_template(
            test_messages,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        mask = result.get("assistant_masks", [])
        if sum(mask) == 0:
            raise ValueError("Patch applied but assistant_masks is all zeros")
        print(f"  Chat template patched successfully.")
        print(f"  Validation: {sum(mask)}/{len(mask)} tokens marked as assistant.")
    except Exception as e:
        raise RuntimeError(
            f"Chat template patch failed validation: {e}\n"
            f"The template may have changed from the expected format.\n"
            f"FALLBACK: Remove this patch call and use "
            f"DataCollatorForCompletionOnlyLM instead (see guide)."
        )


# ============================================================
# 2. DATA LOADING AND PRE-TOKENIZATION
# ============================================================

def load_and_split_data(data_path, seed=42):
    dataset = load_dataset("json", data_files=str(data_path), split="train")
    
    # Build stratification key from metadata
    # Combines opening_type + turn bucket so splits are balanced on both
    def add_strat_key(row):
        otype = row["metadata"]["opening_type"]
        turns = row["metadata"]["turns"]
        turn_bucket = "short" if turns <= 4 else "mid" if turns <= 8 else "long"
        row["_strat_key"] = f"{otype}_{turn_bucket}"
        return row
    
    dataset = dataset.map(add_strat_key)
    
    # Stratified split using sklearn
    indices = list(range(len(dataset)))
    strat_labels = dataset["_strat_key"]
    
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.1, random_state=seed, stratify=strat_labels
    )
    
    train_val_labels = [strat_labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.111, random_state=seed, stratify=train_val_labels
    )
    
    splits = DatasetDict({
        "train":      dataset.select(train_idx),
        "validation": dataset.select(val_idx),
        "test":       dataset.select(test_idx),
    })
    
    
    # Print distribution verification
    for split_name in ["train", "validation", "test"]:
        types = [r["metadata"]["opening_type"] for r in splits[split_name]]
        turns = [r["metadata"]["turns"] for r in splits[split_name]]
        print(f"  {split_name}: n={len(types)}, "
              f"types={dict(Counter(types))}, "
              f"turn_avg={sum(turns)/len(turns):.1f}")
    
    # Drop the helper columns
    # splits = splits.remove_columns(["_strat_key"])
    cols_to_keep = {"messages"}
    cols_to_drop = [c for c in splits["train"].column_names if c not in cols_to_keep]
    splits = splits.remove_columns(cols_to_drop)

    return splits


def report_token_lengths(splits: DatasetDict, tokenizer):
    """
    Apply chat template + tokenize to report token length stats.
    This is informational only — the returned dataset is NOT used for training.
    SFTTrainer handles tokenization internally when using assistant_only_loss.
    """

    def process(row):
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
        tokenized = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN)

        # Ensure EOS at the end, if not add it and append a 'True' to the attention_mask for it
        if tokenized["input_ids"] and tokenized["input_ids"][-1] != tokenizer.eos_token_id:
            tokenized["input_ids"].append(tokenizer.eos_token_id)
            tokenized["attention_mask"].append(1)

        tokenized["seq_len"] = len(tokenized["input_ids"])
        return tokenized

    # Tokenize all splits
    original_cols = list(splits["train"].features.keys())
    tokenized = splits.map(process, remove_columns=original_cols) # Drop all columns just keep the returned tokenized from process()

    # --- Token length report ---
    print(f"\n{Fore.CYAN}--- Token Length Report ---{Style.RESET_ALL}")
    for split_name in ["train", "validation", "test"]:
        lengths = tokenized[split_name]["seq_len"]
        max_len = max(lengths)
        avg_len = sum(lengths) / len(lengths)
        over_limit = sum(1 for l in lengths if l >= MAX_SEQ_LEN)
        print(f"  {Fore.YELLOW}{split_name:<12}{Style.RESET_ALL} "
              f"max={max_len}, avg={avg_len:.0f}, "
              f"truncated={over_limit}/{len(lengths)}")

    # Drop the helper column
    tokenized = tokenized.remove_columns(["seq_len"])
    return tokenized


# ============================================================
# 3. MODEL LOADING
# ============================================================

def load_model_and_tokenizer():
    """
    Load Qwen3-4B with either FP8 or BnB 8-bit quantization.
    Returns (model, tokenizer, model_id_used).
    """
    if USE_QUANT_VERSION:
        model_id = MODEL_ID_FP8
        print(f"{Fore.GREEN}Loading FP8 model: {model_id}{Style.RESET_ALL}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype="auto", # Quantized models have mixed precisions depending on layer so 'auto' works best here
                device_map={"": 0},
            )
            print(f"{Fore.GREEN}FP8 model loaded successfully.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}FP8 loading failed: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Falling back to BnB 8-bit...{Style.RESET_ALL}")
            return _load_bnb_model()
    else:
        return _load_bnb_model()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    _setup_tokenizer(tokenizer, model)
    return model, tokenizer, model_id


def _load_bnb_model():
    """Fallback: load bf16 model with BitsAndBytes 8-bit quantization."""
    model_id = MODEL_ID_BF16
    print(f"{Fore.GREEN}Loading BnB 8-bit model: {model_id}{Style.RESET_ALL}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    _setup_tokenizer(tokenizer, model)
    return model, tokenizer, model_id


def _setup_tokenizer(tokenizer, model):
    """
    Qwen3 already has a dedicated pad token:
      pad_token = <|endoftext|>  (id: 151643)
      eos_token = <|im_end|>     (id: 151645)
    They are DIFFERENT, so no Llama-style hacks needed.
    We just verify and set padding_side.
    """
    # Verify pad token exists and differs from eos
    print(f"  pad_token: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})") # !r returns the "official" string representation of an object
    print(f"  eos_token: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")

    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        # Safety fallback, but this shouldn't happen with Qwen3
        print(f"{Fore.YELLOW}WARNING: pad_token missing or equals eos_token. "
              f"Setting pad_token to <|endoftext|>{Style.RESET_ALL}")
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = 151643

    tokenizer.padding_side = "right"  # Standard for causal LM training

    # Sync model config
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"  padding_side: {tokenizer.padding_side}")

    # NEW: Patch chat template for assistant-only loss
    patch_chat_template_for_assistant_loss(tokenizer)


# ============================================================
# 4. LORA CONFIG (fixed across all Optuna trials)
# ============================================================


# LORA_CONFIG = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
#                      "gate_proj", "up_proj", "down_proj"],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM",
# )


# ============================================================
# 5. OPTUNA OBJECTIVE
# ============================================================

def create_objective(base_model_loader, dataset_splits, model_id_used):
    """
    Returns an Optuna objective function. Each trial:
      1. Suggests hyperparameters.
      2. Loads a fresh model + LoRA.
      3. Trains with SFTTrainer.
      4. Returns eval_loss (Optuna minimizes this).
      5. Logs everything to MLflow.
    """

    def objective(trial: optuna.Trial) -> float:
        clear_gpu()

        # --- Suggest hyperparameters ---
        # Ranges informed by Unsloth guide + community best practices for small models


        lora_r = 16
        lora_alpha = lora_r * 2
        lora_dropout = 0.1
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        trial_lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        lr = trial.suggest_float("learning_rate", 8e-5, 2e-4, log=True)
        
        lr_scheduler = "constant_with_warmup"
        epochs = trial.suggest_int("num_train_epochs", 2, 4)


        # batch_size = trial.suggest_categorical("per_device_train_batch_size", [1, 2])
        
        batch_size = 1
        grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [2, 4])

        # if batch_size == 2:
        #     grad_accum = 2  # fixed — batch=2 + grad_accum=4 OOMs on 12GB
        # else:
        #     grad_accum = trial.suggest_categorical("gradient_accumulation_steps", [2, 4])


        warmup_ratio = 0.1
        max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5])
        
        weight_decay = 0.05

        effective_batch = batch_size * grad_accum
        if effective_batch < 4 or effective_batch > 8:
            raise optuna.exceptions.TrialPruned()

        trial_name = f"trial-{trial.number}"
        trial_output_dir = OUTPUT_BASE / EXPERIMENT_NAME / trial_name

        print(f"\n{Fore.GREEN}{'='*60}")
        print(f" TRIAL {trial.number}")
        print(f"  lr={lr:.2e}, scheduler={lr_scheduler}, epochs={epochs}")
        print(f"  batch={batch_size}, grad_accum={grad_accum}, effective_batch={effective_batch}")
        print(f"  warmup_ratio={warmup_ratio:.2f}, grad_norm={max_grad_norm}, weight_decay={weight_decay}")
        print(f"  lora_r={lora_r}, lora_alpha={lora_alpha}, lora_dropout={lora_dropout}")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        # --- Load fresh model for this trial ---
        model, tokenizer, _ = base_model_loader()

        # --- SFT config ---
        sft_args = SFTConfig(
            output_dir=str(trial_output_dir),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=grad_accum,
            optim="paged_adamw_32bit",
            learning_rate=lr,
            lr_scheduler_type=lr_scheduler,
            weight_decay=weight_decay,
            bf16=True,
            tf32=True,
            max_grad_norm=max_grad_norm,
            warmup_ratio=warmup_ratio,

            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,

            max_length=MAX_SEQ_LEN,
            packing=False,
            # Trains model only on assistant responses and not user msgs in conversational datasets
            assistant_only_loss=True,
            # Adds a bit of noise > really good for preventing overfitting on small datasets which is a big risk with small models aswell
            neftune_noise_alpha=5.0,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},

            seed=SEED,
            report_to="none",
        )        

        # --- Train ---
        with mlflow.start_run(run_name=trial_name, nested=True):
            mlflow.log_params({
                "trial_number": trial.number,
                "model_id": model_id_used,
                "learning_rate": lr,
                "lr_scheduler_type": lr_scheduler,
                "num_train_epochs": epochs,
                "gradient_accumulation_steps": grad_accum,
                "effective_batch_size": effective_batch,
                "warmup_ratio": warmup_ratio,
                "max_grad_norm": max_grad_norm,
                "weight_decay": weight_decay,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": ",".join(target_modules),
                "max_seq_len": MAX_SEQ_LEN,
                "neftune_noise_alpha": 5.0,
            })

            try:
                # --- Build trainer ---
                trainer = SFTTrainer(
                    model=model,
                    processing_class=tokenizer,
                    train_dataset=dataset_splits["train"],
                    eval_dataset=dataset_splits["validation"],
                    peft_config=trial_lora_config,
                    callbacks=[EvalLossLoggerCallback()],
                    args=sft_args,
                )
                train_result = trainer.train()
            except torch.cuda.OutOfMemoryError as e:
                print(f"{Fore.RED}[Trial {trial.number}] CUDA OOM occurred.{Style.RESET_ALL}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                traceback.print_exc()

                clear_gpu()
                del model, trainer
                raise optuna.exceptions.TrialPruned()

            except Exception as e:
                print(f"{Fore.RED}[Trial {trial.number}] Training failed with an unexpected error.{Style.RESET_ALL}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {e}")
                traceback.print_exc()

                clear_gpu()
                del model, trainer
                raise

            # --- Eval on validation (best checkpoint) ---
            eval_results = trainer.evaluate()
            eval_loss = eval_results["eval_loss"]

            mlflow.log_metrics({
                "train_loss": train_result.training_loss,
                "eval_loss": eval_loss,
            })

            # --- Save adapter for this trial ---
            adapter_dir = trial_output_dir / "best_adapter"
            trainer.save_model(str(adapter_dir))
            mlflow.log_param("adapter_path", str(adapter_dir))

            print(f"\n{Fore.CYAN}Trial {trial.number} complete: "
                  f"eval_loss={eval_loss:.4f}{Style.RESET_ALL}\n")

        # Cleanup
        del model, trainer
        clear_gpu()

        return eval_loss

    return objective


# ============================================================
# 6. MAIN
# ============================================================

def main():
    torch.manual_seed(SEED)

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" Conversational Steering SFT Training")
    print(f" Model: {'FP8' if USE_QUANT_VERSION else 'BnB 8-bit'}")
    print(f" Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
        print(f" VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}{Style.RESET_ALL}\n")

    # --- Load and prep data ---
    splits = load_and_split_data(DATA_PATH, seed=SEED)
    data_hash = compute_data_hash(DATA_PATH)
    print(f"Data hash: {data_hash}")

    # --- First model load: just for tokenizer + tokenization ---
    print(f"\n{Fore.CYAN}Loading model for tokenizer setup...{Style.RESET_ALL}")
    model, tokenizer, model_id_used = load_model_and_tokenizer()

    # --- Report token lengths (informational only) ---
    # print(f"\n{Fore.CYAN}Reporting token lengths...{Style.RESET_ALL}")
    # report_token_lengths(splits, tokenizer)

    # --- Cleanup: we'll reload per trial ---
    del model
    clear_gpu()

    # --- Model loader factory (called fresh per Optuna trial) ---
    def model_loader():
        return load_model_and_tokenizer()

    # --- MLflow setup ---
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    # --- Optuna study ---
    study = optuna.create_study(
        direction="minimize",
        
        sampler=optuna.samplers.TPESampler(seed=SEED),

        # This kills trials mid training if they're hopefless (e.g. loss isnt getting lower) NopPruner means dont kill runs since we're training on short epochs
        pruner=optuna.pruners.NopPruner(),
        study_name=EXPERIMENT_NAME,
    )

    objective_fn = create_objective(model_loader, splits, model_id_used)

    # Run inside a parent MLflow run
    with mlflow.start_run(run_name=f"optuna-study-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_params({
            "n_trials": N_OPTUNA_TRIALS,
            "model_id": model_id_used,
            "USE_QUANT_VERSION": USE_QUANT_VERSION,
            "data_hash": data_hash,
            "data_path": str(DATA_PATH),
            "max_seq_len": MAX_SEQ_LEN,
            "train_size": len(splits["train"]),
            "val_size": len(splits["validation"]),
            "test_size": len(splits["test"]),
            "neftune_noise_alpha": 5.0,
        })

        study.optimize(objective_fn, n_trials=N_OPTUNA_TRIALS)

        # --- Log study summary ---
        best = study.best_trial
        print(f"\n{Fore.GREEN}{'='*60}")
        print(f" OPTUNA STUDY COMPLETE")
        print(f"  Best trial: #{best.number}")
        print(f"  Best eval_loss: {best.value:.4f}")
        print(f"  Best params:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"{'='*60}{Style.RESET_ALL}")

        mlflow.log_metrics({"best_eval_loss": best.value, "best_trial_number": best.number})
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()})

        # --- Save trial ranking ---
        trial_summary = []
        for t in sorted(study.trials, key=lambda x: x.value if x.value else float("inf")):
            if t.state == optuna.trial.TrialState.COMPLETE:
                trial_summary.append({
                    "trial": t.number,
                    "eval_loss": round(t.value, 4),
                    "params": t.params,
                })
        summary_path = OUTPUT_BASE / EXPERIMENT_NAME / "trial_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(trial_summary, f, indent=2)
        mlflow.log_artifact(str(summary_path))

    # --- Final: eval best model on test set ---
    print(f"\n{Fore.CYAN}Evaluating best model on held-out test set...{Style.RESET_ALL}")
    clear_gpu()

    best_adapter_path = OUTPUT_BASE / EXPERIMENT_NAME / f"trial-{best.number}" / "best_adapter"

    # Load base model
    if USE_QUANT_VERSION:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_FP8, dtype="auto", device_map={"": 0}
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID_BF16,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map={"": 0}, dtype=torch.bfloat16,
        )

    # Load the TRAINED adapter (not a fresh random one)
    model = PeftModel.from_pretrained(model, str(best_adapter_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_FP8 if USE_QUANT_VERSION else MODEL_ID_BF16)
    tokenizer.padding_side = "right"

    # Patch chat template for assistant-only loss
    patch_chat_template_for_assistant_loss(tokenizer)

    from trl import SFTConfig as TestSFTConfig

    test_args = TestSFTConfig(
        output_dir=str(OUTPUT_BASE / EXPERIMENT_NAME / "test_eval"),
        per_device_eval_batch_size=2,
        bf16=True,
        report_to="none",
        assistant_only_loss=True,
        max_length=MAX_SEQ_LEN,
    )

    test_trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=splits["test"], # Not used since we're calling .evaluate() which doesnt do backprop but necessary for SFTTrainer()
        eval_dataset=splits["test"],
        args=test_args,
    )

    test_results = test_trainer.evaluate()
    print(f"\n{Fore.GREEN}Test set eval_loss: {test_results['eval_loss']:.4f}{Style.RESET_ALL}")

    with mlflow.start_run(run_name="test-eval-best"):
        mlflow.log_params({"source_trial": best.number, "adapter_path": str(best_adapter_path)})
        for k, v in test_results.items():
            try:
                mlflow.log_metric(k, float(v))
            except (TypeError, ValueError):
                pass

    print(f"\n{Fore.GREEN}{'='*60}")
    print(f" ALL DONE")
    print(f"  Best adapter:        {best_adapter_path}")
    print(f"  Trial summary:       {summary_path}")
    print(f"  MLflow dashboard:    mlflow ui --backend-store-uri file://{MLRUNS_DIR}")
    print(f"{'='*60}{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
