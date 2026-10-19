# Reproducibility

Git contains the code, DPO input and split, saved study results, and selected synthetic evaluation outputs. It does not contain every artifact needed for a fresh GPU deployment or an exact historical SFT rerun.

## Inspect the project without a GPU

```bash
git clone https://github.com/MrenXX/Saulie.git
cd Saulie
python scripts/check_docs.py
```

The checker uses the Python standard library. It checks local Markdown targets and anchors, exported evidence hashes, preference-data counts and split IDs, and saved evaluation coverage. It does not start models or containers.

Read [history](docs/project-history.md), [SFT](sft/README.md), [DPO](dpo/README.md), and [evaluation](docs/evaluation.md) for the methods and their limits.

## Environment

The historical target is Ubuntu 22.04 under WSL2 with NVIDIA CUDA support, Docker GPU access, Conda, and tmux. The training code was designed around a 12 GB memory budget, but not every saved run establishes a specific GPU model.

Several entry points assume `/root/saulie`. Use that layout in a dedicated environment, or review hardcoded paths before moving it. The Python recipe is [`environment.yml`](environment.yml), with versions in [`requirements.txt`](requirements.txt). Docker references are in [`docker/versions.env`](docker/versions.env), captured from the June 2026 environment.

```bash
cd /root/saulie
conda env create -f environment.yml
conda activate saulgman
cp .env.example .env
```

Replace the example keys and confirm paths. Match `LLM_API_KEY` and `VLLM_API_KEY`. The BGE container has its own Python and TensorRT environment; the Conda package list does not lock all of its dependencies.

## External artifacts

Supply these before starting the full stack:

- `Qwen3-4B-Instruct-2507/`: [upstream BF16 checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), used for training with bitsandbytes quantization.
- `Qwen3-4B-Instruct-2507-FP8/`: [upstream FP8 checkpoint](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-FP8), used for serving.
- `sft/models/steering-sft-v1.1/trial-17/best_adapter/`: the trained SFT adapter. It has no published download in this repo.
- `dpo/train/models/steering-dpo-v1.5/optuna-run-20260602-052732/trial-4/sft_dpo_cat/`: the combined serving adapter. It also has no published download.
- `rag/embed_models/bge-m3/`: [BGE-M3 files](https://huggingface.co/BAAI/bge-m3), including tokenizer and sparse projection weights.
- `rag/embed_models/bge-m3/onnx/model.onnx` and any external tensor data, followed by a GPU-compatible `bge_m3.engine`.
- A populated Qdrant collection, rebuilt from catalog metadata or restored separately.

Do not overwrite the frozen SFT adapter while experimenting. Full model snapshots should retain their config and tokenizer files, not just weight tensors. [Artifact manifest](docs/evidence/README.md)

## BGE and catalog setup

The tracked build script compiles an existing ONNX model. It does not export BGE from PyTorch to ONNX. The audited machine has that export and its external tensor data, but a complete export recipe is not published. This is a real clean-machine reproduction gap.

Once a compatible export and the other BGE files exist:

```bash
bash docker/setup_containers.sh
docker exec tensorrt_bge-m3 bash /workspace/build_engine.sh
docker exec -d tensorrt_bge-m3 sh -c 'cd /workspace && python -u serve.py'
curl -X POST http://127.0.0.1:8888/embed \
  -H 'Content-Type: application/json' \
  -d '{"text":"healthcheck"}'
```

The build script backs up an existing engine. Rebuilding for a GPU does not validate the quality of the exported model. Confirm finite normalized dense vectors and valid sparse outputs. The engine profile is batch at most two and sequence length at most 256.

Follow [the RAG guide](rag/README.md) to download catalog metadata, prepare records, index with batch two, and validate the collection. Run its relative-path commands from `rag/`.

## Start the demo

After the assets and catalog exist:

```bash
conda activate saulgman
bash start_saulie.sh
curl http://127.0.0.1:9000/health
```

[Deployment](docs/deployment.md) covers requests, remote-client constants, ports, and troubleshooting. [Security](SECURITY.md) describes the actual demo bindings and authentication limits.

## Training reproduction

For a new SFT run, supply a dataset and review the trainer constants. Exact historical SFT reproduction is blocked by the missing input with logged hash prefix `ced5b49dd3d0`, unpublished adapter assets, and later trainer edits. The current 500-row local V3 file has a different hash.

For a new DPO run, the V4 input and seed-42 split are tracked. Training still requires the frozen SFT adapter and model checkpoints. [DPO commands](dpo/README.md) describe the current v1.5 entry points, which create new GPU work.

Saved study summaries support inspection. They do not guarantee identical Optuna scheduling, trial numbering, sampling, or numerical results on another device. Two workers and runtime failures affected the original study.

## Validation performed for this overhaul

The documentation checker, artifact hashes, saved counts, command flags, report generation, and source-state preservation were checked. The full GPU environment was not rebuilt. Models were not retrained, catalogs were not reindexed, and the user's live services were not restarted.

Remaining artifact recovery and verification work is listed in [the audit](docs/documentation-audit.md).
