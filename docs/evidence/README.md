# Evidence files

These files preserve selected small results from the original Windows and WSL work. The export date is October 19, 2026. Historical results retain their original contents; they were not regenerated for the documentation rewrite.

- [Export manifest](export-manifest.json): original source labels, destination paths, byte sizes, and SHA-256 hashes for copied files.
- [SFT v1.1](sft-study-v1.1.json) and [v1.2](sft-study-v1.2.json): original saved completed-trial rankings.
- [SFT run metadata](sft-run-metadata.json): selected MLflow parameters and final metrics, with machine-specific source paths reduced to project-relative labels.
- [SFT conversations](sft-conversations.json): base and ten SFT candidates, 14 scenarios per model.
- [Data provenance](data-provenance.json): audited dataset counts, hashes, schema keys, and exact parsed-row overlap counts. An overlap is not proof that a dataset trained a specific adapter.
- [DPO study history](dpo-study-history.json): selected saved snapshots for v1.0 through v1.5. Old RUNNING counts are preserved as recorded, not presented as live processes.
- [Legacy Windows evidence](legacy-windows.json): hashes, dates, and configuration lines from the early Llama scripts. It establishes their setup, not successful completed training.
- [Selection record](selection-record.json): a paraphrase of the author's recorded DPO finalist and deployment choices. Raw private conversations are not included.

The recovered [SFT scenarios](../../sft/sft_eval/eval_skeletons.json) and [base/SFT/DPO outputs](../../dpo/eval/generations_prod_ladder.json) live beside their evaluation code.

Not distributed: model weights, trained adapters, the historical SFT input, BGE ONNX and TensorRT artifacts, catalog files, Qdrant storage, complete MLflow stores, or raw agent chats. [Reproduction boundaries](../../REPRODUCIBILITY.md)

Run `python scripts/check_docs.py` from the repository root to check links, copied evidence hashes, and saved counts.

Exported JSON evidence and the hashed training inputs use LF through targeted `.gitattributes` rules. Other legacy file formatting is preserved. The checker uses the committed LF representation when checking hashes in older Windows checkouts with CRLF line endings.
