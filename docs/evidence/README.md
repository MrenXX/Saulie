# Training and evaluation files

Study summaries, conversation outputs, dataset metadata, and reported judge results are included here. Trained weights and full tracking stores are supplied separately.

- [SFT v1.1](sft-study-v1.1.json) and [v1.2](sft-study-v1.2.json) contain completed-trial rankings.
- [SFT run metadata](sft-run-metadata.json) contains MLflow parameters and final metrics.
- [SFT conversations](sft-conversations.json) compare the base and ten SFT adapters on 14 scenarios each.
- [Judge results](judge-results.json) contains the aggregate scores from the project report, with table and page references.
- [Model selection](selection-record.json) summarizes the SFT and DPO choices.
- [Data provenance](data-provenance.json) lists dataset versions, counts, hashes, and parsed-row overlaps.
- [DPO study history](dpo-study-history.json) contains study summaries for v1.0 through v1.5.
- [Early Llama configurations](legacy-windows.json) records the model and dataset paths used by the Windows scripts.
- [File manifest](export-manifest.json) lists the source paths and SHA-256 hashes of the copied summaries, fixtures, and generation outputs.

The [SFT scenarios](../../sft/sft_eval/eval_skeletons.json) and [base/SFT/DPO outputs](../../dpo/eval/generations_prod_ladder.json) are beside their evaluation code.

Run `python scripts/check_docs.py` from the repository root to check links, hashes, and counts. Hashed JSON inputs use LF through targeted `.gitattributes` rules so checks work across Windows and Linux.
