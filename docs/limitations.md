# Limitations and further work

## Conversation quality

DPO improved average steering and ordinary-conversation scores, but its gains were uneven. SFT remained slightly stronger on unrelated and emotional/lifestyle openings. The deployed model can also miss the best product category, bundle several ideas into one recommendation, or make factual mistakes outside its task. [Evaluation](evaluation.md)

The evaluation uses fixed user-turn scenarios and two LLM judges. The final human comparison was qualitative. More interactive testing would help measure whether the behavior holds up when users respond freely or continue after a recommendation.

## Data and reproducibility

The training sets are small, and product recommendations end the SFT dialogues. Follow-up questions, objections, and post-purchase discussion need additional examples.

The exact SFT input matching the March run hash is not included. Current V3 has different bytes. Trained adapters, the BGE ONNX export, and the product index also need to be supplied separately. A complete ONNX export recipe and downloadable adapters with model revision IDs would make setup easier. [Setup requirements](../REPRODUCIBILITY.md)

The project report contains aggregate judge results. Raw per-conversation score files and the completed second-round generation JSON are not included.

## Retrieval and tools

The catalog contains metadata snapshots rather than live stock. Retrieval has only a small keyword-scored benchmark. A larger annotated query set would help evaluate product relevance more reliably.

The runtime tool-pressure settings improve missed search calls but can make the model search too early. They need testing across more conversations, especially when a user's need changes mid-session.

## Deployment

The stack targets a single machine and a small number of concurrent users. Authentication and request limiting are basic, and some scripts retain placeholder-key fallbacks. [Security](../SECURITY.md)

Further work includes a web interface for streamed chat and product browsing, broader conversation data, repeatable behavioral checks in CI, and load testing. Voice input/output and replicated services would be separate extensions beyond the current text-only setup.
