# Development decisions

I built Saulie to explore whether a small, locally hosted model could lead a useful shopping conversation. The difficult part was getting from an ordinary opening to a justified recommendation. Answering a direct product request was already within the base model's capabilities.

## Choosing the model

I started with Llama-3.2-3B-Instruct. It was light enough to work with, but its tool calling was unreliable for the search step I needed later. I moved to Qwen3-4B-Instruct-2507 for its instruction following and tool use while staying close to the same memory budget.

The task is text-only, so I did not need a multimodal model. An instruction-tuned checkpoint also gave me a better starting point than a raw base model with only a few hundred training examples. LoRA let me train the behavior without updating or storing another full set of base weights.

## Teaching steering before preference

I separated training into SFT and DPO because they solve different problems. SFT demonstrates the task: engage with the opener, ask a useful question, connect the answer to a need, and recommend a category. DPO compares stronger and weaker ways of carrying out that task.

The SFT data deliberately avoids direct requests for products. Those would make the task too easy and spend the limited training set on behavior the base already had. Recommendations are generic categories rather than brands or model names. The catalog supplies real items later, so updating inventory does not require retraining.

The first SFT study exposed two training problems: loss included user tokens, and a misnamed setting disabled warmup. I fixed both in v1.1 and added regularization. v1.2 narrowed the remaining search. Validation loss helped shortlist adapters, but the differences were small and did not tell me which model steered best.

I selected v1.1 trial 17 through generated-conversation review. It was third by validation loss, behind trials 16 and 15. The review considered steering, question quality, the recommendation, tone, naturalness, and difficult openings. That adapter became the frozen starting point for DPO. [SFT training](../sft/README.md)

## Correcting the SFT model's actual mistakes

I built preference data from both synthetic contrasts and SFT trial-17 outputs. The latter put the model's own weak responses on the rejected side, with better rewrites on the chosen side. This targeted the repetitive openings, weak transitions, and mismatched recommendations I was seeing in conversation.

Some comparisons need several turns to show whether a question leads somewhere useful. Their continuations therefore include scripted user replies. I kept those replies as context but masked them out of the loss so training only scores what the assistant controls. [Data preparation](data.md)

## Diagnosing DPO over-optimisation

Early DPO trials reached perfect preference accuracy while producing broken English, repetition, and occasional Chinese text. The metric compared chosen and rejected branches, so it could improve even when both became less likely in absolute terms.

For example, v1.1 trial 29 reached accuracy 1.0 and margin 11.67 while its chosen log probability fell to about -672. Healthy later candidates were around -230. I treated that combination as a warning of over-optimisation. The exact cause of the language drift was not isolated.

Reducing DPO weight at inference helped some conversations, but it diluted an already damaged adapter. I instead reduced training strength, tested smaller ranks and fewer epochs, and added an RPO supervised term to keep chosen responses likely. I also changed the Optuna objective. The first hybrid score still rated broken trials too highly; v1.5 added stricter checks on absolute likelihood, category performance, length effects, and reward margin.

The resulting score filtered candidates for conversation testing. It did not replace that testing. [DPO experiments](../dpo/README.md)

## Sharing the base during training

Loading separate policy and reference models would use too much memory. Both policies share the same frozen base instead. The reference uses the SFT behavior, while the training policy adds the new DPO adapter. Adapter switching and cached reference log probabilities avoid loading a second full model.

Training uses the BF16 checkpoint loaded in bitsandbytes 8-bit. The FP8 path in the training environment could not train these adapters, while FP8 was the serving format used by the vLLM setup. I kept that split and checked behavior after moving the adapter to the serving base.

## Combining adapters for vLLM

The serving path accepts one LoRA per request. During training the policy uses both SFT and DPO, so I needed a single adapter with the same combined update.

Baking SFT into the base would change the numerical setup against which DPO was trained. Instead, I concatenated the low-rank factors so their product equals the sum of the two updates. For trial 4, two rank-16 adapters become one rank-32 adapter. The export checks the update matrices directly before testing forward drift and generated text. [Adapter export](../dpo/README.md#export-for-serving)

## Selecting trial 4

Trial 16 had the highest v1.5 training score. Trial 4 won both judges' first-round steering scores. In the second round, trial 20 had a slightly higher mean, but the judges disagreed on the final recommendation: one favored trial 4's handling of difficult openings, while the other preferred trial 20.

I then had people talk directly to both models and compare their steering, style, and recommendations. That qualitative test favored trial 4. I deployed it and kept trial 20 as the backup. [Results and judging protocol](evaluation.md)

## Building retrieval within the memory budget

I chose BGE-M3 because one model produces both semantic and keyword vectors. This supports hybrid search without loading a second encoder or a separate reranker. Qdrant keeps the index local, and TensorRT reduces embedding latency.

An indexing bug initially hid the value of that setup. The indexer requested batches of 128 from an engine built for at most two, leaving invalid dense vectors in the database. I rebuilt with batch two, richer product text, and vector checks that stop on invalid output. The McAuley catalog with RRF then performed best in the catalog/fusion comparison. [Retrieval](../rag/README.md)

## Fixing the serving behavior

The first API wrapper forced search too early and imposed a rigid prompt full of repeated phrases. It overrode the probing behavior I had trained. I removed the forced first-turn search and rewrote the persona as guidance rather than a list of lines to repeat.

I compared SFT, half-weight DPO, and full-weight DPO under three prompts. Full-weight trial 4 with the shorter `compressed` prompt was the best fit for deployment. A later logits processor addressed missed search calls with increasing pressure across turns, though its early-search tradeoff still needs tuning.

The services run separately so I can restart or debug the model, embeddings, index, and API independently. Streaming and keep-alive comments make longer replies usable through nginx and the optional ngrok tunnel. [Architecture](architecture.md), [deployment](deployment.md)
