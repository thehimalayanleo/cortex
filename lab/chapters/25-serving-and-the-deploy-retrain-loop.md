---
title: "Lab 25: Serving models and agents, and the deploy-retrain loop"
kind: permanent
topics: [lab]
chapter: 25
station: none
recipe: recipes/sft_lora.py
reading_time: 50 min
---

## What you will be able to do

- Explain what a serving engine does between a request and a token (continuous batching, paged KV cache, prefix caching, speculative decoding) and predict throughput and latency from the model's size and the GPU's bandwidth.
- Serve one base model with many LoRA adapters and say why that is the right shape for continual learning across customers.
- Instrument a served model so that every response is a logged event with the model version, ready for Lab 23's miner.
- Run the loop: serve, collect, mine, train, evaluate, redeploy, and report the metric per round with retention on a fixed suite.
- Describe the failure modes of the loop that only appear in production: drift, feedback loops, and a checkpoint that is better on the suite and worse for users.

## The idea in one paragraph

Serving is where the model meets the people whose behaviour is the training signal, so the serving stack is part of the training stack. A modern engine keeps many requests in flight, allocates the KV cache in pages so memory is not wasted on padding, shares the cache for common prefixes (the system prompt, the customer's documents), and can draft with a small model to cut latency. On top of that, one base model can serve hundreds of LoRA adapters by swapping the low-rank matrices per request, which is what lets each customer's model learn from its own log without a copy of the base per customer. The loop closes when the engine's request log is the miner's input and the trainer's output is the next adapter the engine loads. Every part of that loop exists in Cortex at small scale: the 5090 serves, the Traces tab logs, Lab 21's pipeline trains and evaluates.

## The math

### Where the time goes

Decoding one token for a model with $P$ parameters in $b$ bytes each reads $bP$ bytes of weights; on a GPU with memory bandwidth $B$ that takes at least $bP/B$ seconds. For a 1.5B model in bf16 on a 5090 ($B \approx 1.8$ TB/s): $3\,\text{GB} / 1.8\,\text{TB/s} \approx 1.7$ ms per token per sequence, so a single stream cannot exceed about 600 tokens per second no matter how good the kernels are. Batching $n$ sequences reads the weights once for $n$ tokens, so throughput scales with $n$ until the KV-cache reads or compute dominate; that is why continuous batching (admit new requests into a running batch as others finish) is the single largest throughput lever.

The KV cache per token is $2 \times L \times d_{\text{kv}} \times b$ bytes (keys and values, per layer). Paged allocation (vLLM's PagedAttention) stores it in fixed blocks so a request holds only the blocks it uses and memory fragmentation disappears; the practical effect is two to four times more concurrent sequences on the same GPU. Prefix caching shares blocks for identical prefixes, so a 4k-token system prompt costs its prefill once per hour rather than once per request.

Latency has two parts: time to first token (prefill, compute-bound, proportional to prompt length) and time per output token (decode, bandwidth-bound, proportional to model size divided by batch efficiency). Speculative decoding (Lab 17) attacks the second with a draft model; it helps most when the batch is small.

### LoRA serving

A LoRA adapter replaces $W$ with $W + BA$, $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll d$. The engine keeps the base weights once and the adapters in a pool; for each request it adds $B_i A_i x$ for that request's adapter, batched across requests with different adapters by a grouped matmul. The extra cost per token is $2r(d + k)$ multiply-adds per adapted matrix, under a percent of the base at $r = 16$; the memory per adapter is $r(d+k)b$ per matrix, a few tens of megabytes for a 7B model. This is the reason continual learning across tenants is done with adapters: a new checkpoint per customer per week is a small file that loads in a second, and evaluation and rollback are per adapter.

### The loop as a schedule

Round $k$: serve $\pi_k$ (base plus adapter $k$), log events with `model_version = k`, mine (Lab 23), train adapter $k+1$ from adapter $k$ with replay, evaluate on the frozen suite and on a fresh sample of round-$k$ requests, deploy if the paired comparison wins and retention holds, else keep $\pi_k$ and look at the data. Two numbers per round: the fresh-sample win rate against $\pi_k$ and the retention on the fixed suite. Plot both; the loop is healthy when the first rises and the second is flat.

### Feedback loops and drift

Once $\pi_{k+1}$ is deployed, the requests change: users ask for what the model is now good at and stop asking for what it is bad at, so the log narrows. That is a feedback loop, and it looks like improvement in the log while the product's coverage shrinks. The check is the fixed suite (which does not move) and a small fraction of traffic served by the previous adapter as a control, which gives a paired comparison on the live distribution. Drift in the other direction, the world changing under a frozen model, shows as a falling fresh-sample metric with a flat suite; the answer is the same loop, which is the argument for building it.

### Better on the suite, worse for users

A checkpoint can win the suite and lose the product: the suite over-represents what the miner could verify (short, checkable tasks), and the product cares about long, ambiguous ones. The fix is in the suite's composition, not the model: include a slice of unverifiable tasks graded by a calibrated judge (Lab 09) and a slice of long-horizon episodes, and weight the suite by the product's traffic. Ship on the weighted suite, and always with the control-traffic comparison.

## Build it small

Serve the Lab 04 SFT checkpoint with vLLM on the 5090 (`vllm serve <base> --enable-lora --lora-modules sft=<adapter>`); send 1, 8, 32 concurrent requests of 256 tokens with a small script and record tokens per second and p50/p95 time per token; compare to the bandwidth bound above. Turn on prefix caching with a 2k system prompt and measure time to first token before and after. Log every response to a JSONL with the adapter name.

## Build it real

Close the loop on your own traffic: Cortex chat served by the local engine through the same provider interface, the Traces collector as the log, Lab 23's miner, `sft_lora.py` with replay for the adapter, `eval_suite.py` plus a fresh-sample paired comparison, and a manual deploy switch. Two rounds over two weeks, with 10 percent of requests answered by the previous adapter as the control. The deliverable is one plot: win rate and retention per round.

## How it goes wrong

- Throughput far below the bound: tiny batches; enable continuous batching and check that requests actually overlap.
- Out of memory at moderate concurrency: KV cache not paged, or max sequence length set too high; size the cache.
- Adapter swap adds latency: the pool is too small and adapters are reloaded from disk; pin the hot ones.
- Round three is worse than round one: forgetting; retention plot, replay.
- Log improves, product does not: feedback loop; control traffic.

## Measure it

- Tokens per second and p50/p95 time per token at three concurrency levels, against the bandwidth bound.
- Time to first token with and without prefix caching.
- Adapter count served per GPU and memory per adapter.
- Per round: fresh-sample paired win rate with an interval, retention on the fixed suite, fraction of control traffic.

## Exercises

1. Compute the bandwidth bound for a 7B model in bf16 on the 5090 and on an H100, and measure how close vLLM gets at batch 1 and batch 32.
2. Serve three LoRA adapters and measure the per-token cost of the adapter path.
3. Build the request logger with `model_version` and `shown_position` and feed it to Lab 23's miner.
4. Run one round of the loop with replay at 0, 15 and 30 percent; plot retention.
5. Implement a control-traffic split and the paired comparison it enables.

## Test yourself

1. Why is single-stream decoding bandwidth-bound, and what is the bound for a 1.5B bf16 model on a 1.8 TB/s GPU?
2. What does continuous batching change and why is it the largest throughput lever?
3. What problem does paged KV allocation solve?
4. Why is LoRA the right shape for per-customer continual learning at serving time?
5. What are the two numbers per round that say the loop is healthy?
6. What is the feedback loop in a deployed learning system and what is the control for it?
7. How can a checkpoint win the suite and lose the product?
8. What does prefix caching save, and for which prompts?
9. When does speculative decoding help most?
10. Why does the log need the model version on every line?

<details><summary>Answers</summary>

1. Each token reads all the weights once; $3\,\text{GB}/1.8\,\text{TB/s} \approx 1.7$ ms, about 600 tokens per second per stream.
2. Requests join and leave a running batch, so the weights are read once per step for many sequences; throughput scales with the batch until compute or KV reads dominate.
3. Fragmentation and padding waste in the KV cache; fixed blocks let a request hold only what it uses, two to four times more concurrency.
4. One base, many small adapters swapped per request at under a percent extra compute; a new adapter per customer per round is a small file with per-adapter eval and rollback.
5. The fresh-sample paired win rate against the previous adapter and retention on the fixed suite.
6. The deployed model changes what users ask, so the log narrows while looking like improvement; the control is a fraction of traffic served by the previous adapter plus a fixed suite.
7. The suite over-represents verifiable short tasks; weight it by traffic and include judged and long-horizon slices.
8. The prefill of a shared prefix (system prompt, documents), for prompts that repeat it.
9. At small batch, where decode is bandwidth-bound and the GPU has spare compute for the draft-and-verify step.
10. Attribution: which policy produced the sample, needed for off-policy correction, reference rotation, and the round-by-round accounting.

</details>

## What will change, what will not

Engines will change (kernels, disaggregated prefill and decode, better speculative schemes) and the numbers in this chapter will move with the hardware. The bound will not: decode reads the weights, batching amortises the read, the KV cache is the memory budget, and an adapter per tenant is the shape of continual learning until base models are cheap enough to copy. The loop will not change either: serve, log with the version, mine, train with replay, evaluate paired with a control, deploy or do not.

## Read next

- Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (vLLM, 2023).
- Chen et al., "Punica: Multi-Tenant LoRA Serving" (2023); Sheng et al., "S-LoRA" (2023).
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023).
- Lab 13 for the KV cache arithmetic, Lab 17 for speculative decoding, Lab 23 for the miner, Lab 21 for the runner.
