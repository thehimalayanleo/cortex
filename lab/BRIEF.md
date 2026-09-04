# Lab chapters: authoring brief

You are writing one or more chapters of the Cortex Training Lab, a curriculum that lives inside a researcher's second brain. The reader is Ajinkya: PhD in ECE (sparse recovery, differential privacy), senior ML research scientist, comfortable with math and PyTorch, new to some of these areas. He asked for a pedantic teacher, not a summary. Each chapter must let him go from "I've heard of this" to "I can build it, break it, and evaluate it", with the math written out and the code runnable.

## Where files go
`cortex/lab/chapters/NN-slug.md`, one file per chapter, numbered as assigned. Frontmatter:

```yaml
---
title: "Lab 04: SFT loop design"
kind: permanent
topics: [lab]
chapter: 4
station: posttrain        # id of the in-browser station this pairs with, or none
recipe: recipes/sft_lora.py   # the script that runs the real version on the 5090, or none
reading_time: 35 min
---
```

## Required structure (use these headings, in this order)
1. **What you will be able to do** (3 to 5 concrete outcomes)
2. **The idea in one paragraph** (no jargon; a colleague could repeat it)
3. **The math** (every symbol defined; derive, do not assert; LaTeX with `$...$` and `$$...$$`)
4. **Build it small** (a complete, runnable PyTorch or plain-Python snippet under 80 lines that demonstrates the core mechanism on toy data; state the expected output)
5. **Build it real** (the 5090 recipe: what data, what model, what library, what to watch in the logs, how long it takes on one RTX 5090 with 32 GB; point at `recipes/<name>.py` and describe its arguments; do not paste a 400-line script)
6. **How it goes wrong** (5 to 8 failure modes with the symptom, the cause, and the fix)
7. **Measure it** (what to evaluate, which metric, what number is good and why)
8. **Exercises** (4 to 6, increasing difficulty, each with a short answer or a check)
9. **Read next** (5 to 8 references: paper title, first author, year, and one line on why it matters; only real, well-known works; if unsure a reference exists, leave it out)

## Voice and rules
- Teacher voice. Define before use. Show the failure before the fix. Say "you" to the reader.
- Plain prose. No em dashes or en dashes anywhere (use commas, periods, colons, or parentheses). No exclamation marks. No emoji. No bold-label bullet lists. No "let's dive in" or "in this chapter we will". No marketing words.
- Never invent numbers, benchmarks, or citations. Where a number depends on hardware or data, give the formula and a worked example with stated assumptions instead of a fake measurement.
- Code must be correct and self-contained. Prefer PyTorch. Keep snippets short; the chapter is not a repository.
- Where the chapter pairs with a browser station (data, pretrain, midtrain, posttrain, encoder, cluster), reference what the reader will see there in one or two sentences.
- Length: 2,500 to 4,500 words. Depth over breadth. If you must choose, cut a section rather than thin all of them.
- Cross-reference other chapters by number ("see Lab 06") where the concepts connect.

## Shared facts to keep consistent
- The in-browser lab trains a 2-layer, width-48, 3-head character-level transformer with tf.js; stations: data, pretrain, midtrain (mixture + cooldown), posttrain (SFT with answer-only loss, then DPO with a frozen reference), encoder (MLM then InfoNCE), cluster (k-means).
- The real runs execute on Ajinkya's RTX 5090 (32 GB, Blackwell, CUDA 12.8) reachable by SSH, via scripts under `cortex/lab/recipes/`. Libraries available there: PyTorch, Unsloth, TRL, transformers, sentence-transformers, vLLM, lm-eval-harness, Triton.
- Marin (Stanford, marin-community/marin) is the reference open pipeline: JAX/Levanter, steps as a dependency graph, `train_lm(model=llama_nano, datasets={...: weight}, optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1), num_train_steps=...)`.
- The NVIDIA Nemotron agentic and tool-use collection on Hugging Face: datasets `nvidia/Nemotron-SFT-Agentic-v2`, `nvidia/Nemotron-Agentic-v1`, `nvidia/Nemotron-RL-agent-calendar_scheduling`, `nvidia/Nemotron-RL-agent-workplace_assistant`, `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`, `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1`, `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`, `nvidia/Nemotron-Terminal-Corpus` (366k samples), `nvidia/Nemotron-Terminal-Synthetic-Tasks`, `nvidia/Nemotron-SFT-ARC-AGI-v1` (122k), `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1`. Describe their roles from their names and what such datasets contain in general; do not invent row counts, fields, or license terms beyond those listed.
- nomic-embed-text-v1.5: BERT-base shape (12 layers, 768 hidden, 12 heads, 137M params), RoPE, SwiGLU, MLM pretraining at 2048 tokens with 30% masking, contrastive pretraining on ~235M pairs then supervised fine-tuning with hard negatives, task prefixes, mean pooling, Matryoshka dims 64 to 768, 8192 context via RoPE scaling.
