# Cortex Training Lab: recipes

Runnable Python scripts, one per lab chapter, that do the real version of what the
in-browser stations show. Every recipe has a `--smoke` mode that runs on a CPU in
about a minute with no network access on the synthetic corpus from `common.py`
(the same stories, arithmetic strings, Q/A pairs and topic sentences the browser
lab uses), and a real mode for the RTX 5090 (or a Modal GPU through `modal_app.py`).
The smoke path runs the actual algorithm at toy scale; it never skips the loss.

Shared flags on every recipe: `--out DIR` (default `./out/<recipe>`), `--steps N`,
`--seed`, `--device` (auto: cuda if available), `--smoke`, `--max-samples` (cap on
rows pulled from the Hugging Face hub; hub datasets are only touched outside
`--smoke`). Missing optional libraries exit with the `pip install` line to run.

Run everything from the `cortex/` directory. `uv run --python 3.11 --with torch --with numpy python lab/recipes/<name>.py --smoke`
is the zero-setup way to try a smoke run. The 5090 commands below assume the
libraries listed in `lab/BRIEF.md` are installed there.

| recipe | what it teaches | chapter | smoke | real (5090) | Modal |
|---|---|---|---|---|---|
| `pretrain_nano.py` | the pretraining loop on a minimal GPT (RMSNorm, RoPE, SwiGLU); AdamW, warmup+cosine or WSD, clipping, bf16; tokens/s and FLOPs/token; `--loop T` recurrent depth | 02, 16 | `python lab/recipes/pretrain_nano.py --smoke --steps 200` | `python lab/recipes/pretrain_nano.py --steps 2000 --n-layer 6 --d-model 384 --n-head 6 --seq-len 256 --batch 32` | `modal run lab/recipes/modal_app.py --recipe pretrain_nano --args "--steps 2000"` |
| `midtrain.py` | continue a checkpoint on a two-domain mixture with a linear cooldown; per-domain held-out loss shows forgetting | 03 | `python lab/recipes/midtrain.py --smoke --steps 200 --mix a=0.7,b=0.3` | `python lab/recipes/midtrain.py --ckpt out/pretrain_nano/ckpt.pt --mix a=0.7,b=0.3 --steps 1000 --cooldown-frac 0.3` | `modal run lab/recipes/modal_app.py --recipe midtrain --args "--ckpt /vol/pretrain_nano/ckpt.pt --steps 1000"` |
| `sft_lora.py` | answer-only loss mask (fraction of supervised tokens is printed), LoRA, TRL SFTTrainer with Unsloth or peft | 04 | `python lab/recipes/sft_lora.py --smoke --steps 300` | `python lab/recipes/sft_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset nvidia/Nemotron-SFT-Agentic-v2 --max-samples 5000 --steps 300` | `modal run lab/recipes/modal_app.py --recipe sft_lora --args "--max-samples 5000 --steps 300"` |
| `dpo.py` | the DPO loss by hand with a frozen reference (implicit rewards, margin, accuracy), then TRL DPOTrainer | 05 | `python lab/recipes/dpo.py --smoke --steps 100 --beta 0.1` | `python lab/recipes/dpo.py --model Qwen/Qwen2.5-0.5B-Instruct --dataset trl-lib/ultrafeedback_binarized --max-samples 2000 --steps 200` | `modal run lab/recipes/modal_app.py --recipe dpo --args "--max-samples 2000 --steps 200"` |
| `grpo_tool.py` | GRPO for tool calls: strict `<call>{...}</call>` format, shaped verifiable reward, group-normalized advantages, clipped ratio, k3 KL; TRL GRPOTrainer in real mode | 05, 06 | `python lab/recipes/grpo_tool.py --smoke --steps 60` | `python lab/recipes/grpo_tool.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200 --group 8` | `modal run lab/recipes/modal_app.py --recipe grpo_tool --args "--steps 200 --group 8"` |
| `paint_grpo.py` | GRPO with an image reward: a drawing DSL, a numpy renderer, gate + length + similarity reward; the small version of "train to paint with code" | 15 | `python lab/recipes/paint_grpo.py --smoke --steps 60` | `python lab/recipes/paint_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct --steps 200 --group 8` | `modal run lab/recipes/modal_app.py --recipe paint_grpo --args "--steps 200 --group 8"` |
| `embed_contrastive.py` | InfoNCE with in-batch negatives on a bidirectional encoder with mean pooling; recall@1 on held-out pairs; sentence-transformers + Matryoshka in real mode | 07 | `python lab/recipes/embed_contrastive.py --smoke --steps 200` | `python lab/recipes/embed_contrastive.py --pairs-jsonl out/embed_vault/pairs.jsonl --model nomic-ai/nomic-embed-text-v1.5 --matryoshka 768,512,256,128,64` | `modal run lab/recipes/modal_app.py --recipe embed_contrastive --args "--pairs-jsonl /vol/embed_vault/pairs.jsonl"` |
| `embed_vault.py` | chunk and embed the vault (notes + library), write `embeddings.npy`, `chunks.jsonl`, and `{query, positive}` pairs; nearest-neighbour sanity check | 07, 08 | `python lab/recipes/embed_vault.py --smoke` | `python lab/recipes/embed_vault.py --vault ~/Cortex --out out/embed_vault` | `modal run lab/recipes/modal_app.py --recipe embed_vault --args "--vault /vol/Cortex"` (upload the vault to the volume first) |
| `eval_suite.py` | lm-eval-harness tasks plus a generative exact-match eval with a bootstrap 95% CI; LLM-judge hook is a stub | 09 | `python lab/recipes/eval_suite.py --smoke --steps 300` | `python lab/recipes/eval_suite.py --model Qwen/Qwen2.5-0.5B-Instruct --tasks gsm8k --limit 200 --custom-jsonl my_eval.jsonl --chat` | `modal run lab/recipes/modal_app.py --recipe eval_suite --args "--model Qwen/Qwen2.5-0.5B-Instruct --tasks gsm8k --limit 200"` |
| `redteam_suite.py` | indirect prompt injection ASR with a Wilson interval; regex judges; `--train` exports SFT pairs from the failures | 10 | `python lab/recipes/redteam_suite.py --smoke --train` | `python lab/recipes/redteam_suite.py --model Qwen/Qwen2.5-0.5B-Instruct --train` | `modal run lab/recipes/modal_app.py --recipe redteam_suite --args "--model Qwen/Qwen2.5-0.5B-Instruct --train"` |
| `kernel_bench.py` | naive attention vs SDPA backends, exact attention FLOPs and TFLOP/s, KV-cache bytes per token, Triton fused softmax vs torch | 11, 13 | `python lab/recipes/kernel_bench.py --smoke` | `python lab/recipes/kernel_bench.py --seqs 512,1024,2048,4096,8192 --heads 16 --head-dim 128` | `modal run lab/recipes/modal_app.py --recipe kernel_bench --args "--seqs 1024,4096,8192"` |
| `optim_bench.py` | AdamW vs Muon (Newton-Schulz orthogonalization for 2-D weights) on identical data and schedule; METRIC lines carry `opt` | 12 | `python lab/recipes/optim_bench.py --smoke --steps 100` | `python lab/recipes/optim_bench.py --steps 1500 --n-layer 6 --d-model 384 --n-head 6 --seq-len 256 --batch 32` | `modal run lab/recipes/modal_app.py --recipe optim_bench --args "--steps 1500"` |
| `moe_nano.py` | top-k MoE MLP in the minimal GPT, Switch balance loss, router z-loss, per-expert load and a domain x expert usage matrix; active vs total params | 11 | `python lab/recipes/moe_nano.py --smoke --steps 300` | `python lab/recipes/moe_nano.py --steps 2000 --experts 8 --top-k 2` | `modal run lab/recipes/modal_app.py --recipe moe_nano --args "--steps 2000 --experts 8 --top-k 2"` |
| `lean_eval.py` | a model writes Lean 4 proofs, `lake env lean` grades them; pass@1 with a Wilson interval | 14 | `python lab/recipes/lean_eval.py --smoke` | `python lab/recipes/lean_eval.py --lean-project ~/github-repos/sfp/lean-sfp --theorems my_theorems.jsonl` | not on Modal (needs a Lean toolchain in the image) |
| `spec_decode.py` | speculative decoding with the exact accept/reject rule; acceptance rate, tokens per target forward, tok/s, and a total-variation exactness check | 17 | `python lab/recipes/spec_decode.py --smoke --steps 300 --k 4` | `python lab/recipes/spec_decode.py --target Qwen/Qwen2.5-1.5B-Instruct --draft Qwen/Qwen2.5-0.5B-Instruct --k 5` | `modal run lab/recipes/modal_app.py --recipe spec_decode --args "--k 5"` |
| `inspect_model.py` | architecture poking: module tree with shapes and per-block counts, parameter split (embedding / attention / MLP / norms / head, tied or not), derived config (layers, d_model, heads, kv heads, head_dim, vocab, MLP width), FLOPs per token (2N + attention term) and KV bytes per token; then a probe string through forward hooks: residual-stream norm per layer, attention entropy per head, and the logit lens (final norm + unembedding on every layer's residual). Writes `module_table.json`, `per_layer.json`, `attention.json`, `summary.json` for the UI | 11, 13 | `python lab/recipes/inspect_model.py --smoke` (trains the minimal GPT for `--steps`) or `--model out/pretrain_nano/ckpt.pt --text "the cat sat on the"` | `python lab/recipes/inspect_model.py --model Qwen/Qwen2.5-0.5B --text "The capital of France is" --layer 12 --n-heads 8` | `modal run lab/recipes/modal_app.py --recipe inspect_model --args "--model Qwen/Qwen2.5-0.5B"` |
| `modal_app.py` | run any recipe above on a Modal GPU, stream stdout, keep `--out` in the `cortex-lab-out` volume | all | n/a | n/a | `modal run lab/recipes/modal_app.py --recipe <name> --args "..." --gpu H100` |

`common.py` is the shared module: the stdout protocol (`metric`, `status`, `result`, `rollout`, `clip_text`), seeding, device choice, the
synthetic corpus, the character tokenizer, the minimal GPT (with a KV cache for
decoding, left-padded batched generation, an optional recurrent-depth loop and a
bidirectional mode), the LR schedules, checkpoint helpers, and the bootstrap and
Wilson interval functions.

Chapter numbers refer to `lab/chapters/NN-*.md`. Lab 05's text mentions
`recipes/grpo.py`; the GRPO recipe here is `grpo_tool.py` (Lab 06 pairs with it directly).

## The METRIC protocol

The Cortex server parses stdout. Four line types matter; everything else is free-form log text.

```
STATUS {"phase": "train", "msg": "2000 steps, batch 32 x 256 tokens, schedule cosine"}
METRIC {"step": 12, "loss": 2.41, "lr": 0.0006, "tokens_per_s": 91234.5}
METRIC {"step": 100, "loss": 1.98, "val_loss": 2.05}
RESULT {"train_loss": 1.62, "val_loss": 1.71, "checkpoint": "out/pretrain_nano/ckpt.pt"}
```

- `METRIC ` + one JSON object per line, always with an integer `step`; the other
  fields are numbers (a few recipes add one string tag such as `opt` or `impl` so
  several curves can share a chart). NaN and inf are written as `null`.
- `STATUS ` + `{"phase", "msg"}` marks phase changes (data, train, eval, done).
- `RESULT ` + one JSON object, printed exactly once at the end, with the final
  numbers and the paths of any artifacts written under `--out`.
- Every one of these lines is flushed immediately (`print(..., flush=True)`), so
  they stream through SSH and through `modal run` as they are produced.

Helpers: `common.metric(step, **fields)`, `common.status(phase, msg)`, `common.result(**fields)`.

### ROLLOUT lines

```
ROLLOUT {"step": 5, "group": 0, "idx": 2, "prompt": "q: what is 3 + 5\n", "completion": "<call>{\"name\":\"add\",\"args\":{\"a\":3,\"b\":5}}</call>", "reward": 1.0, "advantage": 0.87, "parse": 1, "tool": 1, "answer": 1, "kl": 0.004, "expected": "..."}
```

- `ROLLOUT ` + one JSON object per line: what the model actually produced and the
  numbers that scored it. Unlike METRIC, string fields are expected (prompt,
  completion, ...); they are truncated at the source (`common.clip_text`: prompts to
  300 characters, completions to 600). The server appends them to `rollouts.jsonl`.
- `grpo_tool.py`, `paint_grpo.py`: every `--log-rollouts-every N` steps (default 5,
  plus step 1 so a short run shows something) one line per sample of the first group
  of that step, with `step, group, idx, prompt, completion, reward, advantage, kl`
  and the reward components (`parse, tool, answer` and `expected` for the tool task;
  `gate, length, similarity, n_commands` for painting). In real mode the lines come
  from inside the TRL reward function, so `step` counts reward calls (one per
  generation step) and `advantage` is the group normalization computed for display.
- `dpo.py`: every `--log-rollouts-every` steps (default 10; trainer logs in real
  mode) `--log-rollouts-n` pairs (default 3) with `step, idx, prompt, chosen,
  rejected, chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp,
  chosen_reward, rejected_reward, margin` (the implicit rewards are
  `beta * (logp - ref_logp)`; real mode scores them under the policy and the
  adapter-disabled reference).
- `spec_decode.py`: the first `--log-passes` verify passes (default 6) with `step`
  (pass index), `prompt`, `context_tail`, `draft`, `draft_ids`, `proposed`, `accepted`,
  `accepted_text`, `corrected_token`, `corrected_id`, `correction` (`residual` when a draft
  token was rejected and the replacement came from max(0, p - q); `bonus` when all k
  were accepted and the extra token came from p_k) and `emitted`.
- `0` for any of these flags disables the lines. Helper: `common.rollout(**fields)`.
