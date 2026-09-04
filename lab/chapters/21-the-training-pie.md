---
title: "Lab 21: The training pie: one pipeline from your data to a small reasoning model"
kind: permanent
topics: [lab]
chapter: 21
station: none
recipe: recipes/data_prep.py
reading_time: 60 min
---

## What you will be able to do

- Write a training pipeline as a dependency graph of stages, where each stage is one recipe run and the only thing that passes between stages is a path printed in a RESULT line, and say why that is enough.
- Build a corpus you can draw as a pie: tokens per source, with your own Traces as one slice and a synthetic reasoning set with verifiable answers as another, and choose the mixture on purpose.
- Run the six stages of the reasoning-nano template (data, pretrain, midtrain, SFT, RL, eval) on a CPU in minutes, read each stage's metrics, and explain what each number can and cannot tell you at that scale.
- Trace one example through the whole pie: the line format that pretraining sees, the mask SFT applies to it, the span the RL reward checks, and the span the evaluator extracts, and show they are the same span.
- Scale each stage to one RTX 5090 with the formulas for parameters, tokens, FLOPs and memory, using the throughput your own run prints rather than a number from a table.

## The idea in one paragraph

Every earlier chapter trained one stage in isolation, and that is how you learn a stage, but a model is what comes out of the stages composed. The composition is not complicated: a pipeline is a directed acyclic graph whose nodes are recipe runs and whose edges carry file paths, so the pretraining run prints `checkpoint: out/.../ckpt.pt` and the mid-training run receives that string as `--ckpt`. The runner's whole job is to start a stage when every stage it depends on has finished, to write down which run belongs to which stage so a restart loses nothing, and to let you retry a stage that failed without redoing the ones that did not. What makes the composition interesting is the data: a corpus is a mixture of sources, each with a token count, and the mixture is a decision you make once for pretraining and again for mid-training. In this chapter the sources are your own collector (the Traces tab: pairs you wrote, preferences you recorded, notes) and a synthetic reasoning set whose answers a program can check, which is what lets the last two stages, RL and eval, run without a judge. The Lab's Pipeline tab draws the graph as cards and the corpus as a pie, and the chat has the same buttons.

## The math

### A pipeline as a partial order

Let the stages be $s_1, \dots, s_n$ and let $D(s_i) \subseteq \{s_1, \dots, s_n\}$ be the dependencies of $s_i$. The graph is acyclic when there is a topological order, an assignment of ranks $r(s) \in \{0, 1, 2, \dots\}$ with $r(s_i) > r(d)$ for every $d \in D(s_i)$. The runner needs nothing more than the ready set

$$R_t = \{ s : \text{status}(s) = \text{pending} \text{ and } \forall d \in D(s),\ \text{status}(d) = \text{done} \},$$

recomputed at every tick $t$. Starting every member of $R_t$ is correct because a stage enters $R_t$ only after its inputs exist, and it is complete because a pending stage whose dependencies are all done will be in $R_t$ at the next tick. The reasoning-nano template is a chain, so $|R_t| \le 1$ and the GPU never sees two stages at once; the same rule would run a fan-out (several SFT variants from one mid-training checkpoint) without any change to the runner.

Resumability follows from what is stored. The state of a pipeline is the tuple (status, run id) per stage, written to disk before the run's process is started. If the server restarts, the runs module marks every run that was in flight as failed (it can no longer see the process), the next tick maps that run status onto its stage, and the stage becomes retryable. A retry resets the failed stage and its downstream closure

$$C(s) = \{s\} \cup \{ u : D(u) \cap C(s) \ne \emptyset \}$$

to pending, because every stage that consumed the failed stage's artifact must be recomputed. Nothing upstream is touched, which is the point of making artifacts the only channel between stages.

### Artifacts as the interface

Each recipe prints one `RESULT {...}` line, and a stage's arguments may contain placeholders `{stage:field}` that the runner substitutes from that JSON right before starting the stage. For the reasoning-nano template the argument strings are, verbatim from the template:

```
data:     data_prep       --smoke --out {out}/data --traces-jsonl {traces}
pretrain: pretrain_nano   --smoke --corpus {data:corpus} --out {out}/pretrain --steps 300 --seq-len 256
midtrain: midtrain        --smoke --ckpt {pretrain:checkpoint} --text-a {data:corpus} --text-b {data:reason_text}
                          --mix a=0.4,b=0.6 --steps 150 --out {out}/midtrain
sft:      sft_lora        --ckpt {midtrain:checkpoint} --pairs-jsonl {data:sft} --steps 300 --max-new 64 --out {out}/sft
rl:       grpo_reason     --ckpt {sft:checkpoint} --tasks-jsonl {data:reason_train} --eval-jsonl {data:reason_heldout}
                          --steps 30 --group 4 --batch 4 --out {out}/rl
eval:     eval_suite      --ckpt {rl:checkpoint} --baseline-ckpt {midtrain:checkpoint}
                          --custom-jsonl {data:reason_heldout} --max-new 64 --out {out}/eval
```

`{out}` is the pipeline's folder and `{traces}` is the collector's export, written by the runner right before the data stage (and copied to the GPU box when the executor is ssh, because the vault lives on your laptop). The paths are relative to the working directory of the executor, so a pipeline that runs on the box keeps every artifact on the box. Note what is not passed: no hyperparameters flow between stages, no tensors, no Python objects. A checkpoint carries its own architecture and tokenizer (`common.save_checkpoint` stores both), so `--ckpt` is the entire contract.

### The corpus as a mixture

The data stage builds a set of sources $k = 1, \dots, K$, each a list of documents, with token counts $n_k$ under the tokenizer the pipeline will use (characters in smoke, GPT-2 BPE in real mode). The pie is $p_k = n_k / \sum_j n_j$. Pretraining samples windows uniformly from the concatenation, so the fraction of gradient signal that source $k$ receives is $p_k$ in expectation; the counts are the mixture, there is no separate weight. In the smoke run the synthetic reasoning lines dominate the pie (they are numerous and long relative to the ten stories), which is why the pretraining sample at step 200 already emits `answer:` fragments. That is a property of the counts, not of anything the model was told.

Mid-training changes the mixture explicitly. With domain $a$ the whole corpus and domain $b$ the reasoning lines only, the template's `--mix a=0.4,b=0.6` makes each sequence in a batch a Bernoulli(0.6) draw from $b$, and the loss becomes $0.4\,L_a + 0.6\,L_b$ in expectation. Lab 03 derived what to expect: $L_b$ falls, $L_a$ can rise (that is the `forgetting_a` field), and the linear cooldown over the last 30 percent of steps removes the noise ball on both. The trade is visible in one number pair, `val_a_after` against `val_a_before`.

### One span, four stages

Every reasoning item is written on two lines,

```
q: <question>
a: <chain> answer: <gold>
```

and the four later stages agree on which characters matter. Pretraining sees the whole thing as text. SFT builds `prompt = "q: <question>\na: "` and `answer = "<chain> answer: <gold><eos>"`, labels the prompt positions $-100$, and reports the fraction of targets that carry a gradient (the `supervised_frac` field, about half in the smoke run because prompts and answers have similar lengths). The RL reward is a program,

$$r(y) = 0.2\,[\text{"answer:" occurs in } y] + 0.8\,[\text{norm}(\text{tail}(y)) = \text{norm}(g)],$$

where $\text{tail}(y)$ is the text after the last `answer:` up to the end of its line and $\text{norm}$ lowercases and collapses whitespace. The evaluator extracts the same tail and compares it to the gold with SQuAD-style normalization, so a policy that raises reward raises exact match by construction; there is no gap between what RL optimizes and what eval measures except the 0.2 format bonus, which eval ignores.

The GRPO objective is Lab 05's. For a prompt with $G$ sampled completions and rewards $r_1, \dots, r_G$, the advantages are $A_i = (r_i - \bar r) / (\sigma_r + \epsilon)$, the per-token loss is the clipped ratio term plus $\beta$ times the k3 estimator of the KL to the SFT checkpoint, and a group whose rewards are all equal contributes nothing. That last fact is why the RL stage must follow SFT: a policy that has never emitted `answer:` scores 0 on every sample of every group, every $\sigma_r$ is 0, and the gradient is 0. The `zero_var_groups` metric is the fraction of groups in that state; when it is 1.0 the stage is doing nothing.

### Why a verifiable set, and what it cannot teach

Three item kinds are generated with a seeded random number generator: two-step arithmetic (`what is 12 + 7 - 3`, chain `12 + 7 = 19. 19 - 3 = 16.`), transitive comparison (`ann is taller than bob. bob is taller than cat. who is the tallest`), and a rule with either modus ponens (answer `yes`) or modus tollens (answer `no`). Each has a gold answer that is a function of the question, so the reward needs no model and the held-out split is exact: an item is held out by its position after a seeded shuffle and appears in none of `corpus.txt`, `reason.txt`, or `sft.jsonl`. The limit is equally exact: the model can learn these three functions and nothing else, and a high exact match on them is evidence about the pipeline, not about reasoning in general. Your Traces are the slice that is not a function of anything; they are the reason the pipeline exists and, at this scale, the slice the model learns least from, because they are few.

## Build it small

The runner in eighty lines: stages are functions that take the artifacts of their dependencies and return their own, the state machine is the ready-set rule, and a failure is retried by resetting the downstream closure. The "recipes" here are the reasoning generator and its verifier, so the last line prints an exact match you can check by hand.

```python
import random, re

def make_items(n, seed):
    rng, out = random.Random(seed), []
    for i in range(n):
        a, b, c = rng.randint(1, 20), rng.randint(1, 20), rng.randint(1, 9)
        out.append({"q": f"what is {a} + {b} - {c}", "chain": f"{a} + {b} = {a+b}. {a+b} - {c} = {a+b-c}.", "gold": str(a + b - c)})
    return out

def extract(text):                       # the span every stage agrees on
    return text.rsplit("answer:", 1)[1].strip().splitlines()[0].strip() if "answer:" in text else None

def reward(text, gold):
    ans = extract(text)
    return 0.2 * (ans is not None) + 0.8 * (ans is not None and ans == gold)

# --- stages: each returns the artifacts the next ones read ---
def data(_):
    items = make_items(40, 0); return {"train": items[:30], "heldout": items[30:]}
def train(art):                          # a "model" that memorizes chains: a dict
    return {"ckpt": {it["q"]: f"{it['chain']} answer: {it['gold']}" for it in art["data"]["train"]}}
def rl(art):                             # RL adds one item the policy got wrong, to show the reward moving
    ck = dict(art["train"]["ckpt"]); before = sum(reward(ck.get(i["q"], ""), i["gold"]) for i in art["data"]["heldout"])
    ck[art["data"]["heldout"][0]["q"]] = "answer: " + art["data"]["heldout"][0]["gold"]
    after = sum(reward(ck.get(i["q"], ""), i["gold"]) for i in art["data"]["heldout"])
    return {"ckpt": ck, "reward_before": before, "reward_after": after}
def evaluate(art):
    ck, held = art["rl"]["ckpt"], art["data"]["heldout"]
    em = sum(extract(ck.get(i["q"], "")) == i["gold"] for i in held) / len(held)
    return {"exact_match": em, "n": len(held)}

STAGES = {"data": ([], data), "train": (["data"], train), "rl": (["train", "data"], rl), "eval": (["rl", "data"], evaluate)}

def run(stages, fail_once=()):
    status = {s: "pending" for s in stages}; arts = {}; failed = set()
    while any(v != "done" for v in status.values()):
        ready = [s for s, (deps, _) in stages.items() if status[s] == "pending" and all(status[d] == "done" for d in deps)]
        if not ready:
            raise RuntimeError("stuck: " + str(status))
        for s in ready:
            if s in fail_once and s not in failed:
                failed.add(s); status[s] = "failed"; print(f"{s}: failed, retrying downstream closure")
                closure = {s} | {u for u, (deps, _) in stages.items() if s in deps}
                for u in closure: status[u] = "pending"
                continue
            arts[s] = stages[s][1](arts); status[s] = "done"; print(f"{s}: done ->", {k: v for k, v in arts[s].items() if not isinstance(v, (dict, list))})
    return arts

out = run(STAGES, fail_once=("rl",))
print("exact match on held-out:", out["eval"]["exact_match"], "of", out["eval"]["n"])
```

Expected output: `data`, `train` finish; `rl` fails once and is reset with `eval` (its closure); on the next pass `rl` reports `reward_before` 0.0 and `reward_after` 1.0 (one held-out item now answered, worth 0.2 + 0.8); `eval` prints exact match `0.1` on 10 held-out items. Change `fail_once` to `("data",)` and watch every stage reset, since everything depends on `data`.

## Build it real

The real thing is `server/pipeline.py` with the recipes in `lab/recipes/`. Open the Lab's Pipeline tab, pick `reasoning-nano`, leave smoke on, press Start; or ask the chat to `start_pipeline` with template `reasoning-nano`. Six runs appear one after another in the GPU runs tab, each linked from its stage card, and the pie fills in as soon as the data stage prints its RESULT.

Stage by stage, what each one does and what to read:

1. data (`data_prep.py`). Reads the collector's export (`--traces-jsonl`, written by the runner from `traces.export("all")`): records with `prompt` and `response` become SFT rows, `chosen` and `rejected` become DPO rows, and any `content` of at least 20 characters becomes a pretraining document. Generates `--n-reason` synthetic items (600 in smoke, 6000 by default in real mode), shuffles, holds out `--heldout-frac` (0.15) of them. Writes `corpus.txt`, `reason.txt`, `sft.jsonl`, `dpo.jsonl`, `reason_train.jsonl`, `reason_heldout.jsonl` and prints `sources: {stories, topics, arithmetic, reasoning, traces}` with a token count each. Read: the pie. If `traces` is a sliver, the model will not learn much of you yet; that is a statement about the collector, not the pipeline.

2. pretrain (`pretrain_nano.py --corpus`). The minimal GPT from Lab 02 on `corpus.txt`; in smoke a character vocabulary built from the corpus (so any character your traces contain gets an id), 2 layers, width 64, sequence 256 (the template raises the smoke default because a logic item and its chain need about 230 characters). Read: `val_loss` at the end and the `sample@` lines, which show whether the two-line format has been absorbed.

3. midtrain (`midtrain.py --ckpt --text-a --text-b`). Continues that checkpoint on the 0.4/0.6 corpus/reasoning mixture with a WSD cooldown over the last 30 percent. Read: `val_b_after` against `val_b_before` (the gain on reasoning text) and `forgetting_a` (the price on the whole corpus). Both are per-domain held-out losses, as Lab 03 insists.

4. sft (`sft_lora.py --ckpt --pairs-jsonl`). A lab checkpoint selects the nano path even without `--smoke`; the pairs file replaces the built-in Q/A; 10 percent of pairs are held out. Answer-only loss, minibatches of `--batch` once the pair set exceeds one batch, then greedy decoding with `--max-new 64` and exact match on the `answer:` span. Read: `supervised_frac` first (the mask is working when it is well below 1), `exact_match_heldout` second. A warning line counts examples longer than the context; if it is nonzero, the pretraining `--seq-len` is too short.

5. rl (`grpo_reason.py --ckpt --tasks-jsonl --eval-jsonl`). GRPO on the training items with the verifiable reward, greedy evaluation on the held-out items every 10 steps. The recipe clamps `--max-new` so prompt plus completion fits the context, because the policy-gradient pass scores the whole sequence. Read: `reward_greedy_before` and `reward_greedy_after`, then `after_format` against `after_correct`: the format bit is easy and moves first, correctness is the one that matters. Read `zero_var_groups` too; near 1.0 means nothing is being learned.

6. eval (`eval_suite.py --ckpt --baseline-ckpt --custom-jsonl`). Exact match with a bootstrap 95 percent interval on the held-out items for the RL checkpoint, and the same for the mid-training checkpoint as a baseline, reported as `custom_exact_match`, `baseline_exact_match`, `delta_exact_match`. This is the pipeline's one number, and the Pipeline tab shows it with both intervals.

The smoke run is a plumbing test: a 100 thousand parameter character model trained for a few hundred steps scores a few percent exact match with an interval that reaches zero. It proves that every path resolves, every stage loads the previous checkpoint, and the report is computed from the right files. Do not read capability into it.

### Scaling each stage on one RTX 5090

Uncheck smoke, or ask for `smoke: false`, and the template switches to its real argument strings, which you should treat as a starting point to edit in `pipelines/<id>/pipeline.json` before pressing Start. Everything below is arithmetic from the recipe's own formulas; I have not measured these runs on the 5090, and the throughput term is the one your run prints.

The data stage in real mode counts GPT-2 tokens, generates 20,000 reasoning items, and (only outside smoke) adds `--hf-dataset roneneldan/TinyStories --max-samples 20000` as part of the stories source, so that the pie is not almost entirely synthetic reasoning. The token counts it prints are the mixture; if you want reasoning at a third of the pie, change `--n-reason` or `--max-samples` until the printed counts say so.

Pretraining at `--n-layer 6 --d-model 384 --n-head 6 --seq-len 512 --batch 32 --steps 3000` with the GPT-2 vocabulary. Per layer the model has $4d^2$ attention parameters and $3 d h$ MLP parameters with the SwiGLU hidden width $h = \lceil 8d/3 \rceil$ rounded up to a multiple of 8, so $h = 1024$ for $d = 384$: $4 \cdot 384^2 + 3 \cdot 384 \cdot 1024 = 589{,}824 + 1{,}179{,}648$, about $1.77$ million per layer, $10.6$ million non-embedding over six layers, plus a tied embedding of $50{,}257 \times 384 \approx 19.3$ million. Tokens per step are $32 \times 512 = 16{,}384$, so 3000 steps see about $49$ million tokens, roughly $4.6$ tokens per non-embedding parameter. The recipe's training FLOPs per token are $6N + 12\,L\,d\,S = 6 \cdot 10.6\text{M} + 12 \cdot 6 \cdot 384 \cdot 512 \approx 7.8 \times 10^7$, so the run is about $3.8 \times 10^{15}$ FLOPs. Wall time is that divided by the `tflops` field the METRIC line prints for your card; the line is there so you do not have to guess. Memory is not the constraint at this size: weights, gradients and two Adam moments in fp32 are $16$ bytes per parameter, about $0.5$ GB, and the activations for a $32 \times 512$ batch of a 6-layer, 384-wide model in bf16 autocast are well under the 32 GB the card has. Raise `--batch` or `--seq-len` first when the printed `tokens_per_s` says the card is idle.

Mid-training at 1000 steps with the same batch shape sees a third as many tokens as pretraining, all at the reasoning-heavy mixture with a 30 percent cooldown. The choice to make is `--mix`; the numbers to watch are the two per-domain losses.

SFT at `--steps 2000 --batch 64` sees a few thousand pairs many times over, so `exact_match_train` reaches one before `exact_match_heldout` moves (Lab 04's overfitting signature); stop earlier, or add pairs. RL at `--steps 300 --group 8 --batch 16 --max-new 96` samples $16 \times 8 = 128$ completions per step and scores them with the program; generation through `common.generate` dominates, so this stage is slower per step than the training stages by a large factor. Eval is one greedy pass for two checkpoints.

The same `data_prep` outputs feed an instruct-model version of the last three stages (`sft_lora --model`, `grpo_reason --model`, `eval_suite --model ... --chat`), which this chapter does not run. The `embed-mine` template is the encoder version of the same idea in two stages: `embed_vault` writes `pairs.jsonl`, `embed_contrastive --pairs-jsonl` fine-tunes with InfoNCE (Lab 07) and reports recall at 1 on held-out pairs.

## How it goes wrong

1. Every group has zero variance in the RL stage. Symptom: `zero_var_groups` near 1.0, `reward_mean` flat, `kl` at 0. Cause: the SFT checkpoint does not emit the format, so every completion scores 0 (or every one scores exactly 0.2). Fix: more SFT steps, a longer context so the answer span is not truncated, or a smaller `--temperature` so the format survives sampling.

2. A stage fails on a placeholder. Symptom: the stage's error reads `stage 'data' printed no 'corpus' in its RESULT line`. Cause: the upstream recipe exited before its RESULT, or a field was renamed. Fix: open the upstream run's log (the card links to it), fix the cause, retry the failed stage; the runner re-resolves the arguments at retry time.

3. Prompt plus completion overflows the context. Symptom: an assertion `positions up to N exceed seq_len` in the RL stage, or an SFT warning that examples lose their tail. Cause: `--seq-len` in pretraining is smaller than the longest item. Fix: pretrain with a longer `--seq-len`; every later stage inherits the context from the checkpoint. (The RL recipe now clamps `--max-new` to what fits and refuses to run if fewer than 8 tokens fit.)

4. The held-out set leaks. Symptom: an exact match that looks too good given the model size. Cause: a hand edit that put `reason_heldout.jsonl` items into `sft.jsonl` or into the corpus, or a different seed for the split than for the corpus. Fix: the split is done once in `data_prep.py` after a seeded shuffle and every other file is derived from `train`; regenerate rather than edit.

5. The pie is all synthetic. Symptom: `traces` at 0 tokens. Cause: the export was empty (no `pair`, `preference`, or long `content` records), or on the ssh executor the export did not reach the box. Fix: check the Traces tab's SFT and DPO counts; on ssh, check the run log's first lines for the `scp` step.

6. Mid-training forgets the corpus. Symptom: `forgetting_a` large and positive while `gain_b` is small. Cause: a mixture too far from pretraining's, or too many steps at the stable rate. Fix: move `--mix` toward `a`, or shorten the stage; Lab 03's two kinds of forgetting apply, and only replay cures the intrinsic part.

7. A restart leaves a stage marked running. Symptom: the card says running, the run says failed with "the Cortex server restarted". Cause: the next tick has not happened yet. Fix: wait three seconds, then retry.

8. Two pipelines fight for one GPU. Symptom: both slow, one out of memory. Cause: the runner starts every ready stage of every running pipeline. Fix: pause one (its running stage finishes, nothing new starts) and resume it later.

## Measure it

The pipeline's number is `delta_exact_match` from the eval stage: exact match of the RL checkpoint minus exact match of the mid-training checkpoint on the held-out reasoning items, with a bootstrap 95 percent interval on each. With $n$ items the standard error of an accuracy $p$ is about $\sqrt{p(1-p)/n}$ (Lab 09), so 90 held-out items cannot separate two checkpoints closer than about ten points; raise `--n-reason` before you trust a small delta, since the held-out count scales with it. Report the two intervals, not the delta alone.

Behind that number, one per stage: `val_loss` for pretraining, the pair (`val_b_after`, `forgetting_a`) for mid-training, `exact_match_heldout` for SFT, (`reward_greedy_before`, `reward_greedy_after`) for RL. They are on different scales and should not be combined; their job is to tell you which stage to change when the final number disappoints. And one for the data: the pie, which is the only one of these you chose.

A good result at smoke scale is that every stage finishes and the intervals are computed from the right files. A good result at 5090 scale is a delta whose interval excludes zero on the synthetic items, and, separately, a look at what the model says on a handful of your own traces, which no number here measures.

## Exercises

1. Change `--mix a=0.4,b=0.6` to `a=0.8,b=0.2` in `pipeline.json` before starting and compare `val_b_after` and `forgetting_a` between the two pipelines. Check: `gain_b` is smaller and `forgetting_a` is smaller or negative.

2. Add a fourth item kind to `make_reasoning` in `data_prep.py` (a three-step sum, say) with its chain and gold, run the smoke pipeline, and confirm the pie's `reasoning` slice grows and the held-out file contains the new kind. Check: `grep` the held-out file for your new question pattern.

3. Make the reward in `grpo_reason.py` ignore the format bit (`0.0 * format + 1.0 * correct`) and run the RL stage from the same SFT checkpoint. Check: `zero_var_groups` rises, because fewer groups have any correct sample to separate.

4. Write a second template in `TEMPLATES` that fans out: two SFT stages from the same mid-training checkpoint with different `--steps`, each with its own eval against the same baseline. Check: the Pipeline tab draws two dashed arcs from `midtrain` and the runner starts both SFT stages in the same tick.

5. Break a stage on purpose (an unknown flag in its argument string), watch the pipeline fail, fix the string in `pipeline.json`, and retry. Check: only that stage and its downstream closure re-run; the upstream run ids do not change.

6. From the pretraining stage's `tokens_per_s` after warmup, compute the wall time 3000 steps implies and compare with the card's elapsed time. Check: they agree up to eval and checkpoint overhead.

## Test yourself

1. Why does the runner pass paths between stages rather than keeping the model in memory between them?

<details><summary>Answer</summary>
Because a stage is a process that can run on a different machine (the ssh executor) and can fail or be retried on its own. A path printed in a RESULT line survives a server restart, can be re-read by a retry, and makes the contract between stages inspectable: you can run the next stage by hand with the same string. Keeping objects in memory would tie every stage to one process and lose everything when it died.
</details>

2. The RL stage's reward gives 0.2 for the format and 0.8 for correctness. A colleague proposes 0.5 and 0.5 "so the model learns the format faster". What changes?

<details><summary>Answer</summary>
Groups where every sample has the format but none is correct would still have zero variance, so nothing about learning the format faster follows from the weights once the format is present. What changes is the ratio of advantage assigned to correctness against format when both vary in a group: at 0.5/0.5 a formatted wrong answer and an unformatted wrong answer are as far apart as a right and a wrong answer, and the policy is pushed toward the cheap bit as hard as toward the expensive one. The evaluator ignores the format bit, so the exact match gains nothing.
</details>

3. Why is the held-out split made in the data stage and not in the eval stage?

<details><summary>Answer</summary>
Because three earlier stages (pretraining, mid-training, SFT) would otherwise see the items the evaluator later scores. The split has to happen before any file that trains the model is written, and the same held-out file has to be the one RL evaluates on and eval scores, so it is written once and its path flows forward like every other artifact.
</details>

4. Spot the bug. A template sets `--seq-len 128` for pretraining and `--max-new 96` for the RL stage.

<details><summary>Answer</summary>
The RL policy-gradient pass scores prompt plus completion in one forward pass, and a logic prompt alone is about 125 characters, so prompt plus 96 new tokens exceeds the context. The recipe clamps `--max-new` to what fits and refuses when fewer than 8 tokens fit, but the honest fix is upstream: a longer `--seq-len` in pretraining, which every later stage inherits from the checkpoint.
</details>

5. After a server restart, a stage that was running shows as failed with "the Cortex server restarted while this run was in flight". Was work lost?

<details><summary>Answer</summary>
The run's process is gone, so that stage's work is lost, but nothing upstream is: retry resets the failed stage and its downstream closure to pending, re-resolves their arguments from the upstream RESULT lines that are still on disk, and starts them. The runner does not know whether the killed process had nearly finished; a recipe that wanted to survive that would have to checkpoint mid-run and accept `--init`.
</details>

## What will change, what will not

The runner is the part that will not change, because it is the smallest thing that works: a ready-set rule over a partial order, artifacts as the only channel, state on disk before the process starts. Marin's executor and every workflow engine before it are this rule with more bookkeeping, and whatever replaces the recipes here will still be scheduled by it.

The stages will change in their internals and not in their interfaces. Pretraining and mid-training are a loss and a schedule (Labs 02 and 03), SFT is a mask (Lab 04), RL is a reward and a ratio (Lab 05); newer optimizers, newer RL objectives, and newer architectures slot into the same six cards as long as each prints a checkpoint path. The `answer:` convention is a local choice and a fragile one: real reasoning models use structured outputs or a grader, and when this pipeline moves to an instruct model the reward and the evaluator should read whatever that model's format is, together, as they do here.

The data stage is where the interesting change will happen. Today the pie has one slice that is you and several that are synthetic, and the synthetic slices are functions the model can memorize. As the collector grows, the traces slice becomes the reason to run the pipeline at all, and the evaluation this chapter cannot do, whether the model sounds like you, becomes the one that matters; that needs the judge hook in `eval_suite.py` that is deliberately still a stub. The habit to keep is the one the pie makes visible: the corpus is a decision with numbers attached, made before any training starts.

## Read next

- "Training Compute-Optimal Large Language Models", Hoffmann, 2022. The tokens-per-parameter budget that turns the pretraining stage's step count into a decision rather than a default.
- "Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations", Hägele, 2024. The warmup-stable-decay cooldown the mid-training stage uses, and how long it should be.
- "Training language models to follow instructions with human feedback", Ouyang, 2022. The three-stage SFT, reward, RL recipe this pipeline is a small instance of.
- "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", Shao, 2024. GRPO, the group-normalized policy gradient in the RL stage.
- "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", DeepSeek-AI, 2025. Verifiable rewards on reasoning tasks at scale, and why an SFT cold start precedes RL.
- "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?", Eldan, 2023. The corpus the real data stage adds, and the argument that small models learn a small distribution well.
- "Llama 3 Herd of Models", Grattafiori, 2024. A production account of annealing data, mixture decisions, and per-stage evaluation in one pipeline.
