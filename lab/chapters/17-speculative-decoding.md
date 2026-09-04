---
title: "Lab 17: Speculative decoding"
kind: permanent
topics: [lab]
chapter: 17
station: speculative
recipe: recipes/spec_decode.py
reading_time: 45 min
---

# Lab 17: Speculative decoding

## What you will be able to do

1. Explain, with the roofline from Lab 13, why a target model can score $k$ candidate tokens in one forward pass for about the price of generating one, and say at which batch size that stops being true.
2. Write the draft-then-verify loop with the exact accept and reject rule, prove that the tokens it emits are distributed exactly as the target model's own samples, and state what the acceptance rate is in terms of the distance between draft and target.
3. Derive the expected number of tokens per verify pass, $(1 - \alpha^{k+1}) / (1 - \alpha)$, turn it into a speedup formula with the draft's cost ratio $c$, and pick $k$ from measured $\alpha$ and $c$ rather than from a blog post.
4. Choose a draft (a small sibling model, an n-gram lookup, the target's own early layers, extra prediction heads, a feature-level drafter), and say which one fits which workload and which batch regime.
5. Run `recipes/spec_decode.py` on the 5090 with a small target and a smaller draft from the same family, read acceptance rate, tokens per pass, and tokens per second from its logs, and confirm that greedy outputs are byte-identical to the target's.

## The idea in one paragraph

Generating one token from a large model costs a full read of its weights and produces one token's worth of arithmetic, so the card spends almost all of its time waiting on memory. If you hand the model $k$ tokens at once it reads the weights once and does $k$ tokens' worth of arithmetic, and the time barely moves. Speculative decoding uses that slack: a cheap draft guesses the next $k$ tokens, the large model scores all of them in a single pass, and a small coin-flip rule decides which guesses to keep. The rule is arranged so that the kept tokens are exactly what the large model would have sampled on its own, so quality does not change at all; only the number of tokens you get per pass changes, and that number depends on how often the draft agrees with the target. In the speculative station in the browser a bigram table drafts $k$ characters, the trained tiny transformer verifies them, and you can watch the acceptance rate and tokens per pass move as you swap the draft for a worse one.

## The math

### Why verification is nearly free

Lab 13 derived that a decode step at batch size 1 is a sequence of matrix-vector products with arithmetic intensity about 1 FLOP per byte, a hundred times below the 5090's ridge point of roughly 120 to 160 FLOP per byte, so the step time is bounded by weight bytes over bandwidth, not by arithmetic. Now feed the model $k + 1$ token rows instead of one. Every projection becomes a $(k+1) \times K$ by $K \times N$ matmul with intensity about $k + 1$ FLOP per byte; the weights are still read exactly once. The time bound is

$$
T(k+1) \ge \max\left(\frac{(k+1) W_1}{\pi}, \frac{Q_w + Q_{\text{kv}}}{\beta}\right),
$$

where $W_1$ is the FLOPs for one token, $Q_w$ the weight bytes, $Q_{\text{kv}}$ the cache bytes, $\pi$ the compute peak and $\beta$ the bandwidth. As long as $k + 1$ is well below the ridge, the second term wins and $T(k+1) \approx T(1)$. For a bf16 7B model on the 5090, $Q_w \approx 15$ GB gives $T(1) \approx 10$ ms, and verifying $k = 4$ extra tokens adds attention over the cache for four more query rows and a little more arithmetic, both small. At batch size $B$ the row count is $B(k+1)$ and once that approaches the ridge the verify pass is paid for in compute; this is the first thing that breaks speculation in a busy serving system, and you will see it again below.

### The algorithm

Fix a prefix $x_{\le t}$. The target model defines $p(\cdot \mid x_{\le t})$, the draft defines $q(\cdot \mid x_{\le t})$, both over the same vocabulary with the same tokenizer. One round:

1. Draft. For $i = 1, \dots, k$, sample $\tilde x_i \sim q(\cdot \mid x_{\le t}, \tilde x_{<i})$, one draft forward pass each, and keep the distribution $q_i$ you sampled from.
2. Verify. Run the target once on the prefix plus $\tilde x_1, \dots, \tilde x_k$. Its logits give $p_i = p(\cdot \mid x_{\le t}, \tilde x_{<i})$ for $i = 1, \dots, k+1$; position $k + 1$ is the target's own prediction after the whole draft.
3. Accept or reject, in order. For $i = 1, \dots, k$, draw $r_i \sim U(0, 1)$ and accept $\tilde x_i$ if

$$
r_i < \min\left(1, \frac{p_i(\tilde x_i)}{q_i(\tilde x_i)}\right).
$$

At the first rejection at position $i$, emit a replacement drawn from the residual distribution

$$
p'_i(x) = \frac{\max(0,\, p_i(x) - q_i(x))}{\sum_{x'} \max(0,\, p_i(x') - q_i(x'))},
$$

discard $\tilde x_{i+1}, \dots, \tilde x_k$, and end the round. If all $k$ are accepted, emit one more token $x \sim p_{k+1}$ and end the round.

Every round emits between 1 and $k + 1$ tokens. The rule as written is the one in Leviathan, Kalman, and Matias (2023) and in Chen et al. (2023), which derived it independently; the two papers differ only in notation.

### Why the output is exactly the target's

Take one position and drop the index. The draft proposes $x \sim q$. The probability that the round emits a particular token $x$ at this position is the sum of two disjoint events: $x$ was proposed and accepted, or something was proposed and rejected and $x$ was drawn from the residual.

$$
\Pr[\text{emit } x] = q(x) \min\left(1, \frac{p(x)}{q(x)}\right) + \Pr[\text{reject}] \cdot p'(x).
$$

The first term is $\min(q(x), p(x))$. The rejection probability is one minus the total acceptance mass,

$$
\Pr[\text{reject}] = 1 - \sum_{x'} \min(p(x'), q(x')) =: 1 - \beta.
$$

The residual's normalizer is $\sum_{x'} \max(0, p(x') - q(x'))$. Since $p$ and $q$ both sum to one, $\sum_{x'} (p(x') - q(x')) = 0$, so the positive parts and the negative parts of $p - q$ have equal total mass, and $\min(p, q) = p - \max(0, p - q)$ gives

$$
\sum_{x'} \max(0, p(x') - q(x')) = \sum_{x'} p(x') - \sum_{x'} \min(p(x'), q(x')) = 1 - \beta.
$$

So the second term is $(1 - \beta) \cdot \max(0, p(x) - q(x)) / (1 - \beta) = \max(0, p(x) - q(x))$, and

$$
\Pr[\text{emit } x] = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x).
$$

That is the whole proof for one position: whatever mass the draft under-proposes at $x$ is exactly what the residual adds back, and whatever it over-proposes is exactly what the $\min$ clips. Now the sequence. Conditioned on the prefix, the token emitted at position 1 is distributed as $p_1$ whether it came from acceptance or the residual. If it was accepted, the prefix for position 2 is the same one the draft used, so $p_2$ is the right conditional and the argument repeats. If it was rejected, the round ends and the next round starts a fresh draft from the new prefix. The bonus token after $k$ acceptances is sampled from $p_{k+1}$ directly. By induction on positions, every emitted token is a sample from the target's conditional given everything before it, which is the definition of sampling from the target. The draft never changes what is sampled, only how many samples one target pass yields.

Two remarks you will use. First, the per-position acceptance probability is $\beta = \sum_x \min(p(x), q(x)) = 1 - \mathrm{TV}(p, q)$, one minus the total variation distance between draft and target at that position. The acceptance rate $\alpha$ you measure is the average of $\beta$ over the positions the draft actually visits. So "how good is my draft" has a precise answer: it is the mean TV distance to the target on your traffic, and nothing else about the draft matters. Second, the rule only needs $p$ and $q$ at the proposed token and, on rejection, the full vectors at that one position; you never need to enumerate anything twice.

### Expected tokens per pass

Assume each position is accepted independently with the same probability $\alpha$. The round emits $N = 1 + A$ tokens, where $A$ is the number of acceptances before the first rejection, capped at $k$ (the 1 is either the residual sample or the bonus token). For a nonnegative integer variable, $\mathbb{E}[A] = \sum_{j \ge 1} \Pr[A \ge j]$, and $\Pr[A \ge j] = \alpha^j$ for $j \le k$ and $0$ beyond, so

$$
\mathbb{E}[N] = 1 + \sum_{j=1}^{k} \alpha^j = \sum_{j=0}^{k} \alpha^j = \frac{1 - \alpha^{k+1}}{1 - \alpha}.
$$

Check the limits: $\alpha \to 0$ gives 1 token per pass (you get the residual sample and nothing else), $\alpha \to 1$ gives $k + 1$. At $\alpha = 0.8$ and $k = 4$ the value is $(1 - 0.32768) / 0.2 \approx 3.36$. The independence assumption is wrong in an instructive way: acceptances are correlated because hard positions cluster (the start of a number, a rare name, a code identifier), so the real distribution of $N$ is heavier at both ends than the geometric model predicts. Measure tokens per pass directly and use the formula to reason about $k$, not to report results.

### The speedup

Let one target pass on $k + 1$ rows cost one unit (from the bandwidth argument it is nearly the cost of a single-token step) and let one draft step cost $c$ units, the cost ratio. A round costs $1 + kc$ and yields $\mathbb{E}[N]$ tokens, so

$$
S(k, \alpha, c) = \frac{1 - \alpha^{k+1}}{(1 - \alpha)(1 + kc)}.
$$

At $\alpha = 0.8$, $k = 4$, $c = 0.1$: $3.36 / 1.4 = 2.4$. At $\alpha = 0.5$ with the same $k$ and $c$: $1.94 / 1.4 = 1.39$. At $\alpha = 0.8$ but $c = 0.4$ (a draft that is not small enough): $3.36 / 2.6 = 1.29$. The formula says three things. The gain grows fast with $\alpha$ and is nearly linear in it below 0.5. The best $k$ rises with $\alpha$ and falls with $c$; differentiating is messy, but scanning $k \in \{2, \dots, 8\}$ with measured $\alpha$ and $c$ takes one line. And $c$ is not the parameter ratio: a 0.5B draft is not 7 percent of a 7B target's step, because the small model's decode is launch-bound (Lab 13 measured about 11 microseconds per kernel launch, and a 24-layer model launches hundreds of kernels per token), so $c$ is nearer 0.2 to 0.3 unless the draft runs under a CUDA graph. Measure it.

There is also a memory term. The draft's weights and its own KV cache sit on the card beside the target's, and the target's cache must hold $k$ extra positions per sequence in flight. On 32 GB with a 7B target that is a few GB, fine; with a target that already fills the card it is the reason the draft must be tiny.

### Draft choices

A smaller model from the same family with the same tokenizer is the original and simplest draft. Same tokenizer is not optional: the acceptance rule compares $p$ and $q$ at the same token id, and a draft with a different vocabulary has no $q(x)$ to compare. Families that ship a 0.5B next to a 7B with a shared tokenizer are the natural pairs, and a draft fine-tuned on the target's outputs (distillation) raises $\alpha$ because it lowers TV to the target specifically.

N-gram and prompt lookup drafts need no model at all. Prompt lookup decoding finds the last $n$ tokens of the current context somewhere earlier in the prompt and proposes the tokens that followed there. Its cost $c$ is essentially zero, and on tasks that copy from the input (summarization, extraction, retrieval-augmented answers, code editing where the model rewrites a function it was shown) $\alpha$ is high because the target really does copy. On free generation it proposes nothing useful and the gain vanishes, but it also costs nothing, which is why serving engines enable it by default for long prompts. The draft distribution here is a point mass, so the rule degenerates to "accept if the target would have sampled that token", which is a valid special case with $q$ concentrated on one token: accept with probability $\min(1, p(\tilde x))$, and on rejection sample from $p$ with the proposed token's mass removed.

Self-speculation uses the target as its own draft. One version exits early: run the first $\ell$ of $L$ layers and apply the unembedding (with a small trained exit head, or the final norm and head directly), which gives a rough $q$ at about $\ell / L$ of the cost; the verify pass then reuses the first $\ell$ layers' KV entries. Another version skips a chosen subset of intermediate layers during drafting (the draft-and-verify method of Zhang et al., 2023, selects which layers to skip by search); no extra weights are stored, which matters when memory is the constraint. The cost ratio is bounded below by the fraction of layers kept, so these drafts are not as cheap as a separate tiny model, but they need no second checkpoint.

Medusa (Cai et al., 2024) attaches several extra prediction heads to the target's final hidden state; head $j$ predicts the token $j + 1$ positions ahead from the same hidden vector, in parallel, with no autoregressive draft loop at all. The heads are trained with the backbone frozen (or lightly tuned), and their top candidates are combined into a tree of continuations that a single target pass verifies. The draft cost is one extra set of heads on a pass you were already running, so $c$ is tiny, at the price that heads predicting far ahead from a stale hidden state are individually weak, which is why the tree matters.

EAGLE (Li et al., 2024) drafts at the feature level. A small autoregressive module takes the target's second-to-last-layer features together with the embedding of the token that was actually sampled, predicts the next feature vector, and the target's own unembedding turns predicted features into a token distribution. The argument is that the feature sequence is more regular than the token sequence, and that feeding the sampled token removes the randomness the draft would otherwise have to guess; the reported acceptance lengths are higher than Medusa's at similar cost. That is as far as this chapter goes on it; read the paper for the training objective and the tree construction.

### Tree verification in outline

If the draft can offer several candidates at each position (the top few tokens from $q$, or several heads' predictions), you can verify a tree of continuations in one target pass instead of a single chain. Lay the tree's nodes out as a flat sequence of $M$ tokens, give each node the position id of its depth, and build an attention mask in which each node attends to its ancestors and to the prefix, nothing else. One forward pass then yields the target's distribution at every node. Walking the tree from the root, the acceptance rule is applied branch by branch; a version that stays exact when several siblings are tried in turn (SpecInfer, Miao et al., 2023) subtracts each rejected sibling's mass from the residual before trying the next. The longest accepted path is the round's output. The cost is $M$ rows in the verify pass instead of $k + 1$; on a bandwidth-bound target that is still nearly free up to a few dozen rows, and beyond that the tree starts costing compute. Medusa and EAGLE both use a fixed sparse tree shape chosen offline from candidate-acceptance statistics.

### What breaks it

A poorly matched draft. Since $\alpha = 1 - \overline{\mathrm{TV}}(p, q)$, a draft trained on different data, a different chat template, or a different system prompt disagrees with the target exactly where it matters, and $\alpha$ on your traffic can be far below the number in a paper measured on plain text. The unigram draft in the browser station is the extreme case: it knows the character frequencies and nothing else, $\alpha$ falls, and tokens per pass slide toward 1.

Sampling temperature. The rule is exact at any temperature provided you apply the same temperature, top-$k$, and top-$p$ processing to both $p$ and $q$ and use the processed $q$ that the draft actually sampled from. But $\alpha$ moves. At temperature 0 the rule is deterministic: accept if and only if the draft's argmax equals the target's, so $\alpha$ is the top-1 agreement rate, usually the highest value you will see. As the temperature rises both distributions flatten, the draft's samples wander into tokens the target assigns little mass, and $\alpha$ falls. Report $\alpha$ together with the sampling settings.

Batch size larger than one. Two effects. The verify pass now has $B(k+1)$ rows, and once that is near the ridge the pass costs compute, so the "nearly free" premise fails; at $B = 32$ and $k = 4$ that is 160 rows, at the ridge already. And sequences in the batch accept different numbers of tokens, so after each round the batch is ragged: some sequences advanced 5 positions, some 1. Serving engines handle this with per-sequence position bookkeeping and paged caches, and they turn speculation off above a batch threshold because the gain has gone.

KV cache handling on rejection. The verify pass wrote keys and values for all $k$ drafted positions into the target's cache. If the round accepted $a$ tokens and then rejected, the entries at positions $a + 1$ onward are for tokens that are not in the sequence and must be discarded: crop the cache back to prefix length plus $a$, and note that the residual-sampled replacement has no cache entry yet (it is computed on the next verify pass, as the first row of its input). The draft's cache must be cropped the same way. Forgetting either one is the most common bug in a hand-written loop; the model keeps running and its outputs drift into repetition. With paged caches the crop is a block-table edit, and the drafted positions that were never accepted are why the target cache needs $k$ tokens of slack per sequence.

### How to measure it

Acceptance rate $\alpha$: accepted proposals divided by proposals tested, where a rejection counts as one tested position (a round that rejects at position 3 tested 3 positions, not $k$). Mean accepted length, or tokens per pass: total emitted tokens over verify passes, including the residual or bonus token; compare it to $(1 - \alpha^{k+1})/(1 - \alpha)$ to see how far the independence assumption is off. Tokens per second, wall clock, measured against the same target with the same sampling settings and the same batch size without speculation; the ratio is the real speedup and it includes everything the formula ignores (draft launch overhead, cache crops, the CPU loop). And exactness: with temperature 0 the speculative output must be byte-identical to the target's greedy output for every prompt, which is a cheap, strict test that catches cache bugs and off-by-one position errors. With sampling, a single output cannot be compared, so check a statistic: the mean log-probability under $p$ of speculative outputs against that of plain samples over a few hundred generations, or run the small-vocabulary test below where the distribution can be counted exactly.

## Build it small

The snippet implements the rule with two "language models" that are $V \times V$ tables of next-token probabilities conditioned on the previous token (a Markov chain, so the exactness check can count every conditional). The target $p$ is a random sharp table; the draft $q$ is the target with noise added in log space, so the two disagree by a controlled amount. It generates 100k tokens speculatively with $k = 4$, counts every (previous, next) pair, and compares the empirical conditionals against $p$ with a chi-square statistic (expected value equals its degrees of freedom, $V(V - 1) = 240$, if the sampler is exact) and with the mean total variation across contexts. Two controls make the test meaningful: the same counts from plain sampling of $p$ (should look the same), and from the draft alone (should fail). It also reports the acceptance rate against its prediction $\sum_x \min(p, q)$ and tokens per pass against the formula.

```python
import torch

torch.manual_seed(0)
V, k, N = 16, 4, 100_000                       # vocab, draft length, tokens to generate

def markov_model(sharpness):                   # a "language model" over V tokens conditioned on the previous one
    return torch.softmax(torch.randn(V, V) * sharpness, dim=-1)

p = markov_model(2.0)                                                # target
q = torch.softmax(p.log() + 0.8 * torch.randn(V, V), dim=-1)         # draft: a noisy copy of the target
alpha_true = torch.minimum(p, q).sum(-1)                             # per-context acceptance rate sum_x min(p, q)


def spec_round(prev):
    """Draft k tokens from q, verify them against p. Returns emitted tokens and how many were accepted."""
    draft, ctx = [], prev
    for _ in range(k):                         # draft runs autoregressively
        ctx = torch.multinomial(q[ctx], 1).item()
        draft.append(ctx)
    ctx, out = prev, []
    for i, x in enumerate(draft):              # verify: in a real model these p rows come from ONE target pass
        if torch.rand(()) < min(1.0, p[ctx, x] / q[ctx, x]):
            out.append(x); ctx = x             # accept with probability min(1, p/q)
        else:
            resid = (p[ctx] - q[ctx]).clamp(min=0)          # rejected: resample from norm(max(0, p - q))
            out.append(torch.multinomial(resid / resid.sum(), 1).item())
            return out, i
    out.append(torch.multinomial(p[ctx], 1).item())         # all k accepted: free token from the target's last position
    return out, k


def generate(step_fn, n):
    counts, ctx, rounds, accepted, tested = torch.zeros(V, V), 0, 0, 0, 0
    while counts.sum() < n:
        toks, acc = step_fn(ctx)
        rounds += 1; accepted += acc; tested += min(acc + 1, k)   # a rejection also tests one position
        for t in toks:
            counts[ctx, t] += 1; ctx = t
    return counts, rounds, accepted, tested


def check(counts, name):
    """Chi-square of the empirical next-token counts against p, row by row, plus the mean total variation."""
    rows = counts.sum(-1, keepdim=True)
    expected = rows * p
    chi2 = ((counts - expected) ** 2 / expected).sum().item()
    tv = 0.5 * (counts / rows - p).abs().sum(-1).mean().item()
    print(f"{name:>12}: chi2={chi2:8.1f} (df={V * (V - 1)}, expect ~{V * (V - 1)} if exact)  mean TV={tv:.4f}")


counts, rounds, accepted, tested = generate(spec_round, N)
check(counts, "speculative")
check(generate(lambda c: ([torch.multinomial(p[c], 1).item()], 0), N)[0], "plain p")
check(generate(lambda c: ([torch.multinomial(q[c], 1).item()], 0), N)[0], "draft q only")
a = accepted / tested
print(f"acceptance rate: {a:.3f} (sum_x min(p,q) averaged over contexts visited: "
      f"{(alpha_true * counts.sum(-1) / counts.sum()).sum():.3f})")
print(f"tokens per verify step: {counts.sum().item() / rounds:.3f}   "
      f"formula (1 - a^(k+1)) / (1 - a) = {(1 - a ** (k + 1)) / (1 - a):.3f}")
```

Output from a run on the CPU with PyTorch 2.10, about 12 seconds:

```
 speculative: chi2=   264.9 (df=240, expect ~240 if exact)  mean TV=0.0150
     plain p: chi2=   252.2 (df=240, expect ~240 if exact)  mean TV=0.0133
draft q only: chi2=158892.7 (df=240, expect ~240 if exact)  mean TV=0.2705
acceptance rate: 0.717 (sum_x min(p,q) averaged over contexts visited: 0.715)
tokens per verify step: 2.874   formula (1 - a^(k+1)) / (1 - a) = 2.864
```

Read it in three parts. The speculative counts and the plain-sampling counts are statistically indistinguishable from $p$: chi-square 265 and 252 against 240 expected with a standard deviation of about $\sqrt{2 \cdot 240} \approx 22$, both within about one standard deviation, and a residual TV of 0.015 that is what 100k samples spread over 240 cells give you. The draft alone is off by a TV of 0.27 and a chi-square six hundred times too large, so the check has teeth: the acceptance rule, not the draft's closeness, is what made the first line match. The measured acceptance rate 0.717 agrees with $\sum_x \min(p, q)$ averaged over visited contexts, 0.715, which is the $1 - \mathrm{TV}$ identity, and tokens per pass 2.874 agrees with the formula's 2.864 to within the independence approximation. Now change one thing: set the noise in the draft to 2.0 instead of 0.8 and watch $\alpha$ fall while the chi-square line stays near 240; then delete the `clamp` and the residual, replacing rejection with a plain sample from $p$, and watch the first line's chi-square explode, since that shortcut over-samples what the draft under-proposed.

The browser station is the same loop with the trained character transformer as $p$ and a bigram table from the corpus as $q$: it prints the acceptance rate, tokens per pass, and the theory curve, and brackets the characters that came from the draft in each pass.

## Build it real

`recipes/spec_decode.py` runs a target and a draft from the same family on the 5090 in four modes and reports the same four numbers as the snippet. The default pair is a 7B instruct target with a 0.5B instruct draft that share a tokenizer, both in bf16; together they are about 16 GB of weights, which leaves room for the two caches at a few thousand tokens. A 1.5B target with the 0.5B draft also runs and is useful for debugging, but its decode step is launch-bound rather than bandwidth-bound (Lab 13), so its measured speedup understates what the mechanism does; use it to check exactness, not to report speed.

Arguments. `--target` and `--draft` name the checkpoints; the recipe asserts that their tokenizers produce identical ids on a probe string before loading weights. `--mode baseline|hf|manual|lookup` selects plain generation, `transformers` assisted generation (`model.generate(..., assistant_model=draft)`, which implements the same accept and reject rule when `do_sample=True` and reduces to argmax agreement when it is false), the recipe's own verify loop, or prompt lookup (`prompt_lookup_num_tokens`) with no draft model. `--k` sets the draft length (assisted generation adjusts its own draft length adaptively unless you pin it; the manual loop uses `--k` exactly). `--temperature`, `--top-p`, and `--max-new-tokens` apply identically to every mode. `--prompts` is a JSONL file of prompts; the default set mixes short-answer questions, a summarization prompt with a long input, and a code-editing prompt, so you can see $\alpha$ vary by task. `--batch` runs several prompts at once in the baseline and hf modes, to show the gain shrinking with batch size.

The manual loop is the part to read. Per round it runs the draft for `k` steps with its own `DynamicCache`, runs the target once on the `k` new tokens with `use_cache=True` and takes the logits at all `k + 1` positions, applies the temperature and top-p processing to both `p` and `q` in fp32, applies the rule, and then crops both caches to the accepted length with `cache.crop(n)` before the next round. Position ids are computed from the cache length, not from the input length, which is the line that goes wrong most often.

What it logs. One `METRIC` line per prompt with `accept_rate`, `tokens_per_pass`, `tok_s`, `baseline_tok_s`, and `speedup`, plus for temperature 0 an `exact` flag that is true when the mode's output matches the baseline's string for that prompt; a `RESULT` line at the end with the mean of each over prompts. What to expect from the formulas, not from a measurement: the bandwidth bound for a bf16 7B model on this card is about 10 ms per token, so the baseline should sit within a factor of 1.5 of 100 tokens per second at batch 1; if it does not, the baseline is not bandwidth-bound and the speedup will disappoint for reasons unrelated to speculation. With a draft cost ratio you measure as $c$ and an $\alpha$ you read from the log, the expected speedup is the formula above, and the log's `speedup` should land within 20 percent of it; a larger gap means overhead in the loop (host syncs, per-round Python, cache crops on a fragmented allocator). Runtime is a few minutes per mode over the default prompt set, after the one-time download.

## How it goes wrong

Tokenizer mismatch. Symptom: assisted generation raises, or the manual loop produces plausible-looking garbage with $\alpha$ near zero. Cause: the draft and target do not share a vocabulary, so token ids do not mean the same thing and $q(x)$ is not defined for the target's $x$. Fix: pick pairs from the same family and assert identical ids on a probe string; if you must use a different family, the only exact option is to re-tokenize each proposal, which is slow enough to defeat the purpose.

The rule uses a $q$ the draft did not sample from. Symptom: greedy outputs match the target but sampled outputs are subtly biased (the small-vocabulary test's chi-square is a few times too large). Cause: the draft sampled with top-$k$ or a temperature, but the rule divides by the raw softmax. Fix: apply the identical logit processing to both models and keep the processed $q$ for the rule; the proof needs the $q$ that generated the proposal.

The cache is not cropped. Symptom: the first few rounds are fine, then the text degenerates into repetition; greedy exactness fails on prompts longer than a few rounds. Cause: after a rejection the target's cache still holds keys and values for rejected tokens, so the next pass attends to positions that are not in the sequence. Fix: crop both caches to the accepted length every round; test with temperature 0 and a prompt that forces early rejections (a bad draft) so the bug cannot hide behind high acceptance.

Position ids computed from the input. Symptom: exactness fails only in the manual mode and only after the first round. Cause: the verify pass feeds `k` tokens, and the code used `arange(k)` for their positions instead of `cache_len + arange(k)`, so RoPE rotated them as if they were the start of the sequence (Lab 11 explains why the model still runs). Fix: derive positions from the cache length.

Speedup below one. Symptom: `speedup` in the log is 0.7 to 0.9. Causes, in order of likelihood: the target is small and launch-bound, so its single-token step was never as expensive as the byte count says; the draft is not small enough or is itself launch-bound, so $c$ is 0.4 rather than 0.1; the batch is large enough that the verify pass costs compute; $k$ is too large for a low $\alpha$, so most drafted tokens are thrown away. Fix: compute $S(k, \alpha, c)$ from measured $\alpha$ and $c$ before blaming the implementation, and lower $k$ or switch to a lookup draft when $\alpha$ is low.

Acceptance measured on the wrong distribution. Symptom: a paper's $\alpha$ of 0.85 becomes 0.55 on your prompts. Cause: $\alpha$ is one minus the mean TV between draft and target on the positions the draft visits, and it depends on the task (copying tasks are easy, open-ended generation is hard), on the temperature, and on whether the draft saw the same chat template. Fix: measure on your traffic at your sampling settings; distill the draft on the target's outputs if the gap matters.

Division by a draft probability of zero. Symptom: NaN or inf in the acceptance ratio, or a crash in `multinomial` when the residual sums to zero. Cause: in bf16 the draft assigned an underflowed probability to the token it nonetheless sampled, or $p$ and $q$ are numerically identical at a rejected position so the residual is all zeros. Fix: compute $p$ and $q$ in fp32 from the logits, clamp $q$ below by a tiny epsilon in the ratio, and fall back to sampling from $p$ if the residual's sum is zero (the event has probability zero in exact arithmetic).

A passing test that proves nothing. Symptom: greedy outputs are identical and you ship, then users report the sampled outputs are duller than before. Cause: at temperature 0 the rule reduces to argmax agreement and cannot exercise the residual branch, so cache and position bugs are caught but a wrong residual is not. Fix: keep the greedy test, and add a sampled test on a small-vocabulary or short-answer task where the target's distribution can be estimated by counting, as in the snippet.

## Measure it

Report four numbers for every configuration, and always with the sampling settings and batch size they were measured at: acceptance rate $\alpha$ (a good draft for a same-family pair sits around 0.7 to 0.9 at low temperature on ordinary text and lower on open-ended sampling; a lookup draft on a copying task can exceed 0.9 and on free generation is near 0), tokens per pass (compare with $(1 - \alpha^{k+1})/(1 - \alpha)$, and treat a large shortfall as evidence of clustered rejections that a smaller $k$ would waste less on), wall-clock tokens per second against the unaccelerated target at the same batch size (the only number that is a speedup), and an exactness result (byte-identical greedy outputs, plus a sampled statistic). A speedup of 2 at batch 1 on a bandwidth-bound target with a good draft is realistic; a speedup above $1 + k$ is impossible and indicates a timing bug; a speedup that survives at batch 32 indicates the target was not bandwidth-bound to begin with, which is worth understanding before you celebrate.

The single most useful diagnostic plot is $\alpha$ per position within a round (position 1 through $k$): if it falls steeply, the draft's errors compound and a smaller $k$ or a tree helps; if it is flat, the independence model holds and the formula's $k$ is right.

## Exercises

1. Show that when the draft is a point mass on $\tilde x$ (prompt lookup), the rule reduces to accepting with probability $p(\tilde x)$ and, on rejection, sampling from $p$ with $\tilde x$ removed and renormalized. Check: with $q = \delta_{\tilde x}$, $\min(1, p/q)$ at $\tilde x$ is $p(\tilde x)$, and $\max(0, p - q)$ is $p$ off $\tilde x$ and $0$ at it.

2. In the snippet, replace the residual sample with a plain sample from $p$ on rejection and rerun. Check: chi-square for the speculative line rises by an order of magnitude or more while the acceptance rate is unchanged; explain which tokens are over-represented (those the draft under-proposes, whose deficit is now filled twice).

3. Derive the variance of $N$ under the independence model and compare it with the empirical variance of tokens per pass from the snippet (record the per-round counts). Check: $\mathbb{E}[N^2] = \sum_{j=0}^{k} (2j + 1)\alpha^j$, and the empirical variance is larger when acceptances are correlated.

4. For a bf16 7B target on the 5090 (assume $T(1) = 10$ ms at batch 1) and a draft with measured $c = 0.25$, tabulate $S(k, \alpha, c)$ for $k \in \{1, \dots, 8\}$ and $\alpha \in \{0.5, 0.7, 0.9\}$ and read off the best $k$ for each. Check: at $\alpha = 0.5$ the best $k$ is 2 or 3 and the speedup is under 1.4; at $\alpha = 0.9$ it is $k = 6$ to $8$ and about 2.5.

5. Write the tree version of the snippet for a binary tree of depth 2 (two candidates at position 1, two children each), using SpecInfer's rule that a rejected sibling's mass is removed from the residual before its sibling is tried. Check: the chi-square test still passes, and tokens per pass rises over the chain version at the same $\alpha$.

6. On the 5090, run `recipes/spec_decode.py --mode manual` at temperatures 0, 0.7, and 1.0 with the same prompts. Check: $\alpha$ falls monotonically with temperature, greedy outputs are exact, and the speedup at 1.0 is within 20 percent of $S(k, \alpha, c)$ computed from the logged $\alpha$ and your measured $c$.

## Test yourself

1. A colleague says speculative decoding is a lossy approximation that trades a little quality for speed. What is the precise counter-argument, and what single quantity does the draft affect?

<details><summary>Answer</summary>
The emitted token at each position has probability $\min(p, q) + \max(0, p - q) = p$ exactly, so the output distribution is the target's, not an approximation of it. The draft affects only the acceptance rate $\alpha = 1 - \overline{\mathrm{TV}}(p, q)$ and therefore the number of tokens per verify pass; it cannot change which distribution the tokens come from. What can change quality is an implementation that uses a different $q$ than the draft sampled from, which is a bug, not a property of the method.
</details>

2. Why is the target's verify pass on $k + 1$ tokens about as cheap as a single-token step at batch 1, and at roughly what total row count on the 5090 does that stop being true?

<details><summary>Answer</summary>
At batch 1 a decode step reads all weight bytes to do about one FLOP per byte, far below the ridge of roughly 120 to 160 FLOP per byte, so time is weight bytes over bandwidth. Feeding $k + 1$ rows reuses each weight read for $k + 1$ multiply-adds; intensity becomes about $k + 1$ and the time bound does not change until $B(k + 1)$ approaches the ridge, so somewhere above 100 to 150 rows in total. Beyond that the verify pass costs compute proportional to the row count and speculation gives back its gain.
</details>

3. Derive $\Pr[\text{reject}]$ at one position and show it equals the total variation distance between $p$ and $q$.

<details><summary>Answer</summary>
Acceptance mass is $\sum_x q(x) \min(1, p(x)/q(x)) = \sum_x \min(p(x), q(x))$. Since $\min(p, q) = (p + q - |p - q|)/2$ and both sum to one, this is $1 - \tfrac{1}{2}\sum_x |p(x) - q(x)| = 1 - \mathrm{TV}(p, q)$. So $\Pr[\text{reject}] = \mathrm{TV}(p, q)$.
</details>

4. Spot the bug:

```python
for i, x in enumerate(draft):
    if torch.rand(()) < min(1.0, p[ctx, x] / q[ctx, x]):
        out.append(x); ctx = x
    else:
        out.append(torch.multinomial(p[ctx], 1).item())
        return out, i
```

<details><summary>Answer</summary>
On rejection it samples from $p$ instead of the residual $\mathrm{norm}(\max(0, p - q))$. The accepted branch already contributes $\min(p, q)$ at every token; adding $(1 - \beta) p$ instead of $\max(0, p - q)$ gives $\min(p, q) + (1 - \beta)p \ne p$, over-sampling tokens the draft under-proposes and under-sampling the ones it over-proposes. The greedy test passes (rejection at temperature 0 is deterministic either way), which is why the sampled test in the snippet exists.
</details>

5. With $\alpha = 0.6$, is $k = 8$ better or worse than $k = 3$ for a draft with $c = 0.15$? Show the arithmetic.

<details><summary>Answer</summary>
$k = 3$: $\mathbb{E}[N] = (1 - 0.6^4)/0.4 = 2.176$; cost $1 + 0.45 = 1.45$; $S = 1.50$. $k = 8$: $\mathbb{E}[N] = (1 - 0.6^9)/0.4 = 2.475$; cost $1 + 1.2 = 2.2$; $S = 1.13$. The longer draft adds only 0.3 expected tokens (most of the tail is rejected) but pays five more draft steps. At low $\alpha$ keep $k$ short.
</details>

6. Why can a prompt-lookup draft give a large speedup on summarization and none on a creative-writing prompt, when the draft "model" is identical in both cases?

<details><summary>Answer</summary>
The acceptance rate is one minus the TV distance between the draft and the target on the visited positions. On summarization the target copies spans from the input, and a lookup that proposes the continuation of a matched n-gram lands on exactly those spans, so $\alpha$ is high; on free generation the target rarely reproduces an earlier n-gram's continuation, the point-mass draft is almost always wrong, and every round yields one token. Since $c \approx 0$, the failure costs nothing, which is why it is safe to leave on.
</details>

7. After a round accepts 2 of 4 drafted tokens and samples a replacement from the residual, what does each cache contain and what must happen before the next round?

<details><summary>Answer</summary>
The target's cache holds entries for the prefix plus all 4 drafted positions (written during the verify pass); the draft's cache holds the prefix plus 4 as well. Positions 3 and 4 belong to rejected tokens and must be cropped from both. The replacement token has no entry in either cache; it becomes the first input token of the next round (the draft consumes it to start drafting, the target consumes it as the first of the $k + 1$ verify rows), so the next round's position ids start at prefix length plus 3.
</details>

8. Medusa drafts with heads that all read the same hidden state. Why does a head predicting three tokens ahead have a lower acceptance rate than one predicting one ahead, and what does the tree do about it?

<details><summary>Answer</summary>
The hidden state at position $t$ encodes the context up to $t$; predicting $x_{t+3}$ from it requires marginalizing over the unknown $x_{t+1}$ and $x_{t+2}$, so the head's distribution is a mixture over many futures and is diffuse, while the $x_{t+1}$ head sees the relevant context directly. A tree proposes several top candidates at each depth so that one of the paths through the diffuse heads is likely to match the target; the single verify pass scores them all, so the extra candidates cost rows, not passes.
</details>

9. You measure $\alpha = 0.85$ at batch 1 and a speedup of 2.1. At batch 16 with the same models, the speedup is 1.05. Give two independent reasons.

<details><summary>Answer</summary>
First, the verify pass at batch 16 with $k = 4$ has 80 rows, near the 5090's ridge, so its cost is no longer close to a single-token step and the "free verification" premise fails; the baseline at batch 16 also amortizes the weight read across 16 sequences, so its per-token cost was already lower. Second, the rounds are ragged: sequences that rejected early wait for the ones that accepted everything, and the batch advances at the pace of the slowest sequence's bookkeeping while the draft runs $k$ steps for all of them. Neither is fixable by a better draft; it is why serving engines turn speculation off above a batch threshold.
</details>

10. The independence model predicts 2.86 tokens per pass and you measure 2.87 in the snippet, but on a real model you measure 2.3 at the same $\alpha$. What is different, and what would you change?

<details><summary>Answer</summary>
In the Markov toy the acceptance probability at a position depends only on the previous token, and the draft's errors do not compound in a structured way, so acceptances are nearly independent. In a real model, rejections cluster at hard spans (numbers, names, identifiers): once the draft is wrong at position $i$, it is often wrong at $i + 1$ too, and conversely easy spans are accepted in long runs. The distribution of accepted length is then heavier at 0 and at $k$ than geometric, and the mean is lower at the same average $\alpha$. Plot $\alpha$ by position within a round; if it falls with position, shorten $k$ or use a tree; if it is bimodal by prompt, distill the draft on the hard spans.
</details>

## What will change, what will not

The acceptance rule and its proof are not going to change. They are a statement about two distributions on the same finite set, and every variant published since (trees, multiple drafts, self-drafts, feature-level drafts) either uses the rule unchanged or uses a multi-candidate generalization that reduces to it on a chain. The identity $\alpha = 1 - \overline{\mathrm{TV}}(p, q)$ will keep being the whole story of what makes a draft good, and the formula for expected tokens per pass will keep being the first thing to compute before building anything.

The premise that verification is nearly free is a fact about the current balance between memory bandwidth and compute on accelerators, and about serving at small batch. It has held for every GPU generation so far because bandwidth grows more slowly than peak FLOPs, so the ridge keeps moving up and single-stream decode keeps getting more bandwidth-bound, not less. Two things could weaken it: serving systems that run at batch sizes where decode is compute-bound anyway (then speculation gives nothing and costs a draft), and architectures whose per-token state is not a growing cache (state-space and hybrid models change the byte count but not the GEMV intensity, so the premise mostly survives).

The drafts are the moving part. A separate small model is the simplest and the most robust across tasks; lookups win on copying tasks; heads and feature-level drafters win on cost and are tied to a particular target and its training; self-speculation wins when memory is the constraint. Which of these a serving engine picks by default will change with every release, and the acceptance rates in their papers are measured on their benchmarks at their temperatures. The measurement discipline in this chapter (acceptance by position, tokens per pass against the formula, wall clock against the same batch size, exactness against the target) is what transfers.

## Read next

1. "Fast Inference from Transformers via Speculative Decoding", Leviathan, 2023. The rule, the exactness proof, the expected-length formula, and the first measurements with a small T5 draft.
2. "Accelerating Large Language Model Decoding with Speculative Sampling", Chen, 2023. The same rule derived independently at DeepMind, with the residual written as $\mathrm{norm}(\max(0, p - q))$ and results on a 70B target.
3. "Blockwise Parallel Decoding for Deep Autoregressive Models", Stern, 2018. The earlier idea of predicting several positions at once and verifying them in one pass, for greedy decoding; the ancestor of Medusa's heads.
4. "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification", Miao, 2023. Tree attention masks and the multi-candidate acceptance rule that stays exact when siblings are tried in turn.
5. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads", Cai, 2024. Extra heads on the final hidden state, tree candidates, and training with a frozen or lightly tuned backbone.
6. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", Li, 2024. Feature-level autoregressive drafting with the sampled token fed back, and why it accepts more than token-level heads.
7. "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding", Zhang, 2023. Skipping layers of the target to draft, with no extra weights, and how the skipped set is chosen.
8. Lab 13 in this series for the roofline that makes verification cheap, and Lab 11 for the KV cache and the position ids the verify loop must get right.
