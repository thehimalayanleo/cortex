---
title: "Lab 09: Evaluations that mean something"
kind: permanent
topics: [lab]
chapter: 9
station: none
recipe: recipes/eval_suite.py
reading_time: 50 min
---

## What you will be able to do

1. Say precisely what a given eval measures, separate a capability claim from a behaviour claim, and choose log-likelihood or generative scoring for a reason you can defend.
2. Compute how many items you need to detect a difference of a given size, and put a paired confidence interval on any before-and-after comparison rather than reading two numbers off a table.
3. Test a benchmark for contamination with three independent methods and say which of exposure and benefit each one detects.
4. Use a language model as a judge without being fooled by its position, length, and self-preference biases, and report its agreement with your own labels as a number.
5. Build a private eval from your own tasks, then run lm-eval-harness and that private eval on a base model and its SFT checkpoint from Lab 04 on the 5090, and produce one table with intervals.

## The idea in one paragraph

An eval is a sample of questions, a rule for scoring an answer, and an average. Every number you have ever seen on a leaderboard is an estimate of that average from a finite sample, drawn under one particular prompt format, scored by one particular rule, on questions the model may or may not have seen during training. The number means something only if you know all four of those things and have measured how much each could move it: the sampling noise (which is larger than most people think, and shrinks only with the square root of the item count), the prompt sensitivity (which can be several points), the scoring rule (log-likelihood of the right answer versus actually producing it), and contamination (which you can test for). A model judging another model adds a fifth: the judge's own systematic preferences, which you calibrate against your labels like any other instrument. The most valuable eval you will own is a private one built from tasks you actually care about, because it is the only one whose questions the model has certainly never seen and whose scoring rule you wrote.

## The math

### What an eval measures

An eval is a triple: a distribution $\mathcal{D}$ over inputs $x$ (a benchmark is a finite sample from it), a scoring rule $g(x, y) \in [0, 1]$ that grades a model output $y$, and the target quantity

$$
\mu(M) = \mathbb{E}_{x \sim \mathcal{D}}\big[g(x, M(x))\big],
$$

which you estimate by $\hat{\mu} = \frac{1}{n}\sum_{i=1}^{n} g(x_i, M(x_i))$ over $n$ items. Everything else in this chapter is about how $\hat{\mu}$ can differ from the thing you wanted to know, and the first way is that you asked the wrong question. A capability eval asks whether the model can do the task under favourable elicitation: few-shot examples, chain-of-thought, several samples with the best kept. A behaviour eval asks what the model does under one fixed, realistic prompt: whether it refuses, hedges, follows the format, or takes a dangerous action. The same model gets different numbers on the same items depending on which you run, and both are legitimate; the error is to read one as the other, for instance to conclude "the model cannot do X" from a zero-shot behaviour eval, or "the model is safe" from a capability eval that never used the deployment prompt. Decide which question you are asking, then fix the elicitation accordingly and write it down.

### Log-likelihood versus generative scoring

For a multiple-choice item with prompt $x$ and candidate continuations $c_1, \dots, c_m$, log-likelihood scoring computes for each candidate the sum of token log-probabilities under the model,

$$
\ell_j = \sum_{t=1}^{|c_j|} \log p_\theta\big(c_j^{(t)} \mid x, c_j^{(<t)}\big),
$$

and predicts $\arg\max_j \ell_j$. The model never produces anything; you read its probabilities. Longer candidates accumulate more negative terms, so lm-eval-harness reports both `acc` (raw $\ell_j$) and `acc_norm`, which divides by the candidate's length in bytes, $\ell_j / \mathrm{bytes}(c_j)$; a third variant, unconditional normalization, subtracts $\log p_\theta(c_j \mid \text{"Answer:"})$ to remove the prior probability of the answer text itself. When the candidates are single letters (the MMLU convention of comparing the logits of ` A`, ` B`, ` C`, ` D`) the cost is one forward pass per item, but the score now depends on the model having learned the letter-answer convention, which small and base models often have not.

Generative scoring lets the model write, then parses the output and compares it with a reference: exact match after normalization, a regex that extracts the final number, symbolic equivalence for mathematics, unit tests for code. For code, with $n$ samples of which $c$ pass, the unbiased estimator of the probability that at least one of $k$ samples passes is

$$
\widehat{\mathrm{pass@}k} = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}},
$$

which needs $n > k$ samples; the plug-in $1 - (1 - c/n)^k$ is biased downward (the function is concave in $p$, so Jensen's inequality applies) and has higher variance. Generative scoring measures what the model would actually do, including format failures and refusals, which is both its strength (it is the behaviour) and its weakness (a parser bug is indistinguishable from a wrong answer unless you count parse failures separately). Log-likelihood scoring is cheap, deterministic, and applicable to base models that cannot follow instructions, and it says nothing about whether the model would ever produce the answer. Report which one you used; they are different measurements.

### Few-shot formatting effects

The prompt template is part of the eval. The number of shots, their order, the balance of labels among them, the separator between question and answer, and even whether the answer is preceded by a space (which changes the tokenization of the first answer token) each move accuracy, and on small models by several points. Two effects have names. Majority-label bias: the model prefers whichever label appeared most among the shots. Recency bias: it prefers the label of the last shot. Contextual calibration measures the model's output distribution on a content-free input (the same template with the question replaced by "N/A") and rescales predictions so that the content-free input is predicted uniformly; it recovers some of the lost accuracy and, more usefully, tells you how much the template alone was steering the answer. The practical rule is to pin the template, the shot set, and their order, report them, and, when comparing two models, use the same template for both, since the SFT checkpoint and the base model will not have the same sensitivities.

### Contamination and how to test for it

Contamination is the presence of test items, or close paraphrases, in the training data. Distinguish exposure (the model saw the items) from benefit (the score is higher because it saw them); the first does not imply the second, and only the second invalidates the number. Three tests, each detecting something different.

Overlap search detects exposure when you have the training data. Tokenize both sets and count test items that share an $n$-gram with any training document; GPT-3's analysis used 13-grams and reported scores on the clean subset separately. This is the test you can run exactly on your own SFT mixture from Lab 04, and the recipe does.

The exchangeability test detects exposure without training data. Benchmark items are exchangeable: any ordering is as likely as any other. If the model has seen the benchmark file, the canonical order is more probable to it than a random permutation. Compute $\log p_\theta$ of the items concatenated in canonical order and in $R$ random orders, and report the fraction of permutations that score at least as high as the canonical one; that fraction is a valid $p$-value under the null of no exposure, since under the null the canonical order is just one more draw.

Matched-set comparison detects benefit. Write, or have written, a fresh set of items of the same form and difficulty (the recipe's `--fresh` split of your private eval is this by construction), and compare accuracy on the public and fresh sets with a paired-difficulty design; a gap well outside the confidence interval is benefit. Canary strings, a unique token planted in a benchmark file so that its presence in a model's memory can be probed, serve the same purpose in the other direction: put one in your private eval and check occasionally that no model completes it.

### Variance and confidence intervals

Each item's score is a Bernoulli draw with mean $\mu$, so the standard error of $\hat{\mu}$ is

$$
\mathrm{SE}(\hat{\mu}) = \sqrt{\frac{\hat{\mu}(1 - \hat{\mu})}{n}},
$$

and a 95 percent interval is $\hat{\mu} \pm 1.96\,\mathrm{SE}$. Worked example: at $\hat{\mu} = 0.7$ and $n = 500$, $\mathrm{SE} = 0.0205$, so the interval is $\pm 4.0$ points. To shrink it to $\pm 1$ point you need $n = 1.96^2 \cdot 0.21 / 0.01^2 \approx 8{,}067$ items. That is the whole reason a 2-point gap between two models on a 500-item benchmark is not a finding.

Comparing two models on the same items is a paired design, and the pairing is where the power comes from. Let $n_{10}$ be the number of items model A gets right and B wrong, $n_{01}$ the reverse. The difference in accuracies is $(n_{10} - n_{01})/n$, and its variance is

$$
\mathrm{Var}(\hat{\mu}_A - \hat{\mu}_B) = \frac{n_{10} + n_{01}}{n^2} - \frac{(n_{10} - n_{01})^2}{n^3},
$$

which depends on the disagreement rate, not on the accuracies. Worked example: $n = 1{,}000$, $\hat{\mu}_A = 0.70$, $\hat{\mu}_B = 0.73$, and the models disagree on 12 percent of items. Unpaired, the standard error of the difference is $\sqrt{0.21/1000 + 0.197/1000} = 0.020$, and the 3-point gap is inside $\pm 4.0$. Paired, the variance is $0.12/1000 - 0.03^2/1000 = 1.19 \times 10^{-4}$, the standard error is $0.011$, and the interval is $\pm 2.1$ points, so the same data now distinguishes the models. The same computation as a hypothesis test is McNemar's; as a distribution-free procedure it is the paired bootstrap, which resamples items with replacement and recomputes the difference each time. To size a private eval, the number of items needed to detect a paired difference $\delta$ with disagreement rate $q$ at 5 percent significance and 80 percent power is about

$$
n \approx \frac{(1.96 + 0.84)^2\, q}{\delta^2},
$$

so $\delta = 0.02$ and $q = 0.10$ need about 1,960 items, and $\delta = 0.05$ needs about 314. When items come in groups (several questions per passage, several turns per conversation), they are not independent; with $m$ items per group and within-group correlation $\rho$ the design effect is $1 + (m - 1)\rho$ and the effective $n$ is the nominal $n$ divided by it. Resample groups, not items, in the bootstrap.

Sampling at temperature above zero adds a second source of variance, the model's own randomness, on top of the item sampling. Repeat the run and report both, or fix temperature to zero and note that batch composition can still change floating-point results.

### LLM-as-judge and its biases

When no reference answer exists, a stronger model reads the question and one or two answers and produces a score or a preference. Treat it as an instrument with known defects. Position bias: in pairwise comparison the judge prefers the first answer. Verbosity bias: it prefers the longer one. Self-preference: it prefers text in its own style, including its own outputs. Format bias: headings and lists read as quality. Each has a mitigation. Present every pair in both orders and count as a tie any pair where the verdict flips; regress the preference on the length difference and report the length-controlled win rate; use a judge from a different family than either model under test; give the judge a rubric and, where possible, a reference answer, so that it grades against criteria rather than impressions.

Then calibrate. Label a sample yourself (100 items is a start), compute the judge's agreement with you as Cohen's kappa,

$$
\kappa = \frac{p_o - p_e}{1 - p_e},
$$

where $p_o$ is the observed agreement rate and $p_e$ the agreement expected by chance from the two marginals, and report it with the judge's sensitivity $s$ (fraction of your passes it calls pass) and specificity $t$ (fraction of your fails it calls fail). If the judge's errors were random, you could correct its observed pass rate $\hat{r}$ to a true rate by

$$
\hat{\mu} = \frac{\hat{r} - (1 - t)}{s - (1 - t)},
$$

and this formula is worth knowing mostly because it shows how much a mediocre judge distorts: with $s = 0.9$ and $t = 0.8$ an observed 0.62 is a true 0.60, but if $s$ and $t$ differ between the two models being compared, because one writes longer or in the judge's style, no single correction applies and the comparison is biased in a direction the agreement rate does not reveal.

### Agentic and tool-use evals

An agent eval runs the model in a loop against an environment with tools, and the unit is an episode, not an answer. Grade the final state of the environment (the file exists with the right contents, the calendar has the meeting, the database row was updated) and never the transcript, since a model that narrates success is not the same as one that achieved it; test the grader by running a trivially wrong agent through it and confirming it scores zero. Because agents are stochastic over many steps, report both pass@$k$, the chance that at least one of $k$ trials succeeds, and pass$^k$, the chance that all $k$ trials succeed, which is the number a deployment cares about; with independent trials at per-trial success $p$ these are $1 - (1 - p)^k$ and $p^k$, and for $p = 0.8$ and $k = 3$ they are 0.99 and 0.51. Report cost alongside success: tokens, tool calls, wall-clock, and dollars per solved task, since an agent that solves 5 percent more tasks at three times the cost is a different product. Report safety as counts of specific events: irreversible actions taken without permission, tool calls outside the allowed set, and the success rate of injected instructions in tool outputs, which the Nemotron indirect-prompt-injection data from the SFT work of Lab 04 gives you a ready source of. Fix seeds, mock external tools so that the environment is deterministic, and version the environment with the eval.

### A private eval from your own tasks

Collect 100 to 300 tasks from your own work: questions you have asked the Cortex chat, paper-summary requests with the paper attached, coding tasks with tests, derivations with a known result. For each, write the checkable condition and classify it: exact answer, regex-extractable answer, unit test, or judge with rubric. Split off a fresh set of about a fifth, written later or by someone else, for the contamination comparison. Freeze a version, store it off the public internet, and never tune on it; a private eval that leaks into a training mix stops being private. After the first run, do item analysis: an item every model gets right or wrong has zero discriminative power, and an item whose correctness correlates negatively with total score is probably mislabeled. Re-label 30 items yourself after a month and compute your own agreement with yourself; that is the ceiling for any judge.

## Build it small

The snippet implements log-likelihood multiple-choice scoring exactly as a harness does (tokenize the joined string, split by character offset, sum continuation log-probabilities), computes `acc` and `acc_norm` on a twelve-item private eval, and puts a normal and a bootstrap confidence interval around each. It downloads a 135M-parameter model the first time it runs.

```python
# Log-likelihood multiple-choice scoring (acc and acc_norm) plus a bootstrap CI, with a 135M model (CPU, ~1 min)
import math, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
random.seed(0)
name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name).eval()

items = [  # (prompt, choices, index of the correct choice); tiny private eval
    ("Question: What is the capital of France?\nAnswer:", [" Paris", " Berlin", " Madrid", " Rome"], 0),
    ("Question: How many legs does a spider have?\nAnswer:", [" six", " eight", " four", " ten"], 1),
    ("Question: What gas do plants absorb from the air?\nAnswer:", [" oxygen", " nitrogen", " carbon dioxide", " helium"], 2),
    ("Question: Which planet is closest to the Sun?\nAnswer:", [" Venus", " Earth", " Mars", " Mercury"], 3),
    ("Question: What is the boiling point of water in Celsius?\nAnswer:", [" 100 degrees", " 50 degrees", " 0 degrees", " 212 degrees"], 0),
    ("Question: Who wrote Hamlet?\nAnswer:", [" Charles Dickens", " William Shakespeare", " Jane Austen", " Mark Twain"], 1),
    ("Question: What is the largest ocean on Earth?\nAnswer:", [" Atlantic", " Indian", " Pacific", " Arctic"], 2),
    ("Question: What is 7 times 8?\nAnswer:", [" 54", " 48", " 63", " 56"], 3),
    ("Question: What is the chemical symbol for gold?\nAnswer:", [" Au", " Ag", " Gd", " Go"], 0),
    ("Question: How many days are in a leap year?\nAnswer:", [" 365", " 366", " 364", " 360"], 1),
    ("Question: What organ pumps blood through the body?\nAnswer:", [" liver", " lung", " heart", " kidney"], 2),
    ("Question: What is the square root of 81?\nAnswer:", [" 7", " 8", " 6", " 9"], 3),
]

@torch.no_grad()
def continuation_logprob(prompt, cont):
    """Sum of log p(cont tokens | prompt), tokenizing the joined string and splitting by character offset."""
    enc = tok(prompt + cont, return_offsets_mapping=True)
    ids = torch.tensor([enc.input_ids])
    first = next(i for i, (s, e) in enumerate(enc.offset_mapping) if e > len(prompt))  # first token of cont
    logp = model(ids).logits[0, :-1].log_softmax(-1)            # position t predicts token t+1
    tgt = ids[0, 1:]
    per_tok = logp[torch.arange(len(tgt)), tgt]
    return per_tok[first - 1:].sum().item()

acc, acc_norm = [], []
for prompt, choices, gold in items:
    lp = [continuation_logprob(prompt, c) for c in choices]
    acc.append(int(max(range(4), key=lambda i: lp[i]) == gold))
    acc_norm.append(int(max(range(4), key=lambda i: lp[i] / len(choices[i].encode())) == gold))  # byte-normalized

def ci(x, reps=2000):  # percentile bootstrap over items
    means = sorted(sum(random.choices(x, k=len(x))) / len(x) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]

n = len(items)
for label, x in [("acc", acc), ("acc_norm", acc_norm)]:
    p = sum(x) / n
    se = math.sqrt(p * (1 - p) / n)
    lo, hi = ci(x)
    print(f"{label:9s} {p:.3f}  normal 95% CI +/- {1.96*se:.3f}   bootstrap 95% CI [{lo:.3f}, {hi:.3f}]   n={n}")
```

Expected output (CPU; the accuracy depends on the exact model weights, the intervals depend only on $n$):

```
acc       0.750  normal 95% CI +/- 0.245   bootstrap 95% CI [0.500, 1.000]   n=12
acc_norm  0.750  normal 95% CI +/- 0.245   bootstrap 95% CI [0.500, 1.000]   n=12
```

Read it against the math. The model gets nine of twelve, and the interval says the true accuracy is somewhere between one half and one; twelve items tell you almost nothing, which is the point. The normal interval is symmetric and would extend past 1.0 at higher accuracies, which is why the bootstrap (or a Wilson interval) is the right tool at small $n$. Here `acc` and `acc_norm` coincide, but change the wrong choices to long phrases and watch `acc` fall while `acc_norm` holds, because the raw sum penalizes length. Two details in `continuation_logprob` are the ones people get wrong: the joined string is tokenized once and split by character offset rather than tokenizing prompt and continuation separately (tokenization is not compositional at the boundary), and the log-probability of the first continuation token is read at position `first - 1`, since the logits at position $t$ predict token $t + 1$.

## Build it real

The recipe is `recipes/eval_suite.py`. It evaluates a base model and its Lab 04 SFT checkpoint on a fixed public task list through lm-eval-harness, on your private eval with generative scoring, and optionally with a local judge, and writes one table.

Models. `--models base=<hf-id-or-path> sft=<path>`; a LoRA adapter directory is merged into a temporary full checkpoint before evaluation so that both runs use identical serving code (`--no-merge` instead passes the adapter to vLLM's LoRA support). Both models are served through vLLM in bf16; an 8B model is 16 GB of weights and leaves room for the KV cache in 32 GB with `gpu_memory_utilization=0.85` and `max_model_len=4096`.

Public tasks. The script calls `lm_eval` with `--model vllm`, the pinned task list in `--tasks` (a default of a knowledge task, a reasoning task, a math task, and an instruction-following task, at the few-shot counts in `--fewshot`), `--batch_size auto`, `--log_samples` so that per-item results are saved for the paired analysis, and, for the SFT model, `--apply_chat_template --fewshot_as_multiturn` so that the shots are presented as prior turns rather than pasted into one user message. It records the harness version, the task versions, and the exact template in the output directory, because those are part of the measurement.

Private eval. `--private tasks.jsonl` reads items of the form `{"id", "prompt", "answer", "check": "exact" | "regex" | "python" | "judge", "pattern", "rubric", "group"}`. Generation runs at `--temperature 0` with `--max-new-tokens 512`, or with `--samples-per-item n` at a stated temperature for pass@$k$; `python` checks execute the item's test against the extracted answer in a subprocess with a timeout; `judge` items go to the judge stage. The parse-failure rate is reported as its own column. `--fresh fresh.jsonl` scores the held-out matched set for the contamination comparison.

Judge. `--judge <model>` serves a second, larger instruct model from a different family through vLLM, presents each judge item pairwise (base versus SFT) in both orders with the rubric and the reference answer when one exists, counts flipped verdicts as ties, and reports the raw and length-controlled win rate. `--calibrate labels.jsonl` compares the judge with your own labels and prints kappa, sensitivity, and specificity.

Contamination. `--contamination` runs the 13-gram overlap of every public and private item against the SFT training files from Lab 04, and the exchangeability test on the private set with $R = 200$ permutations for each model, printing the $p$-value.

Output. `results/<timestamp>/table.md` with one row per metric: base, SFT, paired difference, 95 percent paired-bootstrap interval (group-resampled when items carry a `group`), $n$, and parse-failure rate; plus the per-item JSONL that produced it.

Arguments: `--models`, `--tasks`, `--fewshot`, `--private`, `--fresh`, `--judge`, `--calibrate`, `--contamination`, `--temperature`, `--max-new-tokens`, `--samples-per-item`, `--seed`, `--out`, and `--no-merge`.

What to watch in the logs. The harness prints per-task accuracy with its own standard error; if the SFT model's log-likelihood scores drop while its generative scores rise, that is the chat-template effect described in How it goes wrong, not a regression. vLLM warns when a prompt exceeds `max_model_len` and truncates; any such warning on a few-shot task means the reported number is on a different prompt than you think. In the private eval, the parse-failure rate is the first column to read; above a few percent, fix the parser or the format instruction before believing the accuracy. In the judge stage, the flip rate (fraction of pairs whose verdict changed with order) is the position-bias measurement; a high flip rate means the judge is guessing on those items.

How long it takes. Generation time is about (items times output tokens) divided by decode throughput. Assume 300 private items at 512 tokens, 154k generated tokens, and an assumed 2,000 tokens per second of batched bf16 decode for an 8B model under vLLM on the 5090: about 80 seconds per model. Log-likelihood tasks are prefill-bound: assume a 14,000-item knowledge benchmark with 4 candidates and 5-shot prompts of about 600 tokens, with prefix caching so the shared prompt is processed once per item; that is roughly $14{,}000 \times 700 \approx 10^7$ tokens, times $2P = 1.6 \times 10^{10}$ FLOPs per token for an 8B model, about $1.6 \times 10^{17}$ FLOPs, which at an assumed sustained 200 TFLOP/s is around 13 minutes per model. Budget under an hour for the full suite on both models, plus the judge stage, which is another generation pass of similar size.

## How it goes wrong

The SFT model scores below the base model on every log-likelihood task, but answers questions better in conversation. The harness fed the SFT model bare text without its chat template, so the continuation probabilities are being read from a model that expects turn markers, and SFT has also concentrated probability on chat-style responses rather than terse continuations. Run with the chat template and few-shot-as-multiturn, and report generative scores alongside; if the log-likelihood gap remains, it may be real, and the generative number is the one that reflects use.

A large gain on a math benchmark after SFT. Before celebrating, run the overlap check: the training mixture very often contains that benchmark's training split, whose items are near-paraphrases of its test split, and sometimes the test split itself via a dataset that bundled it. Report the score on the clean subset and on the fresh private set; if the gain vanishes there, it was contamination.

The same model scores five points apart on two machines. Different harness versions, task versions, or prompt templates; or a different `max_model_len` truncating the shots; or the letter-scoring versus full-continuation convention. Pin the versions, save the exact prompts, and diff them before comparing numbers.

The judge prefers the SFT model 70 to 30, and 55 to 45 when you swap the order. Position bias was doing most of the work. Use both orders for every pair, count flips as ties, and report the flip rate; if it is high, the judge cannot see a difference and the honest result is a tie.

The private generative eval gives a strong model a score near zero. The parser expects "The answer is 42" and the model wrote "**Answer:** 42" or reasoned past the token limit. Read the parse-failure column, look at twenty raw outputs, and fix the extraction or the format instruction; never fix it by tuning the model's output to the parser.

Two runs of the same eval disagree by a point or two at temperature zero. Batched inference changes floating-point summation order, and some kernels are nondeterministic, so greedy decoding is not bit-reproducible across batch compositions. This variance is small compared with item-sampling variance, but it exists; report it from repeated runs, and never read a difference smaller than it.

An agent eval reports high success, and a deliberately broken agent also scores well. The grader read the transcript for phrases like "done" instead of checking the environment state. Grade final state only, and keep the broken-agent check as a permanent test of the grader.

The private eval stops discriminating after two training iterations. The items were too easy, or the eval leaked into the training data through a shared directory. Do item analysis and replace saturated items with harder ones from your recent work; check the overlap of the eval file against every training file; keep the fresh split fresh by writing new items each quarter.

## Measure it

Measure the eval before you measure the model. Report, for the suite itself: the minimal detectable paired difference at your $n$ and observed disagreement rate (from the sample-size formula); the parse-failure rate per generative task; the judge's kappa, sensitivity, and specificity against your labels, and its flip rate; the exchangeability $p$-value and the overlap fraction from the contamination stage; and the item-discrimination distribution, with the count of items every model got right or wrong. A suite is in good shape when the detectable difference is smaller than the effect you care about (a 2-point improvement needs roughly two thousand items; a 5-point one a few hundred), when parse failures are at most a few percent, when the judge's kappa is above roughly 0.6 (the level conventionally described as substantial agreement) and its flip rate is low, when the contamination tests are not significant, and when fewer than a fifth of the items are saturated.

Then measure the model as a difference, not a number: base versus SFT, paired, with the interval, on public tasks and on the private eval separately. Good means the private-eval difference is positive with an interval that excludes zero, the public tasks show no regression outside their intervals (an SFT that helps your tasks and costs general ability is a decision, not a free win), and the fresh split agrees with the main private split. Keep the table; the next SFT run gets compared against it with the same items and the same template.

## Exercises

1. Compute the number of items needed for a $\pm 1.5$-point 95 percent interval at an accuracy of 0.8. Answer: $1.96^2 \times 0.16 / 0.015^2 \approx 2{,}731$.
2. Two models score 0.70 and 0.73 on 1,000 shared items and disagree on 12 percent. Compute the unpaired and paired standard errors of the difference and say which design finds it significant. Answer: unpaired 0.020 (not significant), paired 0.011 (significant); the derivation is in The math.
3. Fine-tune SmolLM2-135M for a few epochs on a 200-item question file in its canonical order, then run the exchangeability test with 100 permutations on the fine-tuned and the base model. Check: the canonical order is the most likely ordering under the fine-tuned model and unremarkable under the base model.
4. Write the lm-eval-harness task YAML for your private eval (a `generate_until` task over a local JSON file with a regex filter and `exact_match`), run it, and compare per-item results with the recipe's own scorer. Check: identical per-item verdicts; any disagreement is a parsing difference you should understand.
5. Label 50 judge items yourself. Suppose the judge agrees with you on 42, calls 30 items pass while you call 28 pass. Compute kappa. Answer: $p_o = 0.84$, $p_e = 0.6 \times 0.56 + 0.4 \times 0.44 = 0.512$, $\kappa = 0.672$.
6. A judge has sensitivity 0.9 and specificity 0.8 against your labels and reports a pass rate of 0.62. Estimate the true pass rate, then explain what assumption makes the estimate untrustworthy when comparing two models. Answer: $(0.62 - 0.2)/(0.9 - 0.2) = 0.60$; the assumption is that sensitivity and specificity are the same for both models' outputs, which fails when one model's style triggers the judge's biases more.

## Test yourself

1. The same multiple-choice set gives a model 0.75 under log-likelihood scoring and 0.40 under generative scoring. Name three mechanisms that produce the gap and say which number you would report for a deployment decision.

<details><summary>Answer</summary>
Log-likelihood scoring never requires the model to produce the answer; it reads probabilities of given candidates. Generative scoring fails when the model refuses or hedges, when it produces the right content in a format the parser does not extract, or when it runs past the token limit while reasoning. The gap can also come from the model preferring the correct candidate only relative to three wrong ones while its unconstrained output is something else entirely. For deployment, the generative number under the deployment prompt is the behaviour users will see, with the parse-failure rate reported next to it so that a parser problem is not mistaken for a model problem.
</details>

2. What does byte-length normalization in `acc_norm` correct, and construct a case where it makes the score worse.

<details><summary>Answer</summary>
Summed log-probabilities decrease with length, so a longer correct answer loses to a shorter wrong one under raw scoring; dividing by bytes compares per-byte likelihood instead. It hurts when the longer candidate is genuinely improbable and its per-byte likelihood is inflated by predictable filler: a candidate like " the Pacific Ocean, which is the largest" gains from the easy tokens after the first, while a single-token wrong candidate has no such cushion. Normalization trades one bias for another; report both and use the one that better matches the task's answer format.
</details>

3. Spot the bug: `ids = tok(prompt).input_ids + tok(cont, add_special_tokens=False).input_ids`, then sum the log-probabilities of the continuation ids from position `len(tok(prompt).input_ids)` onward.

<details><summary>Answer</summary>
Two bugs. Tokenization is not compositional: the tokens of `prompt + cont` around the boundary can differ from the tokens of each piece, so the model is scored on a token sequence it would never see for that string (and a leading space on the continuation may become a different token). And the log-probability of the first continuation token lives at the logits of the last prompt position, so the slice must start one position earlier than the continuation's first index. The snippet in Build it small tokenizes once and splits by character offset, and reads from `first - 1`.
</details>

4. A leaderboard shows model A at 71.2 and model B at 70.4 on the same 1,000 items, without per-item results. Can you conclude A is better, and what would you need?

<details><summary>Answer</summary>
No. Each score has a standard error near $\sqrt{0.71 \times 0.29 / 1000} \approx 1.4$ points, and the unpaired standard error of the difference is about 2.0 points, so a 0.8-point gap is well inside noise. With per-item results you could do the paired analysis; if the two models disagree on, say, 10 percent of items, the paired standard error is about 1.0 point, and the gap is still not significant. You would need either many more items or a larger effect.
</details>

5. A senior researcher says greedy decoding makes the eval deterministic, so a single run is enough and the reported number has no variance. What is wrong?

<details><summary>Answer</summary>
Three things. Greedy decoding removes sampling randomness but not floating-point nondeterminism from batched kernels, so repeated runs can still differ slightly. Determinism of the run says nothing about item-sampling variance: the benchmark is a sample from a task distribution, and the interval from the Bernoulli formula applies regardless of how the outputs were produced. And the prompt template is a hidden variable; a different but equally reasonable template gives a different deterministic number. One run gives you one point estimate with an interval you must still compute.
</details>

6. A judge agrees with human labels 85 percent of the time. A colleague concludes that its win rates are accurate to within 15 points. Why is this wrong in both directions?

<details><summary>Answer</summary>
Agreement rate treats errors as if they were random. Judge errors are systematic (length, position, style), so on a comparison where one model's outputs trigger the bias more, the judge's error can be almost entirely in one direction and shift a win rate by far more than 15 points; conversely, if the biases affect both models equally, the win rate can be accurate to within a point or two despite 15 percent disagreement. What you need is sensitivity and specificity per model, or a length-controlled and order-balanced protocol whose remaining error you have measured against your own labels.
</details>

7. The exchangeability test shows a model has seen your benchmark file ($p < 0.01$), but its accuracy on the benchmark equals its accuracy on a matched fresh set. Is the benchmark number invalid?

<details><summary>Answer</summary>
The test detects exposure, and the matched set detects benefit. Exposure without measurable benefit means the model memorized the ordering but did not learn the answers from it (common for a single pass over a file in a large mixture), and the benchmark number remains a fair estimate of capability, though you should say that the file was seen. The situation to worry about is the reverse: no detectable exposure with a suspicious gain, because paraphrased contamination defeats the ordering test and only the matched set catches it.
</details>

8. Why is $1 - (1 - c/n)^k$ a biased estimator of pass@$k$, in which direction, and what does the combinatorial estimator compute instead?

<details><summary>Answer</summary>
Because $\hat{p} = c/n$ is a noisy estimate and $f(p) = 1 - (1 - p)^k$ is concave in $p$ for $k \ge 1$, so by Jensen's inequality $\mathbb{E}[f(\hat{p})] \le f(p)$: the plug-in underestimates pass@$k$, and the bias is largest at small $n$. A check: with $n = k = 2$ and $p = 0.5$ the true pass@2 is $0.75$, but $\hat{p}$ takes the values $0, 0.5, 1$ with probabilities $0.25, 0.5, 0.25$, so the plug-in's expectation is $0.5 \times 0.75 + 0.25 \times 1 = 0.625$. The combinatorial estimator $1 - \binom{n - c}{k}/\binom{n}{k}$ is the exact probability that a uniformly chosen subset of $k$ of the $n$ samples contains at least one pass; in the same check it gives $0.25 \times 0 + 0.5 \times 1 + 0.25 \times 1 = 0.75$, which is unbiased, as it is in general for $n \ge k$.
</details>

9. An agent succeeds on 80 percent of independent trials of a task. Compute pass@3 and pass$^3$, and say which one a customer experiences.

<details><summary>Answer</summary>
pass@3 $= 1 - 0.2^3 = 0.992$; pass$^3 = 0.8^3 = 0.512$. A customer who runs the agent once experiences the per-trial rate; a customer who needs it to work every time over three uses experiences pass$^3$. Benchmarks that report only pass@$k$ reward variance; consistency is what deployment requires.
</details>

10. Your private eval has 200 passages with five questions each, and the within-passage correlation of correctness is 0.3. What is the effective sample size, and how do you build the bootstrap?

<details><summary>Answer</summary>
Design effect $1 + (m - 1)\rho = 1 + 4 \times 0.3 = 2.2$, so the 1,000 nominal items are worth about 455 independent ones, and intervals computed as if $n = 1{,}000$ are too narrow by a factor of about $\sqrt{2.2} \approx 1.5$. Resample passages with replacement (all five questions travel together), recompute the paired difference on each resample, and take the percentile interval.
</details>

## What will change, what will not

The statistics will not change. A benchmark score is a sample mean with a standard error of order $1/\sqrt{n}$; comparing two models on the same items is a paired design whose power depends on the disagreement rate; grouped items need group-level resampling; a judge is an instrument with a sensitivity and a specificity that you measure against your own labels. Every future eval, whatever it is called, is read through these.

The distinction between exposure and benefit in contamination, and the pairing of a public benchmark with a matched fresh set, will outlast any particular detection method. The $n$-gram overlap check assumes you have the training data; the exchangeability test assumes a fixed file order; both assumptions will be weakened by paraphrased synthetic data, and the matched set will remain the arbiter.

The capability-versus-behaviour distinction will matter more, not less, as models are deployed as agents. State-based grading, consistency over repeated trials, cost per solved task, and counts of unsafe actions were the right units at the time of writing and are the right units in general; what will change is the environments, the tool sets, and the specific safety events that matter.

What will change is the tooling and the benchmarks. lm-eval-harness's flags, vLLM's memory arguments, the LoRA-merging step, the named public tasks (several of the well-known ones were already saturated or contaminated when this was written and will be retired), and the particular judge model are all replaceable, and LLM-as-judge itself may give way to trained verifiers and reward models for many task types. The private eval you build from your own work is the part that keeps its value, because its questions are yours, its scoring rule is yours, and no training corpus contains it.

## Read next

1. Language Models are Few-Shot Learners, Brown, 2020. Establishes log-likelihood few-shot evaluation at scale and the 13-gram overlap analysis of contamination, with clean-subset reporting.
2. Calibrate Before Use: Improving Few-Shot Performance of Language Models, Zhao, 2021. Names majority-label and recency bias in few-shot prompts and introduces contextual calibration.
3. Evaluating Large Language Models Trained on Code, Chen, 2021. The unbiased pass@$k$ estimator and the case for execution-based scoring.
4. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, Zheng, 2023. Position, verbosity, and self-enhancement biases measured, with the swap-order mitigation.
5. Proving Test Set Contamination in Black Box Language Models, Oren, 2023. The exchangeability test that turns a benchmark's ordering into a $p$-value for exposure.
6. tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains, Yao, 2024. State-based grading of agent episodes and the pass$^k$ consistency metric.
7. Lessons from the Trenches on Reproducible Evaluation of Language Models, Biderman, 2024. The lm-eval-harness authors on why versions, templates, and scoring conventions must be pinned and reported.
8. Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations, Miller, 2024. Paired differences, clustered standard errors, and power analysis applied to model evaluation.
