---
title: "Lab 24: Building RL environments and evaluation suites for language models"
kind: permanent
topics: [lab]
chapter: 24
station: posttrain
recipe: recipes/grpo_tool.py
reading_time: 55 min
---

## What you will be able to do

- Write an environment for a language model as four functions (reset, step, reward, done) over a text-and-tools state, and say why the environment, not the algorithm, decides what the model learns.
- Design a reward that is verifiable, partially credited where that is safe, and robust to the three standard hacks (format, length, and tool-spam), and measure the hacking rate.
- Build task families with procedural generation so the training set is unbounded and the held-out set is genuinely held out.
- Turn the same task families into an evaluation suite with contamination checks, paired comparisons, and confidence intervals, and explain why an eval is an environment with the learning turned off.
- Report an RL result the way a reviewer will believe: seeds, intervals, the reward curve and the task-metric curve on the same axis, and the hacking audit.

## The idea in one paragraph

Reinforcement learning for language models is mostly environment engineering. GRPO and PPO are a few hundred lines that have not changed much; what changed between a model that reasons and one that does not is the set of tasks it was trained on and the reward that graded them. An environment is a generator of tasks plus a grader: `reset()` samples a task and returns the prompt, `step(action)` appends the model's text or runs its tool call and returns the next observation, `reward()` grades the episode, `done()` says when to stop. An evaluation suite is the same object with a frozen task list and no gradient. The Lab's tool station (Lab 06) is an environment of exactly this shape; this chapter is about designing new ones and about the discipline that makes their numbers mean something.

## The math

### The environment interface

State $s_t$ is the transcript so far plus any tool state (a filesystem, a database, a game board). The policy emits $a_t$, a string; the environment parses it (a final answer, a tool call, or free text) and returns $s_{t+1}$ and, at the end, $R \in \mathbb{R}$. For single-turn tasks $T = 1$ and the whole design is the grader. For agentic tasks $T > 1$, and the interesting part is what the environment does with malformed actions: reject with a message (teaches format), terminate with zero (teaches caution), or execute anyway (teaches nothing). Pick one and log which branch fired.

### Verifiable rewards and partial credit

A verifiable reward is a program: $R = \mathbb{1}[\text{answer} = \text{reference}]$ for maths, $R = \tfrac{\#\text{tests passed}}{\#\text{tests}}$ for code, $R = \mathbb{1}[\text{state satisfies goal}]$ for tools. Binary rewards are safe and sparse; partial credit densifies them and is where hacking enters. The rule: partial credit is safe when the partial score is monotone in the true quality *and* cannot be raised without raising the true quality. Tests passed is safe if the tests are hidden and diverse; "number of correct digits" is not safe because the model learns to emit many digits. Add a format term only as a gate ($R = 0$ if unparsable), never as a bonus, or the model will learn the format and stop there.

GRPO's advantage is group-relative: for $G$ samples of one prompt,

$$A_i = \frac{R_i - \operatorname{mean}(R)}{\operatorname{std}(R) + \epsilon},$$

so a prompt where every sample scores the same gives zero gradient. That is the reason task difficulty must be tuned to the model: the useful prompts are the ones with $p(\text{solve}) \in (0.1, 0.9)$. Procedural generation with a difficulty knob is what keeps prompts in that band as the model improves; a fixed dataset drifts out of it.

### The three hacks and their audits

*Format hacking*: the model finds the parser's blind spot (two answers, an answer in the reasoning, a code block that prints the expected output without computing it). Audit: sample 50 rewarded episodes and check by hand; the hacking rate is the fraction that were rewarded for the wrong reason. *Length hacking*: if any term grows with length (partial credit for steps, a judge that likes thoroughness), responses grow until the context is full. Audit: mean length per training step on the same plot as reward. *Tool spam*: if a tool call is free, the model calls tools instead of thinking. Charge for them: a small negative per call or a budget in `done()`.

### From environment to evaluation suite

Freeze a task list; run every checkpoint on it with the same decoding settings; report the task metric, not the reward, with an interval. For binary outcomes over $n$ tasks the Wilson interval at $95\%$ is

$$\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n} \quad \text{(approximately, with the Wilson centre correction for small } n\text{)},$$

so at $n = 100$ and $\hat p = 0.6$ the half-width is about $0.096$: two checkpoints must differ by ten points to be told apart on one hundred tasks. Paired comparisons shrink that: on the same tasks, count wins, losses and ties and use a sign test or bootstrap the paired difference. Contamination: hash every training prompt (and its n-grams) and check the suite against the hash set before you trust a number; procedural tasks make this trivial because held-out seeds are held out by construction.

### Seeds and power

RL runs are noisy: the same recipe with three seeds can spread by five points. Report the mean and the spread across seeds, never one run. Before a comparison, decide the effect you care about (say three points), estimate the per-seed standard deviation from a pilot, and compute how many seeds and tasks you need; if the answer is more than you can afford, change the question. Write the success criterion down before the run (the pre-registration note in the Traces tab) so the analysis cannot drift toward the result.

## Build it small

`grpo_tool.py` trains a tool-using policy on the nano model against a calculator-and-lookup environment. Open the recipe: `reset`, `step`, `reward`, `done` are there. Add a task family (unit conversion, date arithmetic, or a two-hop lookup) with a difficulty knob and a held-out seed range. Train 300 steps, three seeds. Plot reward and task accuracy on the same axis; if reward rises and accuracy does not, you have a hack, and the 50-episode audit tells you which.

## Build it real

On the 5090 with the 1.5B from Lab 16 or a Qwen-class base: an environment for a real workflow (a small repo with tests, a spreadsheet with questions, a mock API). Generate tasks procedurally; hide the tests; charge for tool calls; cap episodes. Train with GRPO for a day; evaluate on a frozen suite of 200 with paired bootstrap against the base. For the eval side: `eval_suite.py` takes a task file and a checkpoint and prints the metric with intervals; add your suite as a task file and run it on every checkpoint in the pipeline (Lab 21's `eval` stage).

## How it goes wrong

- Reward climbs, accuracy flat: hacking; audit and gate the format.
- Zero gradient for most prompts: difficulty out of band; adjust the knob or filter prompts by pilot solve rate.
- Great on the suite, bad in the product: contamination or a suite that does not sample the product's distribution.
- Results do not replicate: one seed; report three.
- Tool calls explode: free tools; charge.

## Measure it

- Solve rate per difficulty level, per checkpoint, with intervals.
- Hacking rate from a 50-episode hand audit.
- Mean response length and tool calls per episode on the reward plot.
- Fraction of prompts with non-zero group advantage (the useful fraction).
- Paired win rate between checkpoints with a bootstrap interval; seeds reported.

## Exercises

1. Add a task family with a difficulty knob to `grpo_tool.py` and show the useful-prompt fraction as a function of the knob.
2. Introduce a deliberate parser blind spot, train, and measure how fast the model finds it.
3. Replace the binary reward with partial credit that is not monotone; document the hack that appears.
4. Build a 200-task frozen suite from held-out seeds and run three checkpoints with paired bootstrap.
5. Write a pre-registration note for a comparison, run it, and write the post-mortem.

## Test yourself

1. What are the four functions of an environment, and where does the design effort concentrate?
2. When is partial credit safe?
3. Why does GRPO give zero gradient on some prompts, and what does that imply for task generation?
4. Name the three standard reward hacks and one audit for each.
5. Why should format be a gate and not a bonus?
6. How wide is the 95 percent interval for a 60 percent solve rate on 100 tasks, and what does a paired comparison do about it?
7. What is contamination and why do procedural tasks make the check easy?
8. Why report three seeds?
9. What is an evaluation suite in terms of an environment?
10. What does charging for tool calls change in the learned policy?

<details><summary>Answers</summary>

1. reset, step, reward, done; the effort is in the task generator and the grader.
2. When the partial score is monotone in true quality and cannot be raised without raising true quality (hidden, diverse tests: yes; digit counts: no).
3. The advantage is normalised within the group, so a prompt where all samples score the same has zero advantage; keep prompts in the band where solve probability is between roughly 0.1 and 0.9 with a difficulty knob.
4. Format hacking (hand audit of 50 rewarded episodes), length hacking (length on the reward plot), tool spam (calls per episode; charge for them).
5. A bonus can be collected without solving the task; a gate only removes reward from unparsable outputs.
6. About plus or minus 10 points; pairing on the same tasks removes task variance so smaller differences become detectable.
7. Suite items that appeared in training; with procedural generation, held-out seeds cannot have been trained on by construction, and a hash check confirms it.
8. RL variance across seeds is often as large as the effect; one run is an anecdote.
9. The same environment with a frozen task list, fixed decoding, and no gradient.
10. It removes the free action; the policy learns to think when a call is not worth its cost and to call when it is.

</details>

## What will change, what will not

Environments will get richer (browsers, repos, multi-agent), rewards will come more often from models than from programs, and the algorithms will keep their names. What will not change: the model learns exactly what the reward pays for, the useful prompts are the ones the model half-solves, an eval is an environment with learning off, and a number without seeds and an interval is not a result.

## Read next

- Shao et al., "DeepSeekMath" (2024), for GRPO and the verifiable-reward setup.
- Lambert et al., "Tülu 3" (2024), for RL with verifiable rewards at scale.
- The Prime Intellect `verifiers` library and OpenAI's `gym`-style agent environments, for interface conventions.
- Skalse et al., "Defining and characterizing reward hacking" (2022).
- Lab 09 for the statistics, Lab 06 for the tool station this chapter extends.
