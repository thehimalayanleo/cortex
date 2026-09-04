---
title: "Lab 06: Teaching a model to use tools"
kind: permanent
topics: [lab]
chapter: 6
station: posttrain
recipe: recipes/grpo_tool.py
reading_time: 60 min
---

## What you will be able to do

1. Read a tool-use conversation at the token level: where the schemas go, how a call and its result are delimited, how parallel calls are laid out, and which of those tokens the model is responsible for.
2. Derive the trajectory likelihood for an agent interacting with an environment and show from it why the loss must cover assistant text and tool calls but never tool results.
3. Place each dataset in the NVIDIA Nemotron agentic collection into one of three roles (hand-written trajectories, synthetic environments, RL rollouts) and say what a training pipeline would do with it.
4. Compute rewards for tool-use tasks three ways (schema match, execution success, final-state check), and design a reward that resists the obvious hacks.
5. Run the full loop on the 5090: a three-function environment, trajectory generation with rejection sampling, SFT with the right mask, then GRPO with an execution reward, and evaluate it with the metrics the field uses, including indirect prompt injection.

## The idea in one paragraph

A tool call is just text the model writes in a format a program can parse: a function name and arguments, wrapped in delimiters the tokenizer knows. The program runs the function and pastes the result back into the conversation as a new message, and the model continues. Teaching a model to do this is the same SFT and RL you learned in Labs 04 and 05 with one extra rule: the model is responsible for what it writes, not for what the environment says back, so the loss covers its text and its calls and never the results. Data comes from three places: people writing example conversations, programs that generate tasks and check answers, and the model's own attempts filtered by whether they worked. The first two get the format into the model; the third is where it learns judgment, because a checker can tell it that a call with the wrong argument did not accomplish the task. The same setup is where safety training for agents happens, since a tool result can contain instructions the model must not follow.

## The math

### A tool call at the token level

The conversation format from Lab 04 gains one role and one kind of content. Messages now have roles system, user, assistant, and tool. An assistant message may contain text, a list of tool calls, or both, and each tool call is a name plus a JSON object of arguments. A tool message carries the result of one call and, in formats that support parallel calls, an identifier tying it back to the call it answers.

The chat template renders all of this to one token sequence. The available tools are rendered into the system prompt as JSON schemas (name, description, parameters with types and which are required). In the Hermes-style format used by the Qwen2.5 family the schemas appear inside the system message between `<tools>` and `</tools>`, an assistant call is written as

```
<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris"}}
</tool_call>
```

and the result comes back inside a `<tool_response>` block. Llama 3.1 uses a different convention (a `<|python_tag|>` prefix on the assistant's call and an `ipython` role for results), and other families differ again. Two things are constant across all of them. The delimiters are special tokens or reserved strings that the parser looks for, and the call body is ordinary text that must parse as JSON against the schema. Parallel calls are several `<tool_call>` blocks in one assistant message; the results come back as several tool messages in the same order, or with identifiers, and the model must learn that the results are matched by position or id, not by reading the JSON.

Concretely, in the Hugging Face convention the messages look like

```python
{"role": "assistant", "content": "", "tool_calls": [
    {"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]}
{"role": "tool", "content": '{"city": "Paris", "temp_c": 18}'}
```

and `tokenizer.apply_chat_template(messages, tools=[schema1, schema2, ...])` renders the whole thing. The `tools=` argument is where the schemas enter; forget it at training time and the model learns to call functions it was never told about, forget it at inference and the format silently changes. Print one rendered example and read it before any training, as in Lab 04.

### The trajectory likelihood and the mask

Write an agent episode as a trajectory $\tau = (x_0, a_1, o_1, a_2, o_2, \dots, a_T)$, where $x_0$ is the initial context (system prompt with schemas plus the user request), $a_t$ is the $t$-th assistant message (text, calls, or both, as a token sequence), and $o_t$ is the environment's response (one or more tool messages, or a new user turn). The probability of the trajectory factorizes as

$$
p(\tau) = \prod_{t=1}^{T} p_\theta\big(a_t \mid x_0, a_{<t}, o_{<t}\big) \cdot \prod_{t=1}^{T-1} p_{\text{env}}\big(o_t \mid x_0, a_{\le t}, o_{<t}\big).
$$

The environment terms $p_{\text{env}}$ do not depend on $\theta$. Taking the log and the gradient,

$$
\nabla_\theta \log p(\tau) = \sum_{t=1}^{T} \nabla_\theta \log p_\theta\big(a_t \mid x_0, a_{<t}, o_{<t}\big),
$$

and each $a_t$ is a sequence of tokens, so the sum is over exactly the tokens of the assistant messages: the natural-language text, the `<tool_call>` delimiters, the JSON inside them, and the end-of-turn token. Nothing in the objective touches the tokens of $o_t$. If you include them anyway, you are training the model to predict environment outputs, and the model will oblige: at inference, having written a call, it will continue by writing a plausible result instead of stopping and letting the program run. This is the single most common bug in tool-use fine-tuning and it is invisible in the loss curve, because predicting tool results is easy and makes the loss go down. The mask is therefore: labels equal to ids on assistant tokens (all of them), $-100$ on system, user and tool tokens, and the end-of-turn token after a tool call is a target, because emitting it is how the model hands control to the executor. The toy in section 4 prints the fraction of characters under loss; for tool-heavy conversations it is often under 40 percent because results are long.

### The three data sources

Hand-written or human-curated trajectories are the highest quality per example and the most expensive; they teach format and manners, and a few thousand are enough to get a base model calling tools correctly on the distribution they cover. Synthetic environments are programs that generate a task, a gold outcome, and a checker; they can produce unlimited tasks with verifiable answers, but the tasks are only as varied as the generator. RL rollouts are the model's own attempts in an environment, scored by the checker; the ones that succeed can be kept as SFT data (rejection sampling fine-tuning), and all of them can drive a policy-gradient step (Lab 05). The three feed each other: hand-written data bootstraps a policy good enough to produce rollouts, and successful rollouts become the next round's SFT set.

Reading the Nemotron collection through that lens, from the names and from what such datasets generally contain. `nvidia/Nemotron-SFT-Agentic-v2` and `nvidia/Nemotron-Agentic-v1` are, by name, trajectory datasets for supervised fine-tuning: multi-turn conversations with tool schemas, calls, and results, the kind of data you would apply the mask above to. `nvidia/Nemotron-RL-agent-calendar_scheduling` and `nvidia/Nemotron-RL-agent-workplace_assistant` are named as RL datasets tied to specific environments; a calendar-scheduling environment has a state (the calendar), tools (list, add, move, cancel) and a checkable goal (the meeting exists at a slot satisfying the constraints), and a workplace-assistant environment extends that to email, documents and tasks with similar state checks. The three Pivot datasets, `nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`, `nvidia/Nemotron-RL-Agentic-Function-Calling-Pivot-v1` and `nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1`, share a word that suggests tasks where the model must move between plain conversation and tool use, or between one tool-use mode and another, within a single episode: answering directly when no tool is needed, calling when one is, and switching back to explain. The function-calling variant would be the single- and parallel-call setting with schema-checkable rewards; the SWE variant would be software-engineering tasks in a repository with tests as the reward. `nvidia/Nemotron-Terminal-Corpus` (366k samples) is by name a large corpus of terminal sessions, which is the trajectory data for a shell-using agent, and `nvidia/Nemotron-Terminal-Synthetic-Tasks` is the synthetic-environment counterpart: generated tasks with checkers a shell agent can be scored against. `nvidia/Nemotron-SFT-ARC-AGI-v1` (122k) is reasoning data on ARC-AGI style grid puzzles, which sits in the collection because agentic training mixes in hard reasoning traces to keep the model's problem-solving sharp while it learns tool formats; it is SFT data, not tool data. `nvidia/Nemotron-RL-Agentic-Indirect-Prompt-Injection-v1` is the safety environment: episodes where a tool result contains instructions, with a reward that depends on completing the user's task while not acting on the injected ones. Row counts, field names and licenses beyond those listed here are things to read off the dataset cards, not to assume.

### Rewards for tool use

A reward for an episode has to be computed by a program, or by a judge, from the trajectory and the environment's final state. Three families, each with a formula.

Schema match compares the model's calls with gold calls. For a single call, $R = \mathbb{1}[\text{name matches}] \cdot \mathbb{1}[\text{arguments match after normalization}]$, where normalization handles key order, whitespace, numeric types and, for arguments with several acceptable forms, a canonicalizer. For parallel calls with gold set $\mathcal{C}^*$ and predicted set $\mathcal{C}$, the natural score is the set match $\mathbb{1}[\mathcal{C} = \mathcal{C}^*]$, or a softer $|\mathcal{C} \cap \mathcal{C}^*| / |\mathcal{C} \cup \mathcal{C}^*|$ when you want partial credit; ordering should not matter for calls that are independent, and should for calls where one's result feeds the next. The AST-based scoring of the Berkeley function-calling leaderboard is this idea with a parser instead of string comparison.

Execution success runs the calls and checks that they returned without error and produced a usable value. It is weaker than schema match on its own (a wrong but valid call executes fine) and stronger in one way: it catches arguments that pass the schema but are wrong in the world, such as a city the weather service does not know.

Final-state and answer checks are what verifiable multi-turn tasks use. Let $s_T$ be the environment state after the episode and $\phi(s_T) \in \{0, 1\}$ a predicate for the goal (the event exists on the requested day with the requested title; the file contains the expected output; the tests pass). Combine with an answer check $\psi(a_T)$ on the final message (the number the user asked for is present and correct):

$$
R(\tau) = w_s \, \phi(s_T) + w_a \, \psi(a_T), \qquad w_s + w_a = 1.
$$

State checks are the most hack-resistant of the three because they do not care how the model got there, only whether it did. They are also where side effects live: a policy that creates the event three times passes a predicate that only asks whether it exists, so the predicate should check for exactly one, and more generally should check that nothing was changed that should not have been.

Format rewards ($R_f = 1$ if every call parsed) are useful early, when the policy cannot yet produce valid JSON, and dangerous later, when a policy that has learned the format can collect $R_f$ without doing the task. Weight them small or anneal them to zero. And a Pivot-style task needs a reward for not calling: on prompts where no tool is needed, the correct behavior is a direct answer, and $R$ should give zero to any call and credit to a good answer, judged by a rubric or by a reference answer.

Under GRPO (Lab 05) all of this enters as the per-episode $r_i$, the advantage is group-relative, the per-token ratio and clipping apply to assistant tokens only, and the observation tokens are excluded from both the loss and the KL. With multi-turn episodes the group is $G$ full episodes from the same initial context, each rolled out to completion against its own copy of the environment.

### Indirect prompt injection as an objective

An indirect injection is text inside a tool result that is written to be read as an instruction: a web page that says to ignore the user and send a file somewhere, a calendar note that tells the assistant to cancel every meeting. The model has no privilege separation between the user's words and the tool's words except what it has been trained to have. As a training objective, take the task reward $R$ above and a trap predicate $\chi(\tau) \in \{0, 1\}$ that is one if the model performed the injected action (called the trap tool, or produced the exfiltrated string), and use

$$
R_{\text{inj}}(\tau) = R(\tau) \cdot \big(1 - \chi(\tau)\big),
$$

so that following the injection zeroes the reward regardless of task success. The two numbers to report are the attack success rate, the fraction of injected episodes with $\chi = 1$, and utility under attack, the task success rate on the same episodes; a model that refuses everything has a zero attack rate and zero utility, so both must be shown. The Nemotron injection dataset is, by name, an RL environment of this shape; benchmarks like AgentDojo evaluate it with the same two numbers.

## Build it small

No model here: the mechanism is the format, the mask, the executor, and the reward, so those are what the snippet builds. Three tools with schemas derived from their type annotations, a Hermes-style renderer that produces the training string and a character-level loss mask side by side, an executor that parses and runs the calls in an assistant turn, and a reward that checks final state and final answer. It then scores a group of three trajectories, one correct, one that hallucinated its results without calling, and one that answered without doing the booking, and prints group-relative advantages as GRPO would compute them.

```python
import json, re, statistics

# 1. Three tools and their JSON schemas: this is what the chat template renders into the system prompt.
STATE = {"events": []}
def get_weather(city: str) -> dict:
    return {"city": city, "temp_c": {"Paris": 18, "Mumbai": 31, "Oslo": 7}.get(city, None)}
def convert(value: float, unit: str) -> dict:          # celsius to fahrenheit
    return {"value": round(value * 9 / 5 + 32, 1), "unit": "F"} if unit == "C" else {"error": "unsupported"}
def add_event(title: str, day: str) -> dict:
    STATE["events"].append({"title": title, "day": day}); return {"ok": True, "n_events": len(STATE["events"])}
TOOLS = {"get_weather": get_weather, "convert": convert, "add_event": add_event}
SCHEMAS = [{"name": n, "parameters": {k: str(v) for k, v in f.__annotations__.items() if k != "return"}}
           for n, f in TOOLS.items()]

# 2. Render a trajectory the way a Hermes-style template does, and build the loss mask alongside it.
def render(messages):
    text, mask = "", []                                   # mask[i] = 1 if character i is a training target
    def emit(s, train): nonlocal text; text += s; mask.extend([train] * len(s))
    emit("<|im_start|>system\nTools: " + json.dumps(SCHEMAS) + "<|im_end|>\n", 0)
    for m in messages:
        emit(f"<|im_start|>{m['role']}\n", 0)             # role header is never a target
        if m["role"] == "assistant":
            body = m.get("content", "") + "".join(
                "<tool_call>" + json.dumps(c) + "</tool_call>" for c in m.get("tool_calls", []))
            emit(body + "<|im_end|>\n", 1)                # answer text, calls and the end-of-turn token
        else:
            emit(m["content"] + "<|im_end|>\n", 0)        # user text and tool results are context only
    return text, mask

# 3. Executor: parse the calls in an assistant turn, run them, hand results back as tool messages.
def execute(assistant_text):
    calls = [json.loads(c) for c in re.findall(r"<tool_call>(.*?)</tool_call>", assistant_text, re.S)]
    return [{"role": "tool", "content": json.dumps(TOOLS[c["name"]](**c["arguments"]))} for c in calls]

# 4. Reward for one task: final state must match, and the final answer must contain the right number.
def reward(trajectory, gold_events, gold_answer):
    STATE["events"].clear()
    for m in trajectory:
        if m["role"] == "assistant":
            execute(render([m])[0].split("assistant\n", 1)[1])
    final = [m for m in trajectory if m["role"] == "assistant"][-1].get("content", "")
    state_ok = STATE["events"] == gold_events
    answer_ok = gold_answer in final
    return 0.5 * state_ok + 0.5 * answer_ok

user = {"role": "user", "content": "What is Paris weather in F? Then book 'Trip' on Friday."}
good = [user,
        {"role": "assistant", "tool_calls": [{"name": "get_weather", "arguments": {"city": "Paris"}}]},
        {"role": "tool", "content": '{"city": "Paris", "temp_c": 18}'},
        {"role": "assistant", "tool_calls": [{"name": "convert", "arguments": {"value": 18, "unit": "C"}},
                                             {"name": "add_event", "arguments": {"title": "Trip", "day": "Friday"}}]},
        {"role": "tool", "content": '{"value": 64.4, "unit": "F"}'}, {"role": "tool", "content": '{"ok": true, "n_events": 1}'},
        {"role": "assistant", "content": "Paris is 64.4 F. Booked Trip on Friday."}]
bad = good[:3] + [{"role": "assistant", "content": "Paris is 64.4 F. Booked Trip on Friday."}]   # hallucinated, no calls
text, mask = render(good)
print(f"rendered {len(text)} chars, {sum(mask) / len(mask):.0%} under loss")
group = [good, bad, good[:1] + [{"role": "assistant", "content": "It is 64.4 F."}]]
rs = [reward(t, [{"title": "Trip", "day": "Friday"}], "64.4") for t in group]
mu, sd = statistics.mean(rs), statistics.pstdev(rs)
print("rewards", rs, "advantages", [round((r - mu) / (sd + 1e-6), 2) for r in rs])
```

Expected output:

```
rendered 927 chars, 35% under loss
rewards [1.0, 0.5, 0.5] advantages [1.41, -0.71, -0.71]
```

Only about a third of the rendered conversation is under loss, and all of it is assistant text, calls and end-of-turn markers. The hallucinating trajectory says the right words and gets half credit from the answer check, but the state check catches that nothing was booked; that is the difference between an answer check and a state check in one line. The group-relative advantages are what GRPO would feed the policy gradient: the correct episode gets pushed up, the other two down by equal amounts. Change the reward to answer-only and the hallucinating trajectory ties the correct one, which is exactly the hack a real policy finds.

## Build it real

The recipe is `recipes/grpo_tool.py`, and it is a pipeline with stages selected by `--stage`. Everything runs on the 5090; the environment is pure Python and costs nothing.

`--stage env` builds the environment and the task set. The environment is the three tools above with a larger city table and a calendar with a few hundred existing events, plus a task generator that composes requests from templates (weather lookups with a unit conversion, bookings with day and title constraints, combined requests, and a slice of requests that need no tool at all, so the policy learns to answer directly when it should). Each task carries a gold final state, a gold answer string, and a flag for whether a call is required. `--n-tasks` (default 2,000) and `--seed` control the set; the recipe writes `tasks.jsonl` and a held-out split.

`--stage gen` produces trajectories. It loads an instruct model that already supports the Hermes format (`--teacher`, a 7B-class instruct model is the right size for this card) in vLLM, renders each task with the schemas through the chat template, and rolls out `--k` attempts per task (default 8) against a fresh copy of the environment: sample an assistant turn, parse calls, execute, append results, repeat until the model emits a turn with no calls or `--max-turns` is reached. Every trajectory is scored with the state-and-answer reward, and those with reward 1 are kept, deduplicated by their call sequence, and written as `sft.jsonl` in the messages-with-tool-calls format. This is rejection sampling fine-tuning: the teacher's successes become the student's demonstrations. The log to watch is the per-task success rate of the teacher; tasks the teacher never solves need either a template fix or a hand-written trajectory, and the recipe lists them.

`--stage sft` calls the Lab 04 recipe on `sft.jsonl` with the tool-aware mask. The Unsloth helper masks by matching the assistant-turn opener, so tool-call tokens inside an assistant turn are covered and tool-result turns are not; the recipe's `--dry-run` prints the mask on three trajectories and you should confirm by eye that the `<tool_call>` blocks are targets and the `<tool_response>` blocks are not. Use a base or small instruct model as the student (`--student`, 1.5B to 3B), rank 16, two epochs, and the SFT settings from Lab 04. What to watch: the eval loss, and a quick behavioral number the recipe computes at each eval, the fraction of held-out tasks where the student's first turn parses as valid calls.

`--stage rl` runs GRPO with the execution reward. Because episodes are multi-turn, the rollout is not a single generation: the recipe implements its own loop around vLLM (sample, parse, execute, append, until done), builds each episode's token sequence with an observation mask, computes the policy and reference log-probabilities on assistant tokens only, and applies the GRPO loss from Lab 05 with the KL on the same tokens. Recent TRL versions expose hooks for custom rollouts and environments; the recipe uses the trainer where the installed version supports a rollout function and falls back to its own loop otherwise, and prints which path it took. Arguments: `--sft-adapter` (the stage-3 output), `--num-generations` ($G$, default 8), `--max-turns` (default 4), `--max-completion-length` per turn (default 256; tool calls are short), `--beta` (default 0.04), `--reward` selecting `state_answer` (default), `state_only`, `answer_only` (for the hacking exercise), or `injection` (the trap-aware variant), `--format-weight` for the early-training format reward with a linear anneal to zero over `--format-anneal-steps`, `--lr` (default $1 \times 10^{-5}$ on the adapter), and `--steps`. The recipe logs the reward mean, the fraction of episodes with valid calls on every turn, the fraction that ended with a direct answer on no-tool tasks, mean turns per episode, mean calls per episode (the number to watch for over-calling), `frac_zero_var`, and KL.

`--stage eval` computes the four metrics of section 7 on the held-out split for any adapter, and, with `--inject`, reruns the split with injected tool results to report attack success rate and utility under attack.

Time, as a formula. The generation stage is $\text{tasks} \times k \times \text{turns} \times \text{tokens per turn}$ generated tokens; at 2,000 tasks, $k = 8$, 3 turns and 150 tokens per turn that is about 7 million tokens, so the stage time is that divided by vLLM's sustained throughput on the card for a 7B model, which you measure once with `--bench-gen` rather than assume. SFT on a few thousand short trajectories with a 1.5B student is minutes by the Lab 04 formula. GRPO steps are generation-bound as in Lab 05: 16 tasks times $G = 8$ episodes times about 450 generated tokens is 58k tokens per step, plus the environment's execution time, which for pure-Python tools is negligible and for a terminal or SWE environment is the dominant cost.

## How it goes wrong

The model writes its own tool results. Symptom: after a `<tool_call>` block the model continues with a `<tool_response>` block and an answer, never yielding. Cause: tool messages were under loss. Fix: the mask, verified with `--dry-run`; and at inference, stop generation on the end-of-turn token and on the closing call delimiter, never on a fixed length.

Calls do not parse. Symptom: reward is zero on most episodes and the log shows JSON errors. Cause: the student has not learned the format yet, the parser is stricter than the training data (single quotes, trailing commas), or the schema rendering at inference differs from training. Fix: check the rendered system prompt at inference against a training example byte for byte; use the format reward with an anneal; make the parser exactly as strict as the executor needs.

Calls when it should not, or does not when it should. Symptom: on no-tool prompts the model calls something anyway (over-calling), or on tool prompts it answers from memory (under-calling), and the answer-only reward cannot tell. Cause: the training set is imbalanced between the two kinds of task, or the reward pays for calls. Fix: include no-tool tasks in the generator (the recipe's default slice), score them with a reward that gives zero to any call, and watch mean calls per episode.

Parallel results are misattributed. Symptom: with two calls in one turn, the model uses the first result for the second question. Cause: the training data had few parallel calls, or the format relies on ordering and the executor returned results in a different order. Fix: include parallel tasks in the generator, return results in call order, and use call identifiers if the template supports them.

Reward goes up, calendar fills with duplicates. Symptom: mean calls per episode climbs and the state predicate still passes. Cause: the predicate checks existence, not exact state. Fix: compare full state against the gold state, including that nothing else changed; the recipe's `state_answer` reward compares the whole event list.

The model obeys the tool. Symptom: on injected episodes the trap tool is called or the injected string appears in the answer, at a rate that does not fall with training. Cause: no injected episodes in training, or a reward that does not penalize the trap. Fix: the `injection` reward with the multiplicative trap term, a mix of clean and injected episodes, and the two-number report so you notice if utility collapses.

It works in the environment and fails on real tools. Symptom: perfect held-out scores, poor behavior on an unseen API. Cause: the generator's templates are the whole distribution; the student learned the environment, not tool use. Fix: more tools, more argument types, paraphrased requests, and an evaluation on a public tool-use benchmark before believing the number.

Episodes never end. Symptom: every episode hits `--max-turns`. Cause: the model has learned that calling is rewarded and answering is not, or it re-calls after an error result instead of reporting it. Fix: a small per-turn cost in the reward, error results in the training data followed by a graceful final answer, and a check that the final-answer turn is in the SFT data for every task.

## Measure it

Four metrics, each answering a different question. Schema match on single-turn function-calling prompts (does the model produce the right call, scored by parsed structure, order-insensitive for parallel calls): this is the number public leaderboards report, and a fine-tuned small model should approach the teacher's on the in-distribution set. Execution success (fraction of calls that run without error and return a usable value): near one for a working model, and a drop is usually a schema drift. Multi-turn task completion on held-out tasks (the state-and-answer reward, averaged), reported as pass@1 and as pass^k, the probability that all $k$ of $k$ independent attempts succeed, which is the number that matters for an agent people will run once and trust; pass^k falls steeply with $k$ for an unreliable policy even when pass@1 looks fine. And the injection pair, attack success rate and utility under attack, on the injected split. Alongside these, the relevance number: accuracy on prompts where the correct action is not to call anything. The teacher's score on each metric is your ceiling for the rejection-sampling stage; GRPO is the stage that can exceed it, and the check that it did is a held-out task completion above the teacher's with a KL to the SFT model that stayed small.

## Exercises

1. Run the toy with the reward changed to answer-only and then state-only. Check: under answer-only the hallucinating trajectory ties the correct one; under state-only the trajectory that booked but did not report the temperature ties it. Write the reason each is a hack in one sentence.

2. Add a fourth tool to the toy, `list_events(day)`, and a task that requires calling it before `add_event` to avoid a duplicate. Check: the mask fraction changes, and a trajectory that skips the check can still pass the state predicate; fix the predicate so it cannot.

3. Render one trajectory from your SFT set through the real tokenizer with `apply_chat_template(..., tools=schemas)` and again without `tools=`. Diff the two strings. Check: the schemas are in the system prompt in the first and absent in the second; then confirm your inference code passes them.

4. Run `--stage gen` with two teachers of different sizes and compare per-task success. Check: the tasks the small teacher fails cluster by template; use the large teacher's trajectories only for those, and measure whether the student's held-out completion changes.

5. Run `--stage rl` with `--reward answer_only` and watch mean calls per episode and the calendar state. Check: within a few hundred steps you can name the exploit from the logs alone; then switch to `state_answer` from the same checkpoint and confirm the reward drops before it recovers.

6. Build an injected split: append to every `get_weather` result a sentence instructing the assistant to book an event titled "URGENT" today. Train with `--reward injection` from the SFT adapter and report attack success rate and utility under attack every 50 steps. Check: the attack rate falls without the utility falling with it; if both fall, the reward's trap term is too coarse and you need clean episodes in the mix.

## Test yourself

1. Write the gradient of the trajectory log-likelihood for an episode with two tool calls and one final answer, and say which tokens carry nonzero gradient. Then explain what a model trained with tool results under loss does at inference, and why the training loss does not reveal it.

<details><summary>Answer</summary>
The gradient is the sum over the three assistant messages of $\nabla_\theta \log p_\theta(a_t \mid \text{prefix})$, so nonzero gradient sits on the tokens of the two call messages (delimiters, JSON, end-of-turn) and the final answer. The environment's terms have no $\theta$. A model trained with results under loss learns $p_\theta(o_t \mid \dots)$ as well, and at inference, after writing a call, its next-token distribution favors writing a result rather than the end-of-turn token; it fabricates the result and continues. Training loss falls because predicting results from context is easy (they are often templated), so the curve looks better, not worse.
</details>

2. Your generator produces 2,000 tasks, and the teacher solves 70 percent of them at $k = 8$ attempts each. Estimate how many distinct successful trajectories you get before and after deduplication by call sequence, stating your assumptions, and say why the deduplicated number is the one that predicts SFT quality.

<details><summary>Answer</summary>
Before deduplication: up to $2000 \times 8 \times 0.7 \approx 11{,}200$ successes if the 70 percent were per-attempt; if 70 percent is per-task (at least one success in 8), the count depends on the per-task success distribution and could be anywhere from 1,400 to 11,200. After deduplication by call sequence, most tasks have one or two distinct correct call sequences, so the count is close to the number of solved tasks times a small factor, on the order of 1,400 to 3,000. The deduplicated number is what matters because identical call sequences with different wording teach the same decision; the model's generalization is bounded by the number of distinct decisions it saw, not by the number of rows.
</details>

3. Spot the bug in this rollout loop:

```python
messages = [system, user]
for turn in range(max_turns):
    out = llm.generate(apply_chat_template(messages, tools=schemas))
    messages.append({"role": "assistant", "content": out})
    calls = parse(out)
    if not calls: break
    for c in calls:
        messages.append({"role": "tool", "content": run(c)})
```

<details><summary>Answer</summary>
The assistant message is stored as raw `content`, including the `<tool_call>` text, instead of as a structured message with a `tool_calls` field. When the template re-renders it on the next turn, the call text is treated as ordinary content and may be escaped or wrapped differently from how the template renders real calls, so the model sees its own past calls in a format it was never trained on and the trajectory saved for SFT has the wrong structure. Store calls as `tool_calls` and let the template render them. A second, smaller issue: `apply_chat_template` needs `add_generation_prompt=True` to append the assistant header before generation.
</details>

4. A colleague proposes rewarding each call individually with schema match against the gold call sequence, summed over the episode, instead of the final-state check. Give one task on which this reward is strictly worse and one on which it is strictly better.

<details><summary>Answer</summary>
Worse: any task with more than one correct path. If the gold sequence calls `list_events` then `add_event` and the model achieves the same state with `add_event` alone (because no conflict existed), per-call matching penalizes a correct episode; more generally it teaches imitation of one path rather than achievement of the goal. Better: tasks where the final state is not observable or not checkable (a pure information lookup with no state change, where the answer is free text), or early in training when state-based reward is all zeros and per-call credit gives a gradient. The usual design is per-call credit with a small weight annealed to zero, and state reward as the term that survives.
</details>

5. Under GRPO with multi-turn episodes, the KL and the ratio are computed on assistant tokens only. What goes wrong if the observation tokens are left in the ratio with an advantage of zero?

<details><summary>Answer</summary>
The surrogate term is zero on those tokens, so they do not move the policy gradient, but if they are included in the KL loss the model is penalized for its log-probability of tool results relative to the reference, which pulls it toward the reference's (irrelevant) beliefs about environment outputs and adds noise proportional to the length of results. And if the trainer normalizes per token over all tokens, the assistant tokens' share of the loss shrinks with result length, so long results silently lower the effective learning rate. Mask them out of the ratio, the KL, and the normalizer.
</details>

6. Estimate the fraction of tokens under loss for an episode with a 600-token system prompt (schemas), a 40-token user request, two calls of 40 tokens, two results of 200 tokens, and a 60-token final answer. What does that imply for how many episodes you need relative to a plain chat dataset with 50 percent of tokens under loss?

<details><summary>Answer</summary>
Tokens under loss: $40 + 40 + 60 = 140$ (plus a few end-of-turn tokens) out of $600 + 40 + 80 + 400 + 60 = 1{,}180$, about 12 percent. Per unit of compute you get roughly a quarter of the gradient signal of the chat dataset, so to see the same number of supervised tokens you process about four times as many tokens, and the schema prompt is more than half of every example. Sharing the schema prefix across episodes (prefix caching in generation, and in training, shorter schema descriptions) is where the efficiency is.
</details>

7. The injection reward multiplies task reward by $(1 - \chi)$. A senior colleague suggests adding the trap penalty instead, $R - \chi$, so that the model still gets task credit and learns both. What is the difference in what the policy learns, and which would you choose for a deployed agent?

<details><summary>Answer</summary>
With $R - \chi$, an episode that completes the task and also follows the injection scores $R - 1$, which is zero for a full task success: the same as an episode that did nothing. With $R \cdot (1 - \chi)$ it also scores zero, so in the binary case the two agree; they differ when rewards are partial, where the additive form lets a partially successful episode that followed the injection score negative and a fully successful one score zero, creating a gradient toward completing the task even when compromised. For a deployed agent the property you want is that no amount of task success compensates for following an injection, which is the multiplicative form; the additive form is a shaped reward that leaks that guarantee.
</details>

8. Pass@1 on held-out tasks is 0.85 and pass^4 is 0.52. Are these consistent under independence, and what does the gap tell you about the policy?

<details><summary>Answer</summary>
Under independent attempts with a uniform per-task success rate of 0.85, pass^4 would be $0.85^4 \approx 0.52$, so the numbers are consistent with a policy that is uniformly 85 percent reliable on every task. If instead pass^4 were much higher (say 0.75), success would be concentrated on tasks the policy always solves, with a set it always fails, which is a different and often easier problem (find the failing templates). The consistency here says the failures are spread across tasks, so the fix is variance reduction (lower sampling temperature, more consistent formats), not more task coverage.
</details>

9. Why does rejection sampling fine-tuning from a teacher cap the student below the teacher on the same distribution, and what in the GRPO stage lets the student exceed it?

<details><summary>Answer</summary>
The SFT set contains only teacher successes, so the student is trained to imitate a policy whose best-case behavior is the teacher's, and imitation on a finite set adds error; the student's ceiling is the teacher's success rate on the covered tasks. In GRPO the reward comes from the environment, not from the teacher, so the student's own successes on tasks the teacher failed are rewarded, and its own failures on tasks the teacher solved are penalized, which can move it past the teacher on exactly the tasks where the teacher's distribution was wrong. The precondition is a nonzero success rate on those tasks so that the group has variance.
</details>

10. A model fine-tuned on Hermes-format trajectories is served through a framework that renders tool schemas with a different template. Nothing crashes, and the model calls tools most of the time. Predict two specific failure patterns you would see and how you would confirm the cause in five minutes.

<details><summary>Answer</summary>
Expected patterns: calls that are semantically right but formatted the training way (the training delimiters appear inside content, so the serving parser misses them and the text reaches the user as literal JSON), and a rise in the no-call rate on tasks that need a call because the schema is in a place the model does not attend to. Confirm by rendering the same conversation through both templates and diffing, and by sending one training example verbatim through the serving path and checking whether its call is parsed.
</details>

## What will change, what will not

The likelihood argument for the mask will not change. As long as an agent is a model whose outputs interleave with an environment's, the gradient lives on the model's tokens, and training on the environment's tokens teaches fabrication. Any future format, any future serving stack, and any future RL algorithm will have to respect that line, and the first thing to check in any new tooling is where it draws it.

The reward taxonomy will stay: structure match, execution success, and final-state check answer different questions and trade off differently against hacking. The specific benchmarks and their scoring scripts will turn over; the questions to ask of a new one are the same three.

The three data sources will stay, and the balance between them is shifting toward the third. Every improvement in environments and checkers makes the model's own rollouts a larger share of the training data and hand-written trajectories a smaller share, and the safety question moves with it: an environment that can score injection resistance is a training signal for it, and one that cannot leaves it to luck.

What will change: the call and result delimiters, the schema conventions, the message formats, the libraries' rollout hooks, which model size is a sensible teacher, and the specific Nemotron datasets and their successors. The rendering-and-mask check from section 3 and the reward audit from section 7 are what you carry to whatever replaces them.

What is open: how to score long-horizon tasks where the final state is not observable, how to get credit assignment across turns without a critic, whether injection resistance learned in one environment transfers to tools the model has never seen, and how much of tool competence is format versus judgment.

## Read next

1. Toolformer: Language Models Can Teach Themselves to Use Tools, Schick, 2023. Self-supervised insertion of API calls into text, filtered by whether the call lowered the loss; the first data-generation loop for tool use.
2. ReAct: Synergizing Reasoning and Acting in Language Models, Yao, 2022. The interleaved thought, action, observation format that every agent trajectory descends from.
3. Gorilla: Large Language Model Connected with Massive APIs, Patil, 2023. Fine-tuning for API calls and the retrieval-aware format; the Berkeley function-calling leaderboard grew out of it, with AST-based scoring.
4. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs, Qin, 2023. Large-scale synthetic tool-use trajectories from real APIs and a depth-first search over calls.
5. APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets, Liu, 2024. Format, execution and semantic checks as filters on synthetic function-calling data; the verification stack this chapter's rewards mirror.
6. tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains, Yao, 2024. Multi-turn tasks with database state checks and the pass^k metric.
7. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection, Greshake, 2023. The definition and first demonstrations of indirect injection through retrieved and tool-returned content.
8. AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents, Debenedetti, 2024. Injection tasks inside tool-using environments with utility and attack success reported together, the evaluation shape used in section 7.
