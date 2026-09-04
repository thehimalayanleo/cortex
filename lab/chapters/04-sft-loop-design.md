---
title: "Lab 04: Designing the SFT loop"
kind: permanent
topics: [lab]
chapter: 4
station: posttrain
recipe: recipes/sft_lora.py
reading_time: 55 min
---

## What you will be able to do

1. Take a raw chat dataset, render it through a tokenizer's chat template, and verify by hand which token ids carry loss and which do not, including the end-of-turn token.
2. Derive the assistant-only loss from the conditional likelihood and explain, with the gradient in front of you, what goes wrong when you train on prompt tokens too.
3. Write the LoRA update from scratch, choose rank, alpha and target modules with a parameter and memory budget you computed yourself, and explain why LoRA tolerates a learning rate about ten times higher than full fine-tuning.
4. Run a curation loop on the 5090 with Unsloth and TRL: train, evaluate, read failures, rewrite data, retrain, and know which three numbers tell you to stop.
5. Mix a small slice of pretraining-style data into SFT and measure whether it bought you retention or just cost you steps.

## The idea in one paragraph

A pretrained model is a very good text continuer that has no idea it is supposed to be an assistant. Supervised fine-tuning shows it a few thousand conversations written the way you want it to talk, and trains it to reproduce only the assistant's side of each conversation, not the user's. That last point is the whole design: the question is context, the answer is the target. Because the model already knows almost everything it needs, the update can be tiny, so you write it as a low-rank correction on top of frozen weights (LoRA), which fits on one 32 GB card even for an 8B model. Then the work becomes a loop over data rather than over hyperparameters: train, look at what the model gets wrong, fix the examples, train again. You stop when held-out answer loss stops falling and a small behavioral check stops improving, which usually happens after one to three passes over the data.

## The math

### Tokens, turns and templates

A chat is a list of messages, each with a role (system, user, assistant, and in Lab 06, tool). The model only ever sees a flat token sequence, so a chat template is a deterministic function from the message list to a string, and the tokenizer turns that string into ids. For a ChatML-style template the string for one exchange looks like

```
<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nWhat is 7 times 8?<|im_end|>\n<|im_start|>assistant\n56<|im_end|>\n
```

The tokens `<|im_start|>` and `<|im_end|>` are special tokens: single ids that the tokenizer never produces from ordinary text, so the model can learn structural meaning for them that no user-typed string can spoof. Two facts about them matter for training. First, a base model has never seen them, so their embedding rows are whatever the initialization left there (often random, sometimes the mean of other rows); the model has to learn those rows during SFT, which is why some recipes unfreeze the embedding and output rows for exactly those ids. Second, the end-of-turn token is how the model learns to stop. If your loss mask excludes it, the model learns to write good answers and then keeps going, because nothing ever taught it that turns end. You will see the failure in section 6.

Every tokenizer on Hugging Face ships its template in `tokenizer.chat_template`, and `tokenizer.apply_chat_template(messages, tokenize=False)` renders it. Always render one example, print it, and read it before you train. Half of all SFT bugs are visible in that one print.

### The assistant-only loss, derived

Write one rendered conversation as a token sequence $x = (x_1, \dots, x_T)$ and let $\mathcal{A} \subseteq \{1, \dots, T\}$ be the set of positions that belong to assistant turns, including each turn's end-of-turn token. Let $\theta$ be the parameters and $p_\theta(x_t \mid x_{<t})$ the next-token distribution. Plain language-model training maximizes

$$
\log p_\theta(x) = \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}).
$$

What you actually want the model to be good at is producing an assistant turn given everything before it. For a single-turn chat with prompt tokens $x_{1:m}$ and answer tokens $x_{m+1:T}$, the object of interest is the conditional

$$
\log p_\theta(x_{m+1:T} \mid x_{1:m}) = \sum_{t=m+1}^{T} \log p_\theta(x_t \mid x_{<t}),
$$

which is the full log-likelihood minus $\sum_{t \le m} \log p_\theta(x_t \mid x_{<t})$, the log-likelihood of the prompt itself. Those dropped terms are the model's estimate of the distribution of user questions. Nothing in your objective wants the model to be a good generator of user questions, and training on them has a concrete cost: the prompt is typically longer than the answer, so under a full loss most of the gradient signal pulls the weights toward imitating users and system prompts. In the multi-turn case the same argument applies turn by turn: each assistant turn is conditioned on the whole prefix, so the per-example loss is

$$
\mathcal{L}(\theta) = -\frac{1}{|\mathcal{A}|} \sum_{t \in \mathcal{A}} \log p_\theta(x_t \mid x_{<t}).
$$

In code this is a labels tensor equal to the input ids where $t \in \mathcal{A}$ and $-100$ elsewhere, which `cross_entropy(ignore_index=-100)` skips. Note the shift: the logits at position $t-1$ predict $x_t$, so the mask is applied to the target, not to the input. The input still contains the user tokens; masking them from the loss does not hide them from attention.

One subtle choice is the normalizer. The formula above averages over the assistant tokens of one example. Most trainers instead average over all unmasked tokens in the batch, which weights a 400-token answer forty times more than a 10-token one. That is usually what you want for a chat model (long answers are where quality lives), but it means a handful of very long examples can dominate a batch. If you see the loss curve jump in step with a few long examples, this is why.

### Padding versus packing

Examples have different lengths. Padding pads each to the longest in the batch and wastes compute on pad tokens; the waste fraction for a batch of $B$ examples with lengths $\ell_i$ and padded length $L = \max_i \ell_i$ is

$$
w = 1 - \frac{\sum_{i=1}^{B} \ell_i}{B \cdot L}.
$$

For a batch of four chats of lengths 180, 260, 900 and 1,900 tokens, $w = 1 - 3240 / 7600 \approx 0.57$: more than half the FLOPs go to padding. Sorting by length before batching helps; packing removes the waste entirely by concatenating examples into fixed-length rows of, say, 2,048 tokens with an end-of-text separator. Packing changes the attention pattern unless you take care: with an ordinary causal mask, tokens of the second example attend to tokens of the first, and the model learns to use a previous, unrelated conversation as context. The correct fix is a block-diagonal mask, which in practice means resetting position ids at each boundary and using the variable-length kernel of flash attention (TRL calls this `padding_free`, and its packing strategy pairs with it). Verify your library does this: check that the position ids in a packed row restart from zero at each example boundary. If they do not, either accept the leakage (it is a small effect for short examples, a real one for long ones) or switch to length-sorted padding.

### LoRA: the low-rank update

A linear layer computes $h = W x$ with $W \in \mathbb{R}^{d \times k}$. Full fine-tuning updates all $dk$ entries. LoRA freezes $W_0$ and adds a product of two thin matrices:

$$
W = W_0 + \frac{\alpha}{r} B A, \qquad B \in \mathbb{R}^{d \times r}, \; A \in \mathbb{R}^{r \times k}, \; r \ll \min(d, k).
$$

$A$ is initialized with small random entries (PEFT uses a uniform distribution of width about $1/\sqrt{k}$, so each entry has variance of order $1/k$) and $B$ is initialized to zero, so at step zero the model is exactly the base model. The forward pass costs one extra pair of skinny matmuls, $x \mapsto Ax \mapsto BAx$, and at inference you can merge $W_0 + \frac{\alpha}{r} BA$ into a single matrix, so there is no serving cost.

The parameter count is $r(d + k)$ per adapted matrix. For Llama-3-8B shapes (hidden 4096, intermediate 14,336, 8 key-value heads so the key and value projections are 4096 to 1024, 32 layers) adapting every linear layer in a block gives, per layer,

$$
r \cdot \big[(4096 + 4096) + 2(4096 + 1024) + (4096 + 4096) + 2(4096 + 14336) + (14336 + 4096)\big] = r \cdot 81{,}920,
$$

and over 32 layers $r \cdot 2.62\text{M}$. At $r = 16$ that is about 42M trainable parameters, roughly half a percent of the base. Notice where they go: the two MLP projections up and down plus the gate account for $55{,}296$ of the $81{,}920$, so an adapter that only touches attention (a common default in early tutorials) skips two thirds of the adaptable surface. The empirical finding, repeated across the papers in Read next, is that adapting all linear layers with a modest rank beats adapting attention only with a large rank.

Why is a low-rank update enough? The change a fine-tune needs is a function of a few thousand examples, and the gradient of the loss with respect to $W$ is a sum of outer products, one per token: $\partial \mathcal{L} / \partial W = \sum_t \delta_t x_t^\top$, where $\delta_t$ is the backpropagated error at that layer. Its rank is bounded by the number of distinct directions in the $x_t$ and $\delta_t$ that matter, and the accumulated update over a short fine-tune stays close to low rank in practice. LoRA does not assume the base weights are low rank; it assumes the change is. When that assumption fails (learning a genuinely new domain with a lot of new facts, or a new language), you will see LoRA plateau above full fine-tuning, and the fix is to raise the rank or fine-tune fully.

### Rank and alpha

The scale $\alpha / r$ multiplies the product. Its purpose is to make the size of the update independent of $r$ so that hyperparameters transfer when you change rank. Under the assumptions above, the entries of $BAx$ grow like $\sqrt{r}$ (a sum of $r$ random-signed terms), so dividing by $r$ overcorrects and dividing by $\sqrt{r}$ is the scale-invariant choice; that observation is rsLoRA, and it matters once $r$ is in the hundreds. At $r \le 64$ the common convention $\alpha = 2r$ (so the multiplier is 2) works and is what the recipe defaults to. The important number is the product $\text{lr} \cdot \alpha / r$, not either factor alone.

### Why LoRA wants a higher learning rate

This is the derivation that most people skip, and it explains a rule of thumb you will otherwise have to memorize. Consider Adam. After the first few steps its update to each parameter entry has magnitude of order the learning rate $\eta$, nearly regardless of the gradient's scale, because the gradient is divided by its running root-mean-square. In full fine-tuning, each entry of $W$ therefore moves by about $\eta$ per step.

In LoRA, the same per-entry step of size $\eta$ is applied to $B$ and $A$, and the induced change in $W$ is

$$
\Delta W = \frac{\alpha}{r} \left( \Delta B \, A + B \, \Delta A \right).
$$

Early in training $B \approx 0$, so the second term is negligible and $\Delta W_{ij} \approx \frac{\alpha}{r} \sum_{l=1}^{r} \Delta B_{il} A_{lj}$. Each $\Delta B_{il}$ has magnitude about $\eta$ with a sign set by the gradient, and each $A_{lj}$ has magnitude about $1/\sqrt{k}$ with a random sign. The sum of $r$ such terms has magnitude about $\eta \sqrt{r} / \sqrt{k}$, so

$$
|\Delta W_{ij}| \approx \eta \cdot \frac{\alpha}{r} \cdot \sqrt{\frac{r}{k}} = \eta \cdot \frac{\alpha}{\sqrt{r k}}.
$$

For $k = 4096$, $r = 16$, $\alpha = 32$ that factor is $32 / \sqrt{65{,}536} = 32 / 256 = 1/8$. The same learning rate moves the effective weights about eight times less under LoRA than under full fine-tuning, so to get a comparable trajectory you raise $\eta$ by roughly that factor. This is why LoRA recipes cluster around $1\times 10^{-4}$ to $2\times 10^{-4}$ while full fine-tuning clusters around $1\times 10^{-5}$ to $2\times 10^{-5}$. The derivation assumes Adam's sign-like update, the PEFT initialization of $A$, and independence of signs; it is a scaling argument, not a law, and the correct value on your data is still found by a short sweep. But it tells you which direction to sweep and how far.

### QLoRA: quantized base, bf16 adapters

QLoRA stores the frozen $W_0$ in 4-bit NormalFloat (NF4), a data type whose 16 levels are placed at the quantiles of a standard normal so that normally distributed weights use the levels evenly. Weights are quantized in blocks of 64 with one scale per block; the scales themselves are quantized to 8 bits in a second pass (double quantization). The cost per weight is 4 bits plus about $8/64$ bits for the scale plus a small second-level term, about 0.52 bytes per parameter. In the forward pass each block is dequantized to bf16 just before its matmul, so the arithmetic is the same as bf16 LoRA; only the memory and a dequantization overhead differ.

A worked budget for Llama-3-8B on a 32 GB card, with the assumptions stated: the embedding and output matrices ($2 \times 128{,}256 \times 4096 \approx 1.05\text{B}$ parameters) are usually kept in bf16 at 2 bytes each, about 2.1 GB; the remaining roughly 7B parameters at 0.52 bytes are about 3.6 GB; LoRA weights at $r = 16$ are 42M parameters, and with Adam's two moments and a master copy in fp32 they cost $42\text{M} \times 12 \approx 0.5$ GB. That is about 6.2 GB before activations. Activations with gradient checkpointing are dominated by one stored input per layer, $B \cdot L \cdot 4096 \cdot 2$ bytes for 32 layers, which at batch 4 and 2,048 tokens is about 2.1 GB, plus one layer's worth of recomputation buffers. The number that surprises people is the logits: $B \cdot L \cdot V \cdot 4$ bytes in fp32 is $4 \times 2048 \times 128{,}256 \times 4 \approx 4.2$ GB, more than the quantized model. Unsloth and recent TRL versions chunk the cross-entropy so this never materializes at once; if you write your own loop, do the same or drop the batch size.

### Learning rate, epochs, and mixing

SFT is a short run, so the schedule is simple: linear warmup over the first few percent of steps, then cosine or linear decay to zero or to a tenth. One to three epochs. The reason for so few is that the model memorizes the data quickly (thousands of examples, billions of parameters) and the held-out loss turns upward after the point of memorization while the training loss keeps falling. Smaller datasets tolerate more epochs, because each epoch is fewer steps; what matters is the total number of gradient steps relative to the number of distinct examples.

Forgetting is real even with LoRA. The clean way to think about it is that the fine-tuning objective is $\mathcal{L}_{\text{sft}}$ alone, and nothing in it penalizes drift on inputs it never sees. Mixing in a fraction $\lambda$ of pretraining-style or general instruction data turns the objective into $(1 - \lambda)\mathcal{L}_{\text{sft}} + \lambda \mathcal{L}_{\text{mix}}$, which anchors the model on a broad input distribution. The midtrain station in the browser shows the same trick one stage earlier, as a mixture with a cooldown. A fraction between 5 and 20 percent is typical; the exercise in section 8 asks you to measure the tradeoff rather than guess it. In the toy below you will see forgetting happen in front of you, and one of its causes is more interesting than drift: when the fine-tuning set has no variation on some input feature, the model is free to stop reading that feature.

## Build it small

The snippet trains a tiny causal transformer as a "base model" on addition with a loss over every token, then freezes it, wraps every linear layer in a LoRA adapter, and runs SFT on multiplication with loss only on the answer digits and the end token. It prints trainable parameter counts and teacher-forced exact-answer accuracy on both tasks before and after. The in-browser posttrain station does the same thing at character level: the highlighted tokens on the SFT tab are the only positions being graded.

```python
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0)
PLUS, EQ, Q, A, EOS, PAD, MUL = 10, 11, 12, 13, 14, 15, 16
V, T, D, H = 17, 10, 64, 4

def make(n, op, full=False):
    a, b = torch.randint(0, 10, (n,)), torch.randint(0, 10, (n,))
    ids = torch.full((n, T), PAD); lab = torch.full((n, T), -100)
    for i in range(n):
        y = int(a[i] * b[i]) if op == MUL else int(a[i] + b[i])
        seq = [Q, int(a[i]), op, int(b[i]), EQ, A] + [int(c) for c in str(y)] + [EOS]
        ids[i, :len(seq)] = torch.tensor(seq)
        start = 0 if full else 6            # full: LM loss on every token; else answer + EOS only
        lab[i, start:len(seq)] = ids[i, start:len(seq)]
    return ids, lab

class Block(nn.Module):
    def __init__(s):
        super().__init__()
        s.qkv, s.o = nn.Linear(D, 3 * D), nn.Linear(D, D)
        s.up, s.down = nn.Linear(D, 4 * D), nn.Linear(4 * D, D)
        s.n1, s.n2 = nn.LayerNorm(D), nn.LayerNorm(D)
    def forward(s, x):
        B, L, _ = x.shape
        q, k, v = s.qkv(s.n1(x)).view(B, L, 3, H, D // H).transpose(1, 3).unbind(2)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.o(att.transpose(1, 2).reshape(B, L, D))
        return x + s.down(F.gelu(s.up(s.n2(x))))

class LM(nn.Module):
    def __init__(s):
        super().__init__()
        s.tok, s.pos = nn.Embedding(V, D), nn.Embedding(T, D)
        s.blocks = nn.Sequential(*[Block() for _ in range(2)])
        s.head = nn.Linear(D, V)
    def forward(s, ids):
        return s.head(s.blocks(s.tok(ids) + s.pos(torch.arange(ids.shape[1]))))

class LoRA(nn.Module):
    def __init__(s, base, r=8, alpha=16):
        super().__init__()
        s.base, s.scale = base, alpha / r
        s.A = nn.Parameter(torch.randn(r, base.in_features) / base.in_features ** 0.5)
        s.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero: W is unchanged at step 0
    def forward(s, x):
        return s.base(x) + s.scale * F.linear(F.linear(x, s.A), s.B)

def loss_fn(model, ids, lab):
    logits = model(ids[:, :-1])                 # logits at t predict token t+1
    return F.cross_entropy(logits.reshape(-1, V), lab[:, 1:].reshape(-1), ignore_index=-100)

def accuracy(model, op, n=1000):
    ids, lab = make(n, op)
    with torch.no_grad(): pred = model(ids[:, :-1]).argmax(-1)
    ok = (pred == lab[:, 1:]) | (lab[:, 1:] == -100)
    return ok.all(1).float().mean().item()      # whole answer and EOS right, teacher-forced

def train(model, op, steps, full=False, lr=3e-3):
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr)
    for _ in range(steps):
        loss = loss_fn(model, *make(64, op, full)); opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

model = LM()
train(model, PLUS, 2000, full=True)             # "pretraining": next-token loss on everything
print(f"base   add {accuracy(model, PLUS):.2f}  mul {accuracy(model, MUL):.2f}")
for p in model.parameters(): p.requires_grad_(False)
for blk in model.blocks:                         # wrap every linear layer with a LoRA adapter
    blk.qkv, blk.o, blk.up, blk.down = (LoRA(m) for m in (blk.qkv, blk.o, blk.up, blk.down))
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA trainable {n_tr} of {sum(p.numel() for p in model.parameters())}")
train(model, MUL, 2000)                          # SFT: loss on answer tokens only
print(f"SFT    add {accuracy(model, PLUS):.2f}  mul {accuracy(model, MUL):.2f}")
```

Expected output, from one run on a CPU with the seed above (about two minutes; your numbers will differ slightly by platform):

```
base   add 1.00  mul 0.03
LoRA trainable 16384 of 119185
SFT    add 0.01  mul 1.00
```

Read the three lines slowly. The base model learns addition perfectly and knows nothing about multiplication. Sixteen thousand adapter parameters, 14 percent of the model, are enough to teach it multiplication to 100 percent with answer-only loss. And addition is gone. Two things caused that. The learning rate is high and every linear layer is adapted, so the adapter has plenty of capacity to overwrite behavior. More importantly, every SFT example uses the multiplication operator, so the operator token is constant in the fine-tuning set and the cheapest solution is to ignore it entirely and always multiply. The multiplication token's embedding row was never trained (it is frozen at its random initialization), and the model still learned to use it, which is the same thing that happens to a base model's untrained chat tokens during SFT. Exercise 2 asks you to fix the forgetting by mixing.

## Build it real

The recipe is `recipes/sft_lora.py`. It loads a base model with Unsloth's `FastLanguageModel.from_pretrained` (optionally in 4-bit), attaches LoRA with `get_peft_model`, renders the dataset with the tokenizer's chat template, and hands it to TRL's `SFTTrainer` with assistant-only loss. Unsloth's `train_on_responses_only` and TRL's `assistant_only_loss=True` do the same masking through different mechanisms; the recipe uses the Unsloth helper because it works with any template as long as you give it the strings that open a user turn and an assistant turn.

Arguments and what they control:

`--model` is a Hugging Face id or local path. Start with an 8B-class base model, not an instruct model, so you can see SFT do its job from nothing; a 1B or 3B model is fine for iterating on data and takes a fraction of the time. `--data` and `--eval` are JSONL files with one `{"messages": [...]}` object per line in the standard role/content format. `--r` and `--alpha` set the rank and scale (defaults 16 and 32). `--targets all` adapts every linear layer in the transformer blocks; `--targets attn` adapts only q, k, v and o so you can measure the difference. `--lr` defaults to $2 \times 10^{-4}$, `--epochs` to 2, `--warmup` to 0.03 of the steps, with a cosine decay. `--max-len` is the packed row length (2,048 by default), `--bsz` the per-device batch and `--grad-accum` the accumulation, so the effective batch is their product times the row length in tokens. `--load-in-4bit` selects QLoRA. `--packing` enables packing; the recipe checks that the installed TRL supports padding-free packing and refuses to pack otherwise, for the reason in section 3. `--mix path.jsonl --mix-frac 0.1` interleaves a second dataset at the given fraction for the forgetting experiments. `--train-embeddings` unfreezes the embedding and output rows so untrained chat tokens can be learned; you need it when the base tokenizer did not already have the template's special tokens. `--out` is the run directory; the recipe writes the adapter, a merged bf16 copy if you pass `--merge`, and `metrics.jsonl`.

What to watch in the logs. The training loss on assistant tokens should start somewhere between 1.5 and 3 nats per token for a base model that has never seen the template (the first few dozen steps are mostly the model learning the special tokens, and the loss falls fast), then settle into a slow decline. Gradient norm should be stable and of order one after warmup; spikes mean the learning rate is too high for this rank or a pathological example (a very long or repetitive one) just went through. The eval loss, computed every `--eval-every` steps, should track the training loss down during the first epoch; the gap between them is your memorization meter. Tokens per second tells you whether packing worked: with padding you will see it fluctuate with batch composition, with packing it is flat. Unsloth prints peak reserved memory at the end; if you are within 2 GB of 32, reduce `--bsz` before the next run so that the occasional long example does not kill a three-hour job at the end.

How long it takes, as a formula with stated assumptions rather than a measurement. For LoRA the backward pass does not compute weight gradients for the frozen matrices, so the cost per token is about $4N$ FLOPs (forward $2N$, backward through activations $2N$) rather than the $6N$ of full training, where $N$ is the parameter count of the base. For an 8B model that is $3.2 \times 10^{10}$ FLOPs per token. A dataset of 10,000 chats averaging 1,000 tokens, two epochs, is $2 \times 10^7$ tokens, so $6.4 \times 10^{17}$ FLOPs. If you take the card's dense bf16 tensor-core peak from the spec sheet as $P$ and assume a realized utilization of 30 percent for a LoRA workload with quantization overhead, then with $P = 200$ TFLOP/s the run takes $6.4 \times 10^{17} / (0.3 \times 2 \times 10^{14}) \approx 10{,}700$ seconds, about three hours. Halve the model to 4B and you halve it; use a 1B model for the curation loop and it is under twenty minutes per iteration, which is the whole point of iterating on data with a small model first.

The curation loop, concretely: train on version $k$ of the data; run the eval prompts through the adapter with greedy decoding; read every failure (not a sample of them, all of them, while the dataset is small); classify each failure as a format problem, a knowledge problem, a stop problem, or a style problem; fix the examples that would have taught the right behavior, or write the three to five new ones that cover the gap; delete examples that are wrong or that a reviewer would not be proud of; retrain. Three iterations of this with a 1B model beat one iteration of hyperparameter search with an 8B model almost every time, because a wrong example is a wrong gradient no matter how good the optimizer.

## How it goes wrong

The model never stops. Symptom: answers are correct then trail into a new user turn, or repeat. Cause: the end-of-turn token is not in the loss mask, either because the masking helper matched the assistant prefix but cut the mask before the closing token, or because the dataset's rendered strings lack the closing token. Fix: print one example's labels next to its tokens and confirm the id of `<|im_end|>` (or the model's equivalent) is a target at the end of every assistant turn.

The model answers by restating the question. Symptom: outputs begin with a paraphrase of the prompt or a system-prompt-like preamble. Cause: loss on user and system tokens; the model learned that generating question-shaped text is rewarded. Fix: the mask; verify with the same print, and check the fraction of tokens under loss, which for a typical chat set should be between 25 and 60 percent.

Eval loss rises after the first epoch while training loss falls. Symptom: the classic divergence. Cause: memorization of a small dataset; with a few thousand examples and a learning rate of $2 \times 10^{-4}$, the second pass is mostly memorization. Fix: stop at the eval minimum (the recipe saves a checkpoint at each eval), lower the rank, or get more distinct examples. Do not fix it with dropout on the adapter and call it done; dropout delays the divergence rather than removing it.

Loss spikes or becomes NaN. Symptom: a sudden jump followed by garbage outputs. Causes: fp16 instead of bf16 (overflow in the attention logits), a learning rate that is high for a rank of 128 or more, or a corrupted example with tens of thousands of repeated tokens. Fix: bf16 always on Blackwell; clip the gradient norm at 1; find and remove the example by logging the batch index at the spike.

Packed examples bleed into each other. Symptom: the model occasionally answers a question from an earlier, unrelated conversation, or cites facts that appeared only in a previous example. Cause: packing with a plain causal mask. Fix: padding-free packing with position-id reset, or no packing.

Special tokens appear as text in the output. Symptom: the model writes the literal string `<|im_end|>` or `</s>`. Cause: the training data was rendered to a string and then tokenized with `add_special_tokens` handling that split the marker into ordinary pieces, so the model learned to produce the pieces. Fix: tokenize with the chat template (`apply_chat_template(tokenize=True)`) or check that the special ids appear in the id sequence, not their spelled-out fragments.

The model got worse at things you did not train. Symptom: a general benchmark drops by several points, or the model refuses to answer things it used to answer. Cause: drift with no anchor, made worse by a narrow dataset (recall the toy: constant features get ignored). Fix: mix 5 to 20 percent general data, lower the learning rate, adapt fewer modules, or fewer steps. Measure before and after; do not assume.

Style tics. Symptom: every answer begins the same way, or the model overuses a phrase. Cause: the dataset does, because many of its examples were generated by the same model or written by the same person on the same afternoon. Fix: deduplicate near-duplicates, cap the number of examples per source, and read the first sentence of fifty random examples; if you can predict it, so can the model.

## Measure it

Three numbers, in order of how much to trust them. First, held-out assistant-token negative log-likelihood, in nats per token, on a set of chats written by a different person than the training set. It should fall from the base model's value and stop falling; the minimum is your stopping point. Its absolute value depends on the data and is not comparable across datasets, so only compare it across runs on the same eval set. Second, a behavioral check: 50 to 200 held-out prompts with a deterministic checker where possible (exact match for closed-form answers, a JSON validator for structured outputs, a regular expression for format constraints) and a rubric otherwise; report the pass rate, the rate of correct stopping (the answer ended with the end-of-turn token before `max_new_tokens`), and the length distribution. A good SFT run has a stopping rate near 100 percent and a length distribution that matches the training data's; a rising mean length across runs is an early warning. Third, retention: run a small fixed subset of general tasks through lm-eval-harness (a few hundred questions from a knowledge benchmark and a few hundred from a reasoning one) on the base and on the adapter. A drop within about one point is noise at that sample size; a drop of several points is forgetting and the mixing fraction is your lever. The posttrain station shows the browser version of the first number as "answer loss".

When to stop: the eval NLL has not improved for two consecutive evaluations, the behavioral pass rate has plateaued, and the failures you read in the last curation round are ones the data cannot fix (they need a bigger model or a different objective, such as the preference training in Lab 05).

## Exercises

1. Print the rendered template and the label mask for one three-turn conversation using the recipe's `--dry-run` flag (it renders ten examples and prints tokens next to labels without training). Count the tokens under loss. Check: every assistant turn ends with an end-of-turn token that is a target, and no user token is.

2. In the toy, change the SFT stage so that each batch is 70 percent multiplication and 30 percent addition (call `make` twice and concatenate). Check: multiplication accuracy still reaches about 1.0 and addition stays above 0.9. Then try 5 percent addition and see what the minimum anchoring fraction is for this toy.

3. Repeat the toy's SFT stage with the adapter only on `qkv` and `o`. Check: multiplication takes more steps to reach the same accuracy, or does not reach it at the same rank; note that the MLP holds most of the adaptable capacity.

4. Compute the effective-weight step factor $\alpha / \sqrt{rk}$ for your chosen $r$, $\alpha$ and the largest $k$ in your model (the down projection's input dimension). Then run the recipe at three learning rates spanning a factor of ten around $2 \times 10^{-4}$ on a 1B model for 200 steps each and plot the eval loss. Check: the best rate is within a factor of three of the value the scaling argument predicts from a full fine-tuning rate of $1.5 \times 10^{-5}$.

5. Take a run that shows the epoch-two divergence and train a second run at half the rank. Check: the eval minimum moves later in training and is at most slightly worse; if it is much worse, the task needed the rank and the fix is more data, not less capacity.

6. Write a checker for one behavioral property you care about (for instance, "answers to arithmetic questions contain exactly one number and it is correct") and run the curation loop three times with a 1B model. Keep a log of failure classes per round. Check: the dominant failure class changes between rounds; if it does not, you are fixing the wrong examples.

## Test yourself

1. The loss mask is applied to the labels, and the labels are the input ids shifted by one. A colleague builds the mask on the input ids instead and shifts afterward. Which token of each assistant turn ends up unmasked that should not be, and which one masked that should not be?

<details><summary>Answer</summary>
The logits at position $t-1$ predict token $t$. Marking the assistant span on the inputs and then shifting the labels by one moves the span one position late in the target frame: the first assistant token (right after the assistant header) is masked out, and the first token after the assistant turn (the newline or the next role header) is unmasked. In practice the model loses the loss on the first answer token and gains a spurious loss on the token following the end-of-turn marker. The fix is to build the mask in the target frame, or to build labels first and then mask.
</details>

2. Suppose your dataset has 4,000 examples averaging 900 tokens, 40 percent of them assistant tokens. You train two epochs at an effective batch of 32 rows of 2,048 packed tokens. How many optimizer steps is that, and how many times does the model see each assistant token? What does that imply about the eval curve?

<details><summary>Answer</summary>
Total tokens per epoch is $4000 \times 900 = 3.6\text{M}$; packed rows of 2,048 give about 1,758 rows per epoch, and at 32 rows per step that is about 55 steps per epoch, 110 total. Each assistant token is seen exactly twice. With only 110 steps the warmup fraction and the decay shape matter a lot (3 percent warmup is three steps), and the eval curve will be coarse. Expect the eval minimum somewhere late in epoch one to early in epoch two, and evaluate at least every 10 steps or you will miss it.
</details>

3. State the two assumptions behind the claim that LoRA's effective step on $W$ is $\eta \alpha / \sqrt{rk}$, and give one situation in which the argument breaks down.

<details><summary>Answer</summary>
Assumptions: Adam's per-entry step has magnitude about $\eta$ regardless of gradient scale (true after the moment estimates settle), and the entries of $A$ are independent with magnitude about $1/\sqrt{k}$ and random sign, so the sum over $r$ terms grows like $\sqrt{r}$. It breaks down once $B$ is no longer near zero, because the $B \Delta A$ term contributes and the two terms are correlated; it also breaks down with SGD, where the step is proportional to the gradient and the gradient with respect to $B$ is itself scaled by $A$, so the effective step scales differently ($\propto \alpha^2 / r^2$ in the product, which is why SGD on LoRA is much more sensitive to $\alpha$).
</details>

4. Your packed dataset uses padding-free packing and the position ids are correctly reset. A colleague points out that the model can still tell where a packed example begins. How, and does it matter?

<details><summary>Answer</summary>
Position zero (or the first few positions) carries information: the first tokens of a row see no context, and rotary embeddings make relative position visible, so the model knows it is at a boundary. This is also true of ordinary unpacked training, where every example starts at position zero, so it does not matter for training quality. What would matter is if the model could attend across the boundary, which the block-diagonal mask prevents.
</details>

5. You train with `--load-in-4bit` and observe that eval loss is slightly worse than the bf16 LoRA run at the same settings, and the merged model is worse still. Explain both gaps.

<details><summary>Answer</summary>
The first gap is quantization error in the frozen base: NF4 perturbs $W_0$, the adapter learns on top of the perturbed weights, and the perturbation is not fully absorbed. The second gap comes from merging: `W_0 + BA` is computed with a dequantized $W_0$, and if you then save in bf16 you have a model whose base differs from what the adapter was trained against (the adapter compensated for one set of quantization errors and now sits on a different set). To merge a QLoRA adapter faithfully, merge into the same dequantized weights the training used, or keep the adapter separate at inference.
</details>

6. Spot the bug:

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100
labels[:, :prompt_len] = -100
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1), ignore_index=-100)
```

<details><summary>Answer</summary>
Two problems. `logits` at position $t$ predict token $t+1$, so either the logits must be sliced to `[:, :-1]` and labels to `[:, 1:]`, or the trainer must do the shift internally; as written, each logit is scored against the token at its own position, and the model learns the identity map. Second, `prompt_len` is a single integer, which only works if every example in the batch has the same prompt length; with left padding the prompt does not even start at position zero. The fix is a per-example mask computed in the target frame.
</details>

7. A senior colleague argues that since LoRA freezes the base weights, forgetting is impossible: the original model is recoverable by dropping the adapter. What is right and what is wrong about this?

<details><summary>Answer</summary>
Right: the base weights are untouched, so you can always disable the adapter and get the base model back; that is a real operational advantage. Wrong: forgetting is a property of the model you deploy, and the deployed model is base plus adapter. The adapter can change the function on any input, and the toy shows a 14 percent-of-parameters adapter erasing a task completely. Recoverability and retention are different properties.
</details>

8. Two runs have identical eval NLL. One has 99 percent correct stopping and the other 80 percent. How can the NLL be the same, and which number tells you more?

<details><summary>Answer</summary>
NLL is a per-token average dominated by the many content tokens; the single end-of-turn token per turn contributes almost nothing to the mean even if its probability is much lower in the second run. The behavioral stopping rate is measuring exactly that one token under generation, where it decides whether the answer ends. When they disagree, the behavioral number is the one that predicts what users see. This is also why you should look at the per-token loss of the end-of-turn id separately.
</details>

9. Estimate the number of LoRA parameters for $r = 8$ on all linear layers of a model with hidden 3,584, intermediate 18,944, 28 layers, 4 key-value heads of dimension 128, and compare with $r = 64$ on attention only.

<details><summary>Answer</summary>
Key and value projections map 3,584 to $4 \times 128 = 512$. Per layer, all linear: $(3584 + 3584) + 2(3584 + 512) + (3584 + 3584) + 2(3584 + 18944) + (18944 + 3584) = 7168 + 8192 + 7168 + 45056 + 22528 = 90{,}112$, times $r = 8$ and 28 layers gives about 20.2M. Attention only at $r = 64$: $(7168 + 8192 + 7168) \times 64 \times 28 \approx 40.4$M. The attention-only run has twice the parameters but no adaptable surface in the MLP, which holds most of the model's capacity; parameter count alone is not the right comparison.
</details>

10. Why is the mean-over-batch-tokens normalizer the default in most trainers, and when would you switch to a per-example mean?

<details><summary>Answer</summary>
Per-batch-token averaging makes the loss an unbiased estimate of the per-token NLL over the data distribution, which is what pretraining optimizes and what the eval metric measures, and it gives long answers weight proportional to their tokens. You would switch to per-example averaging when you want each conversation to count equally regardless of length, for example when a few very long examples are dominating the gradient, or when the behavior you care about is decided in the first few tokens of every answer (a format decision or a refusal) and long tails are diluting it.
</details>

## What will change, what will not

The conditional-likelihood argument will not change. Whatever the model architecture, if you are training a system to produce outputs given inputs, the objective is the log-likelihood of outputs given inputs, and including the inputs in the loss is a modeling choice with a cost you can state. Loss masking, the requirement that the stop signal be a target, and the batch-versus-example normalizer are consequences of that argument and will be true of any autoregressive trainer.

The low-rank-update idea will outlive the specific LoRA parameterization. The underlying claim is that the change needed by a short fine-tune lives in a small subspace relative to the model, and that you can exploit that to fit the run on less hardware and to keep the base recoverable. Variants already change the initialization, the scaling with rank, the factorization, and whether the base is quantized; expect the names and the defaults to keep changing, and expect the $\eta \alpha / \sqrt{rk}$ style of scaling argument to remain the way you reason about any of them.

The data loop will not change, and if anything it will matter more. Every generation of models has made the model less of a bottleneck and the examples more of one. The habit of reading every failure and fixing the examples is the durable skill; the tooling around it is not.

What will change: the libraries (Unsloth, TRL, PEFT and their argument names), the recommended learning rates, the packing implementation, the quantization format, and the specific numbers in the memory budget. Treat every default in the recipe as a snapshot. The formulas in section 3 are how you recompute them when the snapshot goes stale.

What is genuinely open: how much of what SFT teaches is format versus capability, how to select or weight examples automatically rather than by hand, and whether the memorization-versus-generalization tradeoff at epoch two can be pushed by anything other than more distinct data. Lab 05 is where you will see that some of what SFT cannot fix, a preference objective can.

## Read next

1. LoRA: Low-Rank Adaptation of Large Language Models, Hu, 2021. The original parameterization, initialization and the $\alpha / r$ scale; read it for the argument that the update is low rank, not the weights.
2. QLoRA: Efficient Finetuning of Quantized LLMs, Dettmers, 2023. NF4, double quantization, and paged optimizers; the reason an 8B fine-tune fits on one consumer card.
3. LIMA: Less Is More for Alignment, Zhou, 2023. A thousand carefully written examples against much larger noisy sets; the strongest case for the curation loop over the volume loop.
4. LoRA Learns Less and Forgets Less, Biderman, 2024. Careful comparison of LoRA and full fine-tuning on code and math, including where LoRA plateaus and what it buys in retention.
5. Training language models to follow instructions with human feedback, Ouyang, 2022. The SFT stage as the first step of the InstructGPT pipeline; read section 3 for how the demonstration data was collected and used.
6. Finetuned Language Models Are Zero-Shot Learners, Wei, 2021. Instruction tuning across many tasks; the origin of mixing many task formats to get generalization to new instructions.
7. A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA, Kalajdzievski, 2023. The $\alpha / \sqrt{r}$ argument and why the default scale breaks at large rank.
8. LoRA Without Regret, Schulman and collaborators at Thinking Machines, 2025. The empirical account of the higher learning rate, the all-layers finding, and where LoRA matches full fine-tuning.
