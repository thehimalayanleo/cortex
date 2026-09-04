"""Shared helpers for the Cortex Training Lab recipes.

Every recipe imports this module. It holds the pieces that are the same
across recipes so each recipe file can stay focused on one algorithm:

  * the stdout protocol the Cortex server parses (METRIC / STATUS / RESULT)
  * seeding and device selection
  * the synthetic corpus that the in-browser lab (lab/index.html) uses, so the
    numbers you see in the browser and in a --smoke run come from the same text
  * a character tokenizer
  * a minimal GPT (RMSNorm, rotary positions, causal attention, SwiGLU) with
    an optional recurrent-depth loop and an optional bidirectional mode for
    encoders; it runs on CPU
  * small training utilities (schedules, batching, checkpoints, generation)
  * statistics helpers (bootstrap CI, Wilson interval)

Nothing here downloads anything. Optional third-party libraries are imported
lazily by the recipes through `require()`, which turns an ImportError into a
message that says what to pip install.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# stdout protocol
# --------------------------------------------------------------------------


def _num(v):
    if isinstance(v, torch.Tensor):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def metric(step: int, **fields) -> None:
    """Print one `METRIC {...}` line. Values should be numbers."""
    rec = {"step": int(step)}
    rec.update({k: _num(v) for k, v in fields.items()})
    print("METRIC " + json.dumps(rec), flush=True)


def status(phase: str, msg: str = "") -> None:
    print("STATUS " + json.dumps({"phase": phase, "msg": msg}), flush=True)


def result(**fields) -> None:
    """Print the single final `RESULT {...}` line."""
    rec = {k: (_num(v) if isinstance(v, (int, float, torch.Tensor)) else v) for k, v in fields.items()}
    print("RESULT " + json.dumps(rec), flush=True)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------
# argparse, seeds, devices, optional imports
# --------------------------------------------------------------------------


def base_parser(recipe: str, description: str) -> argparse.ArgumentParser:
    """The flags shared by every recipe."""
    p = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--out", default=os.path.join("out", recipe), help="directory for checkpoints and artifacts")
    p.add_argument("--steps", type=int, default=None, help="optimizer steps (recipe picks a default if omitted)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto", help="cuda, cpu, mps, or auto")
    p.add_argument("--smoke", action="store_true", help="tiny offline configuration on the synthetic corpus")
    p.add_argument("--max-samples", type=int, default=20000, help="cap on rows loaded from the Hugging Face hub")
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def pick_device(arg: str = "auto") -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def require(module: str, pip_name: str | None = None, extra: str = ""):
    """Import an optional dependency or exit with a message that says what to install."""
    try:
        return __import__(module)
    except ImportError as e:
        pip_name = pip_name or module
        raise SystemExit(
            f"missing optional dependency '{module}': {e}\n"
            f"install it with:  pip install {pip_name}\n{extra}".rstrip()
        )


def autocast_ctx(device: torch.device):
    """bf16 autocast on cuda, no-op elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    import contextlib

    return contextlib.nullcontext()


# --------------------------------------------------------------------------
# synthetic corpus (same text as lab/index.html)
# --------------------------------------------------------------------------

STORIES = """the little fox found a red ball under the tree. she rolled it to the river and it floated away. the frog said, do not be sad, i will get it. the frog swam and pushed the ball back. the fox was happy and they played all day.
a boy named tom had a blue kite. the wind was strong so the kite went high. tom held the string tight. his dog ran and barked at the sky. when the sun went down they walked home.
the cat sat on the warm step. a bird sang in the tall tree. the cat watched and did not move. then the rain came and the cat ran inside. it slept by the fire until morning.
mia lost her hat in the park. her friend sam looked under the bench. the hat was on a duck. they laughed and the duck gave it back. mia said thank you and they shared a cake.
the old bear liked honey more than fish. every day he walked to the big tree. the bees were busy but kind. they gave him a little pot. the bear said this is the best day.
a small robot lived in a shed. it liked to count the stars at night. one night a star fell into the garden. the robot picked it up and it was warm. it kept the star in a jar by the window.
the girl planted a seed in a cup. she gave it water every morning. a green leaf came out, then a flower. she took it to school and everyone smiled. the teacher put it on the sunny sill.
two rabbits raced to the hill. the grey one was fast but the brown one knew a short path. they reached the top together and ate clover. the moon rose and they hopped home side by side.
the boat was tiny and yellow. dad and lily rowed across the pond. a fish jumped and splashed them. lily giggled and dad rowed faster. on the far side they found wild berries.
the wind took the farmer's hat over the fence. the goat caught it on one horn. the farmer laughed and gave the goat an apple. after that the goat wore the hat every windy day."""


def arithmetic_lines(n: int = 160, seed: int = 7, max_operand: int = 20) -> list[str]:
    """The same LCG-driven arithmetic strings the browser lab generates."""
    s = seed
    out = []

    def r():
        nonlocal s
        s = (s * 48271) % 2147483647
        return s % max_operand

    for i in range(n):
        a, b = r(), r()
        op = i % 3
        if op == 0:
            out.append(f"{a} + {b} = {a + b}")
        elif op == 1:
            out.append(f"{a} - {b} = {a - b}")
        else:
            out.append(f"{a} * {b} = {a * b}")
    return out


QA = [
    ("what color is the sky", "blue"), ("what do bees make", "honey"), ("what do fish do", "swim"),
    ("how many legs has a dog", "four"), ("what does a cat say", "meow"),
    ("what is 2 + 3", "5"), ("what is 4 * 2", "8"), ("what is 9 - 4", "5"), ("what is 6 + 6", "12"), ("what is 3 * 3", "9"),
    ("what is warm", "the sun"), ("what falls from clouds", "rain"), ("what do birds do", "sing"), ("what is 7 + 1", "8"), ("what is 5 - 2", "3"),
    ("who rowed the boat", "dad and lily"), ("what did the fox find", "a red ball"), ("what did tom fly", "a blue kite"),
    ("what did the bear like", "honey"), ("what did the girl plant", "a seed"),
]

# (prompt, chosen, rejected): chosen answers are short and correct, rejected are wordy or wrong
PREFS = [
    ("what is 2 + 3", "5", "2 + 3 is a sum and the sum is 6"), ("what color is the sky", "blue", "the sky is green most days"),
    ("what do bees make", "honey", "bees make milk and eggs"), ("what is 4 * 2", "8", "4 * 2 = 6"),
    ("what do fish do", "swim", "fish fly in the sky"), ("how many legs has a dog", "four", "a dog has three legs"),
    ("what is 9 - 4", "5", "it is 4"), ("what does a cat say", "meow", "a cat says woof"),
    ("what falls from clouds", "rain", "cake falls from clouds"), ("what is 6 + 6", "12", "13"),
    ("what did the fox find", "a red ball", "a blue kite"), ("what did tom fly", "a blue kite", "a red ball"),
]

TOPICS = {
    "weather": ["the rain fell all afternoon on the wet roofs", "a cold wind pushed the clouds across the hills", "snow covered the road before dawn", "the storm bent the trees and rattled the windows", "fog sat low over the river until noon", "the sun broke through after the shower", "hail tapped on the tin roof for a minute", "a warm breeze dried the grass by evening", "thunder rolled far off beyond the fields", "frost drew lines on the glass at night", "the sky cleared and the stars came out", "heavy clouds gathered over the bay", "the drizzle turned into a downpour", "ice made the path slick and bright"],
    "cooking": ["she stirred the onions until they turned soft and sweet", "the bread rose slowly in the warm oven", "he salted the water and dropped in the pasta", "the soup simmered with garlic and thyme", "butter melted in the pan and began to foam", "they chopped carrots for the stew", "the cake cooled on the rack by the window", "a pinch of pepper finished the sauce", "rice steamed under a tight lid", "she whisked eggs with a little milk", "the roast rested before he carved it", "lemon juice brightened the dressing", "the pie crust baked to a golden brown", "he toasted seeds for the salad"],
    "machines": ["the engine turned over and settled into a hum", "a gear slipped and the belt began to squeal", "the pump pushed water up through the pipe", "he tightened the bolt with a long wrench", "the motor drew too much current and tripped the fuse", "pistons moved in the cylinder with a steady beat", "the drill bit chewed through the steel plate", "oil dripped from the crankcase onto the floor", "the fan spun faster as the sensor warmed", "a spring returned the lever to its stop", "the compressor kicked on with a thud", "gears meshed in the gearbox under load", "the valve opened and the pressure fell", "the shaft spun true after he balanced it"],
}


def story_sentences() -> list[str]:
    out = []
    for line in STORIES.split("\n"):
        for s in line.split(". "):
            s = s.strip().rstrip(".")
            if s:
                out.append(s)
    return out


def synthetic_text(kind: str = "all") -> str:
    """Concatenated synthetic training text. kind: stories | arith | all."""
    parts = []
    if kind in ("stories", "all"):
        parts.append(STORIES)
    if kind in ("arith", "all"):
        parts.append("\n".join(arithmetic_lines()))
    return "\n".join(parts) + "\n"


ALL_SYNTHETIC_CHARS = sorted(set(
    synthetic_text("all")
    + "".join(q + a for q, a in QA)
    + "".join(p + c + r for p, c, r in PREFS)
    + "".join(s for v in TOPICS.values() for s in v)
    + "abcdefghijklmnopqrstuvwxyz0123456789 .,:?!'\"{}[]()<>/=+-*_\n"
))


# --------------------------------------------------------------------------
# character tokenizer
# --------------------------------------------------------------------------


class CharTokenizer:
    """Maps characters to ids. id 0 is PAD, id 1 is EOS, the rest are characters."""

    PAD, EOS = 0, 1

    def __init__(self, chars: Sequence[str] | None = None):
        chars = list(chars) if chars is not None else ALL_SYNTHETIC_CHARS
        self.itos = ["<pad>", "<eos>"] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.eos_id = self.EOS
        self.pad_id = self.PAD

    def encode(self, s: str, add_eos: bool = False) -> list[int]:
        ids = [self.stoi[c] for c in s if c in self.stoi]
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.EOS and stop_at_eos:
                break
            if i >= 2:
                out.append(self.itos[i])
        return "".join(out)

    def to_dict(self) -> dict:
        return {"kind": "char", "chars": self.itos[2:]}

    @classmethod
    def from_dict(cls, d: dict) -> "CharTokenizer":
        return cls(d["chars"])


class TiktokenWrapper:
    """Thin wrapper so a tiktoken encoding has the same interface as CharTokenizer."""

    def __init__(self, name: str = "gpt2"):
        tiktoken = require("tiktoken")
        self.enc = tiktoken.get_encoding(name)
        self.name = name
        self.eos_id = self.enc.eot_token
        self.pad_id = self.eos_id
        self.vocab_size = self.enc.n_vocab

    def encode(self, s: str, add_eos: bool = False) -> list[int]:
        ids = self.enc.encode_ordinary(s)
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids: Iterable[int], stop_at_eos: bool = True) -> str:
        ids = [int(i) for i in ids]
        if stop_at_eos and self.eos_id in ids:
            ids = ids[: ids.index(self.eos_id)]
        return self.enc.decode(ids)

    def to_dict(self) -> dict:
        return {"kind": "tiktoken", "name": self.name}


def tokenizer_from_dict(d: dict):
    if d["kind"] == "char":
        return CharTokenizer.from_dict(d)
    if d["kind"] == "tiktoken":
        return TiktokenWrapper(d["name"])
    raise ValueError(f"unknown tokenizer kind {d['kind']}")


# --------------------------------------------------------------------------
# minimal GPT
# --------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int
    n_layer: int = 2
    d_model: int = 64
    n_head: int = 4
    seq_len: int = 128
    loop: int = 1          # apply the block stack this many times (recurrent depth) with input injection
    causal: bool = True    # False gives a bidirectional encoder
    dropout: float = 0.0
    tie_embeddings: bool = True


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


def rope_cache(seq_len: int, head_dim: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """cos and sin tables of shape (seq_len, head_dim // 2)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, S, D). Rotates pairs (x[..., :D/2], x[..., D/2:]) by position-dependent angles."""
    S = x.shape[-2]
    cos = cos[:S].to(x.dtype)[None, None]
    sin = sin[:S].to(x.dtype)[None, None]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.causal = cfg.causal
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, S, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if attn_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0)
        else:
            # attn_mask: (B, S) bool, True where the key is a real token. Combine with causal if needed.
            m = attn_mask[:, None, None, :].expand(B, 1, S, S)
            if self.causal:
                m = m & torch.ones(S, S, dtype=torch.bool, device=x.device).tril()[None, None]
            # a padded query row would be fully masked and produce NaN; let every position see itself
            m = m | torch.eye(S, dtype=torch.bool, device=x.device)[None, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=self.dropout if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, S, C))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or int(8 * d / 3)
        hidden = (hidden + 7) // 8 * 8
        self.w_gate = nn.Linear(d, hidden, bias=False)
        self.w_up = nn.Linear(d, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg.d_model)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.n1(x), cos, sin, attn_mask)
        return x + self.mlp(self.n2(x))


class GPT(nn.Module):
    """Pre-norm decoder (or encoder when cfg.causal is False).

    With cfg.loop = T the block stack is applied T times. Before each pass after
    the first, the token embedding is added back to the hidden state ("input
    injection"), which is what lets a recurrent-depth model keep track of the
    input across iterations. Parameter count does not change with T; compute
    per token grows linearly with T.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = rope_cache(cfg.seq_len, cfg.d_model // cfg.n_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def hidden(self, idx: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        S = idx.shape[1]
        assert S <= self.cfg.seq_len, f"sequence {S} longer than seq_len {self.cfg.seq_len}"
        e = self.tok_emb(idx)
        h = e
        for t in range(self.cfg.loop):
            if t > 0:
                h = h + e
            for blk in self.blocks:
                h = blk(h, self.rope_cos, self.rope_sin, attn_mask)
        return self.norm(h)

    def forward(self, idx: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.lm_head(self.hidden(idx, attn_mask))

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Forward+backward training FLOPs per token, 6 * N_non_embedding * loop + attention term.

        The attention term is 12 * n_layer * d_model * seq_len per token (Kaplan et al.
        style accounting of the S x S score and value products), also scaled by loop.
        """
        N = self.num_params(non_embedding=True)
        c = self.cfg
        return c.loop * (6 * N + 12 * c.n_layer * c.d_model * c.seq_len)


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=ignore_index)


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None,
             eos_id: int | None = None, greedy: bool = False, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Sample continuations for a batch of prompts (B, S). Stops early once every row has emitted eos.

    Prompts of different lengths can be batched by LEFT-padding them and passing
    attn_mask (B, S) with False at pad positions (see left_pad). Rotary positions
    are relative, so a shifted start does not change what the real tokens see.
    """
    model.eval()
    B = idx.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=idx.device)
    for _ in range(max_new_tokens):
        ctx = idx[:, -model.cfg.seq_len:]
        m = attn_mask[:, -model.cfg.seq_len:] if attn_mask is not None else None
        logits = model(ctx, attn_mask=m)[:, -1, :].float()
        if greedy or temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
        if eos_id is not None:
            nxt = torch.where(done[:, None], torch.full_like(nxt, eos_id), nxt)
        idx = torch.cat([idx, nxt], 1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.ones(B, 1, dtype=torch.bool, device=idx.device)], 1)
        if eos_id is not None:
            done |= nxt[:, 0] == eos_id
            if bool(done.all()):
                break
    return idx


def left_pad(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad id lists for batched generation. Returns (ids, mask) with mask True at real tokens."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, L - len(s):] = True
    return ids, mask


# --------------------------------------------------------------------------
# training utilities
# --------------------------------------------------------------------------


def lr_at(step: int, total: int, peak: float, warmup: int, min_ratio: float = 0.1, schedule: str = "cosine",
          cooldown_frac: float = 0.2) -> float:
    """Warmup then cosine decay to peak*min_ratio, or warmup-stable-decay (wsd) with a linear cooldown."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if schedule == "cosine":
        p = (step - warmup) / max(1, total - warmup)
        return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
    if schedule == "wsd":
        start = int(total * (1 - cooldown_frac))
        if step < start:
            return peak
        p = (step - start) / max(1, total - start)
        return peak * (1 - (1 - min_ratio) * min(1.0, p))
    if schedule == "constant":
        return peak
    raise ValueError(schedule)


def random_windows(data: torch.Tensor, batch: int, seq_len: int, gen: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch` windows of length seq_len+1 from a 1-D token tensor. Returns (x, y)."""
    n = data.numel() - seq_len - 1
    assert n > 0, f"corpus has {data.numel()} tokens, need more than seq_len+1={seq_len + 1}"
    ix = torch.randint(0, n, (batch,), generator=gen)
    x = torch.stack([data[i: i + seq_len] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + seq_len] for i in ix])
    return x, y


def pad_batch(seqs: list[list[int]], pad_id: int, max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad a list of id lists. Returns (ids, mask) with mask True at real tokens."""
    L = max_len or max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, s in enumerate(seqs):
        s = s[:L]
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return ids, mask


def make_adamw(model: nn.Module, lr: float, weight_decay: float = 0.1, betas=(0.9, 0.95)) -> torch.optim.AdamW:
    """Weight decay on matrices only, none on norms/embeddings-as-vectors (the usual GPT split)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    return torch.optim.AdamW([
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=lr, betas=betas)


def save_checkpoint(path: str, model: GPT, tokenizer, step: int, extra: dict | None = None) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "config": asdict(model.cfg),
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "extra": extra or {},
    }, path)
    return path


def load_checkpoint(path: str, device: torch.device, **cfg_overrides) -> tuple[GPT, object, dict]:
    ck = torch.load(path, map_location="cpu")
    cfg = GPTConfig(**{**ck["config"], **cfg_overrides})
    model = GPT(cfg)
    model.load_state_dict(ck["model"], strict=False)
    tok = tokenizer_from_dict(ck["tokenizer"])
    return model.to(device), tok, ck


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.t
        self.t = now
        return dt


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Percentile bootstrap CI of the mean. Returns (mean, lo, hi)."""
    vals = torch.tensor(list(values), dtype=torch.float64)
    n = vals.numel()
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    means = vals[idx].mean(1)
    lo = torch.quantile(means, alpha / 2).item()
    hi = torch.quantile(means, 1 - alpha / 2).item()
    return vals.mean().item(), lo, hi


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Wilson score interval for a binomial proportion. Returns (p_hat, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


# --------------------------------------------------------------------------
# transformers Trainer bridge (imported lazily by the real-mode recipes)
# --------------------------------------------------------------------------


def make_metric_callback():
    """Return a transformers TrainerCallback that re-emits Trainer logs as METRIC lines."""
    transformers = require("transformers")

    class MetricCallback(transformers.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            fields = {k.replace("/", "_"): v for k, v in logs.items() if isinstance(v, (int, float))}
            metric(state.global_step, **fields)

        def on_train_begin(self, args, state, control, **kw):
            status("train", "trainer started")

        def on_train_end(self, args, state, control, **kw):
            status("done", "trainer finished")

    return MetricCallback()


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
