"""Seed a public demo vault (used when CORTEX_DEMO=1 and the vault is empty). Public arXiv papers only."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import vault  # noqa: E402

PAPERS = [  # (arxiv id, title, status, rating, takeaway, topics)
    ("2502.03407", "Detecting Strategic Deception Using Linear Probes", "read", 4, "Linear probes on activations detect strategic deception where output monitoring alone fails.", ["safety", "interpretability"]),
    ("2511.16035", "Liars' Bench: Evaluating Lie Detectors for Language Models", "read", 4, "Lie detectors are usually validated in narrow settings; this benchmark spans the lies LLMs actually produce.", ["safety"]),
    ("2507.11473", "Chain of Thought Monitorability: A New and Fragile Opportunity for AI Safety", "read", 5, "CoT monitoring is a real but fragile safety lever; training pressure can erode it.", ["safety"]),
    ("2511.13653", "Weight-sparse transformers have interpretable circuits", "inbox", None, "", ["interpretability", "pursuit"]),
    ("2510.04871", "Less is More: Recursive Reasoning with Tiny Networks", "read", 5, "A 7M-parameter recursive network beats far larger models on ARC-style reasoning.", ["post-training"]),
    ("2512.24695", "Nested Learning: The Illusion of Deep Learning Architectures", "inbox", None, "", ["continual-learning"]),
    ("2505.06708", "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", "reading", None, "", ["pretraining", "gpu"]),
    ("2310.01405", "Representation Engineering: A Top-Down Approach to AI Transparency", "reference", 4, "Population-level representations, not neurons, as the unit of analysis for transparency.", ["interpretability"]),
]

TOPICS = [
    {"slug": "pursuit", "name": "Sparse & pursuit methods", "kind": "method", "one_liner": "Greedy selection as a unifying tool."},
    {"slug": "continual-learning", "name": "Continual learning", "kind": "field", "one_liner": "Capacity, forgetting, memory layers."},
    {"slug": "interpretability", "name": "Interpretability", "kind": "field", "one_liner": "Probes, SAEs, circuits."},
    {"slug": "safety", "name": "Alignment & safety", "kind": "field", "one_liner": "Deception, control, monitoring."},
    {"slug": "post-training", "name": "Post-training & RL", "kind": "field", "one_liner": "RLHF, RLVR, distillation."},
    {"slug": "gpu", "name": "GPU kernels & inference", "kind": "method", "one_liner": "Kernels, KV, serving."},
    {"slug": "pretraining", "name": "Pretraining & optimizers", "kind": "field", "one_liner": "Speedruns and optimizers."},
]

PROJECTS = [
    ("Deception probes replication", "active", "research", "Multi-layer probe ensembles replicate; single-layer probes are fragile across models.", "Run the 32B sweep."),
    ("Tiny recursive reasoners", "paused", "research", "TRM reproduces on Sudoku; ARC transfer unclear.", "Try depth 3 with the ARC-AGI-2 split."),
    ("Cortex", "active", "build", "Local second brain with WebMCP tools so a browser agent can read, file, and write alongside you.", "Ship the demo."),
]

NOTE = """## Why probes scale
Linear probe accuracy on deception rises with model size, and multi-layer ensembling removes the "which layer" fragility.

The probe is a direction $w$ in residual space; the score is $\\sigma(w^\\top h_\\ell + b)$. Ensembling averages over layers $\\ell$:

$$p(\\text{deceptive}) = \\frac{1}{L}\\sum_{\\ell} \\sigma(w_\\ell^\\top h_\\ell + b_\\ell)$$

Open question: does the same direction survive RL post-training?
"""


def seed() -> None:
    import json
    vault.ensure_dirs()
    if vault.counts()["papers"] > 0:
        return
    (vault.VAULT / "topics.json").write_text(json.dumps(TOPICS, indent=1))
    for title, status, ptype, verdict, nxt in PROJECTS:
        vault.create_project(title, "", status=status, type=ptype, verdict=verdict, next_action=nxt, topics=[])
    vault.create_note("Why probes scale", "permanent", NOTE, ["interpretability", "safety"])
    vault.create_note("How Cortex works", "permanent",
        "Plain files, one vault. Notes with LaTeX, a PDF library, projects with one-line verdicts, a daily page.\n\n"
        "The chat and the browser agent share the same tools: search the brain, read or file a paper, write a note, update a project. "
        "Every tool call shows in the ledger, so you always see what the agent did.", [])
    vault.append_daily("Demo vault seeded. Try: ask the agent which papers are about deception, then have it file a new arXiv paper.")
    for aid, title, status, rating, takeaway, topics in PAPERS:
        try:
            vault.ingest_arxiv(aid, title=title, status=status, rating=rating, takeaway=takeaway, topics=topics)
        except Exception as e:  # keep going; a missing paper is not fatal for a demo
            print("demo seed skip", aid, e)
    vault.rebuild_index()


if __name__ == "__main__":
    seed()
    print(vault.counts())
