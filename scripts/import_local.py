#!/usr/bin/env python3
"""Build the initial vault from local data only: triaged PDFs (queue.json), Bear notes, and the project/topic seeds.
Idempotent: re-running updates metadata but never duplicates files.

  CORTEX_VAULT=~/Cortex python3.11 scripts/import_local.py --queue /path/queue.json --bear /path/bear_notes.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from server import vault  # noqa: E402

TOPICS = [
    {"slug": "pursuit", "name": "Sparse & pursuit methods", "kind": "method", "one_liner": "OMP, Hummingbird, CAMP: greedy selection as the unifying tool across CL, interp, KV, and reward."},
    {"slug": "continual-learning", "name": "Continual learning", "kind": "field", "one_liner": "Capacity, forgetting, plasticity, memory layers; where sparse updates actually help."},
    {"slug": "interpretability", "name": "Interpretability", "kind": "field", "one_liner": "SAEs, probes, circuits, parameter decomposition; what the model is actually computing."},
    {"slug": "safety", "name": "Alignment & safety", "kind": "field", "one_liner": "Refusal, control, scheming probes, unlearning, agent security."},
    {"slug": "post-training", "name": "Post-training & RL", "kind": "field", "one_liner": "RLHF, RLVR, rewards, distillation, LoRA recipes."},
    {"slug": "gpu", "name": "GPU kernels & inference", "kind": "method", "one_liner": "CUDA/Triton/Gluon, batched linalg, KV compression, serving at scale."},
    {"slug": "pretraining", "name": "Pretraining & optimizers", "kind": "field", "one_liner": "Speedruns, Muon variants, param-efficient from-scratch training."},
    {"slug": "generative-media", "name": "Generative media", "kind": "field", "one_liner": "Diffusion, video, splats, the animation studio stack."},
    {"slug": "competitions", "name": "Competitions", "kind": "life", "one_liner": "Kaggle, GPU MODE, hackathons: deadlines, rules, leaderboard facts."},
    {"slug": "career", "name": "Career", "kind": "life", "one_liner": "Frontier-lab interviews, fellowships, decks, networking."},
]
NOTION_TOPIC_IDS = {  # ids used in the earlier Notion pass -> vault slugs
    "3cf185b1e3cc8191a10fec85067ea375": "pursuit", "3cf185b1e3cc8133b2d7ce6a4b11d9de": "continual-learning",
    "3cf185b1e3cc81ed8cebdeecf40b9887": "interpretability", "3cf185b1e3cc814cb4faf91f834996bd": "safety",
    "3cf185b1e3cc810ea2dff27fea7baf7c": "post-training", "3cf185b1e3cc81dba03edb560b6bf253": "gpu",
    "3cf185b1e3cc811382c9f525473ee5d0": "pretraining", "3cf185b1e3cc81a892e4ff64fe5bdcf4": "generative-media",
    "3cf185b1e3cc812b80abd2ac07e48719": "competitions", "3cf185b1e3cc81e299fad1a06a0266f7": "career",
}

PROJECTS = [
    ("Sparse-memory CL (CAMP)", "active", "research", "Collision-aware matching pursuit Pareto-dominates FAIR TF-IDF and KL-surprise at matched acquisition: +59% early-fact retention, 16 seeds, frozen SmolLM2 + PKM.", "Real Berges memory-layer arch + TriviaQA/NQ QA for a COLM 2027 / workshop paper.", None, "thepursuits/cl_frontier/", ["pursuit", "continual-learning"]),
    ("SVD-OMP parameter decomposition", "paused", "research", "Training-free decomp Pareto-wins VPD on 18/24 Goodfire-67M matrices and 24/24 on causal ablation. Unsubmitted.", "Write the theory section, email Lee Sharkey, pick a live venue.", None, "github.com/thehimalayanleo/svd-omp", ["pursuit", "interpretability"]),
    ("Refusal-Pursuit", "active", "research", "Per-writer readout atoms localize refusal to late MLP/attn writers across OLMo, SmolLM2, Qwen2.5-3B (1.00→0.03 with 32 edits); MMLU intact, no over-refusal. SVD atoms are the wrong basis.", "Decide: paper or bank. Contrast/coverage criterion, not OMP, for ablation.", None, "thepursuits/refusal_pursuit/", ["safety", "pursuit", "interpretability"]),
    ("Celwright animation studio", "active", "build", "Director loop runs end-to-end on the 5090 (Qwen3-VL + 3 critics). Identity lock 0.90 is the lever; regional masks null; zero-shot loop = null vs reseed.", "Next identity lever after regional masks; Wan brick.", None, "thepursuits/celwright/", ["generative-media"]),
    ("GPU MODE Cholesky (B200)", "done", "competition", "Rank 3 of 88 at 323.95µs. Geomean is dominated by the smallest cases; that was the lever.", None, "2026-07-30", "gpu-mode-linalg/linalg/cholesky/", ["gpu", "competitions"]),
    ("GPU MODE eigh (B200)", "done", "competition", "Rank 23 of 109 at 31,825µs; leader ~5x faster. Lever was projected-SYTRD speed via Gluon tcgen05.", None, "2026-07-15", "gpu-mode-linalg/linalg/eigh/", ["gpu", "competitions"]),
    ("NeuroGolf 2026 (Kaggle)", "active", "competition", "Best real LB 7246.15 from untouched top fork + grader-verified bit-parallel solvers. Rule: fork and submit untouched, never blend via local scoring.", None, None, "thepursuits/neurogolf/", ["competitions"]),
    ("AIMO Interp Challenge (NeurIPS 2026)", "active", "competition", "Per-problem robust-vs-spurious classifier for DeepSeek-8B; must beat all-False 0.77 and uncertainty 0.69.", "Build the interp-feature + behavior classifier.", "2026-11-01", "thepursuits/aimo_interp/", ["interpretability", "competitions"]),
    ("Pursuit Tickets pretraining", "active", "research", "Iterative MLP shrink by span-coverage instead of magnitude/Taylor. Gate 0 passed; harness runs.", "Gate 1: real data on the 5090, Pareto vs Bonsai/Liquid.", None, "thepursuits/pursuit_tickets/", ["pretraining", "pursuit"]),
    ("Circuit Pursuit", "paused", "research", "Matching pursuit in the QK/OV basis. Hummingbird beats OMP 17/0/3 at k=8 on 1.7B sentiment (0.758 vs 0.689); the defect is a readout ceiling at both scales.", None, None, "thepursuits/circuit_pursuit/", ["interpretability", "pursuit"]),
    ("Beat-SAM", "banked", "research", "Cannot beat SAM by reshaping it (flatness ceiling). SAM+GCE beats SAM by +1.9pp under 40% label noise only; SAM wins clean.", None, None, "thepursuits/sam_pursuit/", ["pretraining"]),
    ("Superposition CL capacity", "banked", "research", "Superposition breaks the CL dimension wall ~4x, but this is compressed sensing shown cleanly; Hummingbird edge is null on real SAE features (coherence ≈ random).", None, None, "thepursuits/superposition_cl/", ["continual-learning", "interpretability"]),
    ("Memory-layer unlearning", "refuted", "research", "Memory-layer deletion preserves general capability (+0 ppl) but the deep-forgetting / relearn-robustness hypothesis failed on synthetic and TOFU. Do not re-chase.", None, None, "thepursuits/cl_frontier/", ["safety", "continual-learning"]),
    ("Pursuit-Gated Attention (KV)", "refuted", "research", "Headline claims did not reproduce; dominated by top-k. Real finding: coverage selection gives retrieval-robust KV (snap_cover dominates SnapKV).", None, None, "thepursuits/pga/", ["gpu", "pursuit"]),
    ("Frontier-lab interview prep", "active", "career", "Self-built prep SPA (learn + graders + knowledge graph) for Anthropic / OpenAI / Jane Street; Pyodide now, 5090 executor later.", "Keep the daily loop; CF deploy deferred.", None, "thepursuits/interview_prep/", ["career"]),
    ("Weaver Scrape-Verse (WeMakeDevs)", "paused", "competition", "Self-healing scraper built and e2e-tested in weaver/. Submission status after the Aug 23 deadline unconfirmed.", None, "2026-08-23", "thepursuits/weaver/", ["competitions"]),
    ("Cortex (this app)", "active", "build", "Local second brain: vault of markdown + PDFs, FastAPI server, React front-end, chat over OpenCode Go with vault tools, agents run inside the vault.", "Use it daily for a week; then the nightly agent loop.", None, "thepursuits/cortex/", ["career"]),
]

SKIP = re.compile(r"resume|résumé|\bcv\b|curriculum|Mulay|Ravikumar|Hennes|Dungarwal|Zimmer|Sean Lane|offer|visa|EB-?1|I-?20|transcript|reviews?|thesis|Hameed|E1B1|invoice|receipt|Form 13|Exit-Survey|NSF_award|Critical nature|signed|hotcrp|Submission|review-folder|template|formatting instructions|HOWTO|example_paper|Cover-Letter|Personal Statement|I-Corps|Happyou|Volume \d|Total Citations|Award Details|Statistics Surveys|ORIGINAL RESEARCH|External Email|Team Name", re.I)
DRAFT_DUPES = re.compile(r"HummingBird_(Mar|Neurips|First)|FoBA_OLS|ICML_Least_Squares|TMLR__OMP|HummingBird__ICML", re.I)


def import_papers(queue: Path, limit: int | None) -> int:
    rows = json.loads(queue.read_text())
    n = 0
    seen: set[str] = set()
    for r in rows:
        if SKIP.search(r["path"]) or SKIP.search(r["title"]) or DRAFT_DUPES.search(r["path"]):
            continue
        src = Path("~").expanduser() / r["path"]
        if not src.exists() or src.stat().st_size > 60 << 20:
            continue
        key = r["arxiv"] or vault.slugify(r["title"])[:40]
        if key in seen:
            continue
        seen.add(key)
        topics = [NOTION_TOPIC_IDS.get(t, t) for t in r.get("topics", [])]
        meta = {"arxiv": r["arxiv"], "title": r["title"], "authors": r["authors"], "year": r["year"], "abstract": r.get("abstract", ""),
                "link": f"https://arxiv.org/abs/{r['arxiv']}" if r["arxiv"] else "", "topics": topics,
                "type": "paper" if r["arxiv"] else "doc", "status": "inbox"}
        if re.search(r"own|Mulay", r.get("authors", ""), re.I):
            meta["status"] = "reference"
        try:
            vault.ingest_pdf(src, meta, pid=r["arxiv"] or None)
            n += 1
        except Exception as e:
            print("skip", r["path"], e)
        if limit and n >= limit:
            break
    return n


def import_bear(bear: Path) -> int:
    n = 0
    for note in json.loads(bear.read_text()):
        title = note["title"].strip()
        body = re.sub(r"^# .*\n", "", note["text"] or "", count=1).strip()
        if not body or title.lower() == "stuff":
            continue
        slug = vault.slugify(title)
        if vault.get_note(slug):
            continue
        vault.create_note(title, "fleeting", body, [], source="bear", imported=note["modified"])
        n += 1
    return n


def seed() -> None:
    (vault.VAULT / "topics.json").write_text(json.dumps(TOPICS, indent=1, ensure_ascii=False))
    for title, status, ptype, verdict, nxt, deadline, repo, topics in PROJECTS:
        slug = vault.slugify(title)
        fm = {"status": status, "type": ptype, "verdict": verdict, "next_action": nxt, "deadline": deadline, "repo": repo, "topics": topics}
        if vault.get_project(slug):
            vault.update_project(slug, fm)
        else:
            vault.create_project(title, "", **fm)
    if not vault.get_note("rlhf-in-four-equations"):
        vault.create_note("RLHF in four equations", "permanent", RLHF, ["post-training"])
    if not vault.get_note("how-cortex-works"):
        vault.create_note("How Cortex works", "permanent", HOWTO, [])


RLHF = """## Modern RLHF timeline
TAMER → preference RL → Ziegler 2019 → InstructGPT → Constitutional AI / RLAIF → DPO → Llama / Tulu → DeepSeek R1 / RLVR.

## Reward modeling
Train $r_\\phi(x, y)$ from pairwise preferences.

$$P(y_w \\succ y_l \\mid x) = \\sigma\\left(r_\\phi(x, y_w) - r_\\phi(x, y_l)\\right)$$

$$\\mathcal{L}_{RM}(\\phi) = -\\log \\sigma\\left(r_\\phi(x, y_w) - r_\\phi(x, y_l)\\right)$$

Reward models learn relative preference rankings, not absolute truth.

## RL against the reward model
Optimize reward while staying close to the reference model.

$$J(\\pi) = \\mathbb{E}_{x,y}\\left[r_\\phi(x,y)\\right] - \\beta\\, D_{KL}\\left(\\pi \\parallel \\pi_{ref}\\right)$$

The KL term keeps the updated policy close to the SFT / reference model.

## Post-training intuition
Post-training mostly elicits and surfaces knowledge learned during pretraining rather than creating base capabilities from scratch.
"""

HOWTO = """One vault, plain files. Anything that can read a folder can read your brain.

- **Notes** (`notes/`): fleeting is the inbox; promote what survives to permanent. Markdown with `$LaTeX$`.
- **Daily** (`daily/YYYY-MM-DD.md`): one page a day. Dump, then promote.
- **Library** (`library/<id>/`): the PDF, `meta.json`, your `notes.md`. Status inbox → reading → read → reference.
- **Projects** (`projects/`): one line of verdict, one next action. Update, don't append.
- **Chat**: grounded in the vault; it searches before it answers and writes only when you ask. Every tool call shows in the ledger.
- **Agents**: "hand to Codex / OpenCode" runs the agent inside the vault folder with file tools.

Weekly, 20 minutes: empty the library inbox, promote or delete fleeting notes, refresh every active project's verdict.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", type=Path)
    ap.add_argument("--bear", type=Path)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    vault.ensure_dirs()
    seed()
    print("seeded topics + projects + starter notes")
    if a.bear and a.bear.exists():
        print("bear notes:", import_bear(a.bear))
    if a.queue and a.queue.exists():
        print("papers:", import_papers(a.queue, a.limit))
    print("index:", vault.rebuild_index(), "docs")
    print("counts:", vault.counts(), "vault:", vault.VAULT)


if __name__ == "__main__":
    main()
