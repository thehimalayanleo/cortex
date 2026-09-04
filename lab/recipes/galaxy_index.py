#!/usr/bin/env python3
"""Galaxy index: embed every paper in the vault, cluster them into solar systems, group those into universes.

Cheap and local: bge-small-en-v1.5 through sentence-transformers on the CPU (about 30 ms per paper), DBSCAN on the
unit vectors (cosine distance), and agglomerative grouping of the cluster centroids. Labels come from the titles'
distinctive words. Positions are a PCA to 2D and 3D, then a light repulsion pass so points do not stack.

    python galaxy_index.py --vault ~/Cortex --out ~/Cortex/.cortex/galaxy.json
    python galaxy_index.py --smoke            # hashed bag-of-words embeddings, no model download

Prints METRIC / STATUS / RESULT lines. The JSON it writes is what the Galaxy view draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from common import metric, status, result  # type: ignore
except Exception:
    def _p(tag, **kw): print(tag + " " + json.dumps(kw), flush=True)  # noqa: E306
    def metric(**kw): _p("METRIC", **kw)  # noqa: E306
    def status(**kw): _p("STATUS", **kw)  # noqa: E306
    def result(**kw): _p("RESULT", **kw)  # noqa: E306

STOP = set("""a an the of for and or in on to with from by at as is are be we our this that these those into via its their using use based
towards toward over under between beyond than then which what how why when where can could should may might do does did not no
paper papers model models method methods approach approaches learning large language new novel via""".split())


def load_papers(vault: Path, max_chars: int) -> list[dict]:
    out = []
    for d in sorted((vault / "library").iterdir()):
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = json.loads(mp.read_text())
        except Exception:
            continue
        text = ""
        tp = d / "text.txt"
        if tp.exists():
            try:
                text = tp.read_text(errors="ignore")[:max_chars]
            except Exception:
                text = ""
        out.append({"id": d.name, "title": str(m.get("title") or d.name), "authors": m.get("authors") or "", "year": m.get("year"),
                    "status": m.get("status") or "inbox", "topics": m.get("topics") or [], "takeaway": m.get("takeaway") or "", "text": text})
    return out


def doc_text(p: dict) -> str:
    return f"{p['title']}. {p['takeaway']} {p['text'][:1500]}".strip()


def embed_hashed(texts: list[str], dim: int = 256) -> np.ndarray:
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in re.findall(r"[a-z][a-z0-9\-]{2,}", t.lower()):
            if w in STOP:
                continue
            h = int(hashlib.md5(w.encode()).hexdigest(), 16)
            X[i, h % dim] += 1.0 if (h >> 8) & 1 else -1.0
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    return X / n


def embed_model(texts: list[str], model: str, batch: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model, device="cpu")
    t0 = time.time()
    X = st.encode(texts, batch_size=batch, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
    metric(step=1, embedded=len(texts), seconds=time.time() - t0, ms_per_doc=1000 * (time.time() - t0) / max(1, len(texts)))
    return X.astype(np.float32)


def pca(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / max(1e-9, (S ** 2).sum())
    return Xc @ Vt[:k].T, var[:k]


def dbscan_cosine(X: np.ndarray, target_lo: int, target_hi: int) -> tuple[np.ndarray, float]:
    """DBSCAN with cosine distance; eps searched so the number of clusters lands in [target_lo, target_hi] when possible."""
    from sklearn.cluster import DBSCAN
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    best = None
    for eps in np.linspace(0.08, 0.5, 22):
        lab = DBSCAN(eps=float(eps), min_samples=3, metric="precomputed").fit_predict(D)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float((lab == -1).mean())
        score = (0 if target_lo <= k <= target_hi else min(abs(k - target_lo), abs(k - target_hi))) + noise * 2
        if best is None or score < best[0]:
            best = (score, lab, float(eps), k, noise)
        metric(step=int(eps * 100), eps=float(eps), clusters=k, noise_frac=noise)
    _, lab, eps, k, noise = best
    status(phase="cluster", msg=f"dbscan eps={eps:.3f}: {k} clusters, {noise:.0%} noise")
    return lab, eps


def label_for(titles: list[str], global_df: Counter, n_docs: int, k: int = 3) -> str:
    tf = Counter()
    for t in titles:
        for w in set(re.findall(r"[a-z][a-z0-9\-]{2,}", t.lower())):
            if w not in STOP:
                tf[w] += 1
    scored = {w: c * math.log((n_docs + 1) / (global_df[w] + 1)) for w, c in tf.items()}
    top = sorted(scored, key=lambda w: -scored[w])[:k]
    return " · ".join(top) if top else "misc"


def spread(P: np.ndarray, iters: int = 60, radius: float = 0.035) -> np.ndarray:
    """A few repulsion steps in 2D so overlapping points separate; positions are in [-1, 1]."""
    P = P.copy()
    scale = np.abs(P).max() or 1.0
    P /= scale
    for _ in range(iters):
        d = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(d, axis=-1) + 1e-6
        push = np.clip(radius - dist, 0, None) / dist
        np.fill_diagonal(push, 0.0)
        P += 0.5 * (d * push[..., None]).sum(1)
    return np.clip(P, -1.05, 1.05)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vault", default=os.environ.get("CORTEX_VAULT", "~/Cortex"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--clusters", default="8-30", help="wanted cluster count range")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=0, help="ignored; accepted for the run protocol")
    a = ap.parse_args()
    vault = Path(a.vault).expanduser()
    out = Path(a.out or vault / ".cortex" / "galaxy.json").expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    papers = load_papers(vault, a.max_chars)
    if not papers:
        raise SystemExit("no papers in the library")
    status(phase="load", msg=f"{len(papers)} papers")
    texts = [doc_text(p) for p in papers]
    if a.smoke:
        X = embed_hashed(texts)
        model = "hashed-bag-of-words"
    else:
        try:
            X = embed_model(texts, a.model, a.batch)
            model = a.model
        except Exception as e:
            status(phase="fallback", msg=f"model unavailable ({e}); hashed embeddings")
            X = embed_hashed(texts)
            model = "hashed-bag-of-words"
    lo, hi = (int(v) for v in a.clusters.split("-"))
    labels, eps = dbscan_cosine(X, lo, hi)
    P2, var2 = pca(X, 2)
    P3, var3 = pca(X, 3)
    P2 = spread(P2)
    P3 = P3 / (np.abs(P3).max() or 1.0)
    df = Counter()
    for p in papers:
        for w in set(re.findall(r"[a-z][a-z0-9\-]{2,}", p["title"].lower())):
            df[w] += 1
    # solar systems
    ids = sorted(set(labels))
    clusters = []
    for cid in ids:
        idx = np.where(labels == cid)[0]
        titles = [papers[i]["title"] for i in idx]
        clusters.append({"id": int(cid), "label": "unclustered" if cid == -1 else label_for(titles, df, len(papers)), "size": int(len(idx)),
                         "cx": float(P2[idx, 0].mean()), "cy": float(P2[idx, 1].mean()), "centroid": X[idx].mean(0)})
    # universes: agglomerative grouping of cluster centroids (cosine), about sqrt(k) groups
    real = [c for c in clusters if c["id"] != -1]
    universes = []
    if len(real) >= 3:
        from sklearn.cluster import AgglomerativeClustering
        C = np.stack([c["centroid"] for c in real])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        nU = max(2, int(round(math.sqrt(len(real)))))
        ul = AgglomerativeClustering(n_clusters=nU, metric="cosine", linkage="average").fit_predict(C)
        for u in sorted(set(ul)):
            members = [real[i] for i in range(len(real)) if ul[i] == u]
            titles = [papers[i]["title"] for c in members for i in np.where(labels == c["id"])[0]]
            for c in members:
                c["universe"] = int(u)
            universes.append({"id": int(u), "label": label_for(titles, df, len(papers), k=2), "clusters": [c["id"] for c in members], "size": sum(c["size"] for c in members)})
    for c in clusters:
        c.pop("centroid", None)
        c.setdefault("universe", -1)
    umap = {c["id"]: c.get("universe", -1) for c in clusters}
    rows = []
    for i, p in enumerate(papers):
        rows.append({"id": p["id"], "title": p["title"], "year": p["year"], "status": p["status"], "topics": p["topics"], "authors": str(p["authors"])[:120],
                     "x": float(P2[i, 0]), "y": float(P2[i, 1]), "x3": float(P3[i, 0]), "y3": float(P3[i, 1]), "z3": float(P3[i, 2]),
                     "cluster": int(labels[i]), "universe": umap.get(int(labels[i]), -1)})
    # nearest neighbours per paper (for "planets near this one")
    S = X @ X.T
    for i, r in enumerate(rows):
        nn = np.argsort(-S[i])[1:6]
        r["near"] = [[papers[j]["id"], float(S[i, j])] for j in nn]
    data = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "eps": eps, "n": len(rows), "var2": [float(v) for v in var2], "var3": [float(v) for v in var3],
            "papers": rows, "clusters": clusters, "universes": universes}
    out.write_text(json.dumps(data))
    status(phase="done", msg=f"{len(rows)} papers, {len(real)} solar systems, {len(universes)} universes -> {out}")
    result(n=len(rows), clusters=len(real), universes=len(universes), noise=int((labels == -1).sum()), eps=eps, model=model, out=str(out))


if __name__ == "__main__":
    main()
