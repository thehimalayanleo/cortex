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
        lab = DBSCAN(eps=float(eps), min_samples=2, metric="precomputed").fit_predict(D)
        k = len(set(lab)) - (1 if -1 in lab else 0)
        noise = float((lab == -1).mean())
        score = (0 if target_lo <= k <= target_hi else min(abs(k - target_lo), abs(k - target_hi))) + noise * 2
        if best is None or score < best[0]:
            best = (score, lab, float(eps), k, noise)
        metric(step=int(eps * 100), eps=float(eps), clusters=k, noise_frac=noise)
    _, lab, eps, k, noise = best
    status(phase="cluster", msg=f"dbscan eps={eps:.3f}: {k} clusters, {noise:.0%} noise")
    return lab, eps


def split_giants(X: np.ndarray, labels: np.ndarray, cap: int) -> np.ndarray:
    """A "system" of hundreds of papers is a galaxy, not a system. Any cluster above `cap` members is split with spherical
    k-means (unit vectors, cosine) into about size/cap/2 pieces so the map shows real neighbourhoods. Repeats until no
    cluster exceeds the cap or a split stops helping."""
    from sklearn.cluster import KMeans
    labels = labels.copy()
    for _ in range(4):
        big = [c for c in set(labels) if c != -1 and int((labels == c).sum()) > cap]
        if not big:
            break
        nxt = int(labels.max()) + 1
        for c in big:
            idx = np.where(labels == c)[0]
            k = int(max(2, min(12, round(len(idx) / (cap * 0.6)))))
            sub = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(X[idx])
            sizes = np.bincount(sub)
            if sizes.max() >= len(idx) - 1:  # k-means refused to split; leave it
                continue
            for j in range(k):
                if j == 0:
                    continue
                labels[idx[sub == j]] = nxt
                nxt += 1
            status(phase="split", msg=f"system of {len(idx)} split into {k} ({', '.join(str(int(v)) for v in sizes)})")
    return labels

def districts(X: np.ndarray, labels: np.ndarray, per: int) -> np.ndarray:
    """Neighbourhoods inside a system: spherical k-means into groups of about `per` papers. Returns a global district id
    per paper (-1 for drifting papers)."""
    from sklearn.cluster import KMeans
    hood = np.full(len(labels), -1, dtype=int)
    nxt = 0
    for c in sorted(set(labels)):
        if c == -1:
            continue
        idx = np.where(labels == c)[0]
        k = int(max(1, min(12, round(len(idx) / per))))
        if k <= 1 or len(idx) < 4:
            hood[idx] = nxt
            nxt += 1
            continue
        sub = KMeans(n_clusters=k, n_init=3, random_state=0).fit_predict(X[idx])
        for j in range(k):
            m = idx[sub == j]
            if len(m):
                hood[m] = nxt
                nxt += 1
    return hood


def pack(seed: np.ndarray, radii: np.ndarray, iters: int = 250) -> np.ndarray:
    """Place circles so they do not overlap, starting from semantic seed positions (so neighbours stay neighbours) and
    pulling everything toward the origin. Returns centres in the seed's units."""
    P = seed.astype(np.float64).copy()
    n = len(P)
    if n == 1:
        return np.zeros((1, 2))
    rng = np.random.default_rng(0)
    P += rng.normal(scale=1e-3, size=P.shape)
    for it in range(iters):
        d = P[:, None, :] - P[None, :, :]
        dist = np.linalg.norm(d, axis=-1) + 1e-9
        need = radii[:, None] + radii[None, :]
        np.fill_diagonal(need, 0.0)
        overlap = np.clip(need - dist, 0, None)
        push = (d / dist[..., None]) * (overlap * 0.5)[..., None]
        P += push.sum(1)
        P -= P * 0.02  # gravity toward the centre keeps the packing compact
        if it > iters // 2 and overlap.max() < 1e-4:
            break
    return P - P.mean(0)


def layout(X: np.ndarray, labels: np.ndarray, hood: np.ndarray, universe_of: dict[int, int]) -> dict:
    """Hierarchical map: universes are territories, systems are cities inside them, districts are blocks inside cities,
    papers are buildings inside blocks. Every level is a circle packing seeded by a PCA of that level's centroids, so
    semantically close things sit side by side. Returns positions in [-1, 1] and the circles of every level."""
    def seeds(vecs: np.ndarray) -> np.ndarray:
        if len(vecs) == 1:
            return np.zeros((1, 2))
        if len(vecs) == 2:
            return np.array([[-1.0, 0.0], [1.0, 0.0]])
        s, _ = pca(vecs, 2)
        return s / (np.abs(s).max() or 1.0)

    n = len(X)
    pos = np.zeros((n, 2))
    # districts -> papers
    hoods = sorted(h for h in set(hood) if h != -1)
    hood_center, hood_r, hood_vec = {}, {}, {}
    for h in hoods:
        idx = np.where(hood == h)[0]
        r = np.full(len(idx), 1.0)
        pts = pack(seeds(X[idx]) * max(1.0, math.sqrt(len(idx))), r)
        pos[idx] = pts
        hood_r[h] = float(np.max(np.linalg.norm(pts, axis=1)) + 1.6) if len(idx) > 1 else 1.6
        hood_vec[h] = X[idx].mean(0)
    # systems -> districts
    systems = sorted(c for c in set(labels) if c != -1)
    sys_center, sys_r, sys_vec = {}, {}, {}
    for c in systems:
        hs = sorted(set(hood[labels == c]) - {-1})
        V = np.stack([hood_vec[h] for h in hs])
        R = np.array([hood_r[h] for h in hs])
        pts = pack(seeds(V) * (R.sum() / max(1, len(hs))) * 1.2, R * 1.08)
        for h, pt in zip(hs, pts):
            hood_center[h] = pt
        sys_r[c] = float(max(np.linalg.norm(pts, axis=1) + R) + 1.5)
        sys_vec[c] = X[labels == c].mean(0)
    # universes -> systems
    unis = sorted(set(universe_of.values()))
    uni_center, uni_r = {}, {}
    for u in unis:
        cs = [c for c in systems if universe_of.get(c) == u]
        V = np.stack([sys_vec[c] for c in cs])
        R = np.array([sys_r[c] for c in cs])
        pts = pack(seeds(V) * (R.sum() / max(1, len(cs))) * 1.2, R * 1.1)
        for c, pt in zip(cs, pts):
            sys_center[c] = pt
        uni_r[u] = float(max(np.linalg.norm(pts, axis=1) + R) + 3.0)
    # the map -> universes
    if unis:
        V = np.stack([np.mean([sys_vec[c] for c in systems if universe_of.get(c) == u], axis=0) for u in unis])
        R = np.array([uni_r[u] for u in unis])
        pts = pack(seeds(V) * (R.sum() / max(1, len(unis))) * 1.2, R * 1.12)
        for u, pt in zip(unis, pts):
            uni_center[u] = pt
    # compose absolute positions
    hood_abs, sys_abs, uni_abs = {}, {}, {}
    for u in unis:
        uni_abs[u] = uni_center[u]
    for c in systems:
        sys_abs[c] = uni_abs.get(universe_of.get(c, -1), np.zeros(2)) + sys_center[c]
    for h in hoods:
        c = int(labels[np.where(hood == h)[0][0]])
        hood_abs[h] = sys_abs[c] + hood_center[h]
    for i in range(n):
        if hood[i] != -1:
            pos[i] = hood_abs[hood[i]] + pos[i]
    # drifting papers: on the rim of the map
    drift = np.where(hood == -1)[0]
    if len(drift):
        big = max([np.linalg.norm(uni_abs[u]) + uni_r[u] for u in unis] or [10.0]) + 4.0
        ang = np.linspace(0, 2 * math.pi, len(drift), endpoint=False)
        pos[drift] = np.stack([big * np.cos(ang), big * np.sin(ang)], 1)
    scale = float(np.abs(pos).max() + 2.0) or 1.0
    pos /= scale
    return {"pos": pos, "scale": scale,
            "hoods": {h: (hood_abs[h] / scale, hood_r[h] / scale) for h in hoods},
            "systems": {c: (sys_abs[c] / scale, sys_r[c] / scale) for c in systems},
            "universes": {u: (uni_abs[u] / scale, uni_r[u] / scale) for u in unis}}


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
    ap.add_argument("--clusters", default="10-40", help="wanted cluster count range")
    ap.add_argument("--orbit", type=float, default=0.55, help="min cosine to a system centroid for a noise paper to join it")
    ap.add_argument("--max-system", type=int, default=40, help="systems bigger than this are split with spherical k-means")
    ap.add_argument("--district", type=int, default=7, help="about this many papers per district inside a system")
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
    labels = split_giants(X, labels, a.max_system)
    # Planets in orbit: a paper DBSCAN left as noise still belongs to the system it is closest to, if that system's
    # centroid is reasonably near (cosine >= --orbit). Core members keep their label; orbiting ones are marked.
    core = labels.copy()
    cids = sorted(c for c in set(labels) if c != -1)
    if cids:
        cent = np.stack([X[labels == c].mean(0) for c in cids])
        cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        sims = X @ cent.T
        for i in np.where(labels == -1)[0]:
            j = int(np.argmax(sims[i]))
            if sims[i, j] >= a.orbit:
                labels[i] = cids[j]
        status(phase="orbit", msg=f"{int((core == -1).sum() - (labels == -1).sum())} papers attached to their nearest system (cos >= {a.orbit}); {int((labels == -1).sum())} still drifting")
    _, var2 = pca(X, 2)
    P3, var3 = pca(X, 3)
    P3 = P3 / (np.abs(P3).max() or 1.0)
    hood = districts(X, labels, a.district)
    status(phase="districts", msg=f"{len(set(hood) - {-1})} districts of about {a.district} papers")
    df = Counter()
    for p in papers:
        for w in set(re.findall(r"[a-z][a-z0-9\-]{2,}", p["title"].lower())):
            df[w] += 1
    # universes: agglomerative grouping of system centroids (cosine), about sqrt(k) groups
    systems_ids = sorted(c for c in set(labels) if c != -1)
    universe_of: dict[int, int] = {}
    universes = []
    if len(systems_ids) >= 3:
        # size-weighted spherical k-means over system centroids: universes of comparable weight, unlike
        # average-linkage which makes one huge universe plus singletons
        from sklearn.cluster import KMeans
        C = np.stack([X[labels == c].mean(0) for c in systems_ids])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        w = np.array([float((labels == c).sum()) for c in systems_ids])
        nU = max(2, min(8, int(round(math.sqrt(len(systems_ids))))))
        ul = KMeans(n_clusters=nU, n_init=8, random_state=0).fit(C, sample_weight=w).labels_
        for c, u in zip(systems_ids, ul):
            universe_of[c] = int(u)
    elif systems_ids:
        for c in systems_ids:
            universe_of[c] = 0
    status(phase="layout", msg="packing universes > systems > districts > papers")
    L = layout(X, labels, hood, universe_of)
    P2 = L["pos"]
    clusters = []
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        titles = [papers[i]["title"] for i in idx]
        c = {"id": int(cid), "label": "drifting" if cid == -1 else label_for(titles, df, len(papers)), "size": int(len(idx)), "universe": universe_of.get(int(cid), -1)}
        if cid in L["systems"]:
            (cx, cy), r = L["systems"][cid]
            c.update(cx=float(cx), cy=float(cy), r=float(r))
        else:
            c.update(cx=float(P2[idx, 0].mean()), cy=float(P2[idx, 1].mean()), r=0.0)
        clusters.append(c)
    real = [c for c in clusters if c["id"] != -1]
    for u in sorted(set(universe_of.values())):
        members = [c for c in systems_ids if universe_of[c] == u]
        titles = [papers[i]["title"] for c in members for i in np.where(labels == c)[0]]
        (cx, cy), r = L["universes"][u]
        universes.append({"id": int(u), "label": label_for(titles, df, len(papers), k=2), "clusters": [int(c) for c in members],
                          "size": int(sum(int((labels == c).sum()) for c in members)), "cx": float(cx), "cy": float(cy), "r": float(r)})
    hoods = []
    for h, ((cx, cy), r) in L["hoods"].items():
        idx = np.where(hood == h)[0]
        titles = [papers[i]["title"] for i in idx]
        hoods.append({"id": int(h), "label": label_for(titles, df, len(papers), k=2), "size": int(len(idx)), "cluster": int(labels[idx[0]]),
                      "universe": universe_of.get(int(labels[idx[0]]), -1), "cx": float(cx), "cy": float(cy), "r": float(r)})
    umap = {c["id"]: c.get("universe", -1) for c in clusters}
    rows = []
    for i, p in enumerate(papers):
        rows.append({"id": p["id"], "title": p["title"], "year": p["year"], "status": p["status"], "topics": p["topics"], "authors": str(p["authors"])[:120],
                     "x": float(P2[i, 0]), "y": float(P2[i, 1]), "x3": float(P3[i, 0]), "y3": float(P3[i, 1]), "z3": float(P3[i, 2]),
                     "cluster": int(labels[i]), "hood": int(hood[i]), "universe": umap.get(int(labels[i]), -1), "core": bool(core[i] != -1)})
    # nearest neighbours per paper (for "planets near this one")
    S = X @ X.T
    for i, r in enumerate(rows):
        nn = np.argsort(-S[i])[1:6]
        r["near"] = [[papers[j]["id"], float(S[i, j])] for j in nn]
    data = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model, "eps": eps, "n": len(rows), "var2": [float(v) for v in var2], "var3": [float(v) for v in var3],
            "papers": rows, "clusters": clusters, "universes": universes, "hoods": hoods, "levels": ["universe", "system", "district", "paper"]}
    out.write_text(json.dumps(data))
    status(phase="done", msg=f"{len(rows)} papers, {len(hoods)} districts, {len(real)} systems, {len(universes)} universes -> {out}")
    result(n=len(rows), districts=len(hoods), clusters=len(real), universes=len(universes), noise=int((labels == -1).sum()), eps=eps, model=model, out=str(out))


if __name__ == "__main__":
    main()
