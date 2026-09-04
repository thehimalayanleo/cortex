---
title: "Lab 08: Clustering, retrieval, and using embeddings for data work"
kind: permanent
topics: [lab]
chapter: 8
station: cluster
recipe: recipes/index_vault.py
reading_time: 50 min
---

## What you will be able to do

1. Implement k-means with k-means++ seeding, prove that each Lloyd iteration cannot increase the objective, and choose $k$ with silhouette rather than by squinting at an elbow.
2. Decide from the shape of your data whether k-means or HDBSCAN is the right tool, and read HDBSCAN's condensed tree and noise labels correctly.
3. Reduce an embedding cloud to two dimensions for looking with PCA and UMAP, and explain why you should not cluster in UMAP space.
4. Choose between brute force, HNSW, and IVF from a cost formula, then build a hybrid BM25 plus vector index over your vault's PDFs with reciprocal rank fusion.
5. Use clusters to balance a corpus and remove near-duplicates, and connect that to the data work of Lab 01.

## The idea in one paragraph

Once every text is a point on a sphere (Lab 07), a large family of data problems becomes geometry. Grouping points that sit together is clustering, and the workhorse is k-means: guess some centers, assign each point to its nearest one, move each center to the middle of its points, repeat. Finding the points nearest to a new query is retrieval, and for a few million vectors a single matrix multiply on the 5090 is exact and fast enough; beyond that you trade a little recall for a graph or a partition that skips most of the comparisons. Neither embeddings nor word matching wins everywhere, so a practical search runs both and merges the two ranked lists by rank rather than by score. The same clusters that help you see the corpus let you sample it evenly and find near-duplicate documents cheaply, which is the data-curation use that pretraining pipelines actually care about.

## The math

### k-means and Lloyd's algorithm

You have $n$ points $x_1, \dots, x_n \in \mathbb{R}^d$ and want $k$ centers $c_1, \dots, c_k$ and an assignment $a_i \in \{1, \dots, k\}$ that minimize the within-cluster sum of squares, also called inertia:

$$
J(c, a) = \sum_{i=1}^{n} \|x_i - c_{a_i}\|^2 .
$$

Minimizing $J$ jointly is NP-hard even for $k = 2$ in general dimension, so Lloyd's algorithm alternates two partial minimizations. The assignment step fixes the centers and sets $a_i = \arg\min_j \|x_i - c_j\|^2$; for fixed centers this is the best possible assignment, pointwise, so $J$ cannot increase. The update step fixes the assignment and sets each center to the mean of its members, $c_j = \frac{1}{|S_j|}\sum_{i \in S_j} x_i$ with $S_j = \{i : a_i = j\}$. To see that this is the minimizer, differentiate the cluster's contribution $\sum_{i \in S_j}\|x_i - c\|^2$ with respect to $c$: the gradient is $-2\sum_{i \in S_j}(x_i - c)$, which vanishes exactly at the mean, and the function is convex in $c$, so the mean is the unique minimizer and again $J$ cannot increase. Since there are finitely many assignments and $J$ strictly decreases whenever an assignment changes, the algorithm terminates at a fixed point. Nothing in the argument says the fixed point is the global minimum, and in practice it often is not, which is what seeding and restarts are for.

For embeddings you want cosine similarity, not Euclidean distance. For unit vectors the two agree: $\|a - b\|^2 = \|a\|^2 + \|b\|^2 - 2a^\top b = 2 - 2a^\top b$, so the nearest center in Euclidean terms is the most similar in cosine terms provided the centers are also unit vectors. Spherical k-means renormalizes each center after the update step; that is the only change.

### k-means++

Random initial centers tend to land two in one dense region and none in a small far-away cluster, and Lloyd rarely recovers. k-means++ picks the first center uniformly and each next center with probability proportional to $D(x)^2$, the squared distance from $x$ to its nearest already-chosen center:

$$
P(\text{choose } x) = \frac{D(x)^2}{\sum_{x'} D(x')^2}.
$$

Points far from every existing center are likely to be chosen, so every well-separated group gets a seed, but sampling rather than taking the farthest point keeps a single outlier from hijacking a seed. The guarantee is that the expected inertia of the seeding alone satisfies $\mathbb{E}[J] \le 8(\ln k + 2)\, J_{\mathrm{OPT}}$, before any Lloyd iteration; Lloyd then only improves it. The cost is $k$ passes over the data, which is negligible against the iterations that follow.

### Inertia, the elbow, and silhouette

Inertia is monotone in $k$: a solution with $k + 1$ centers can always reproduce a solution with $k$ by placing the extra center on any point, so $J_{k+1} \le J_k$, and at $k = n$ it is zero. That is why plotting inertia against $k$ and looking for the bend (the elbow) is weak evidence: the curve always bends, and where it bends depends on how far you plot. The silhouette of point $i$ compares its mean distance to its own cluster, $a_i$, with its mean distance to the nearest other cluster, $b_i$:

$$
s_i = \frac{b_i - a_i}{\max(a_i, b_i)} \in [-1, 1],
$$

and the silhouette score is the mean over points. It is $+1$ for a point far from every other cluster and tight in its own, $0$ on a boundary, and negative for a point that would rather be elsewhere. Unlike inertia it is not monotone in $k$, so it can peak. It costs $O(n^2 d)$ because it needs all pairwise distances, so on a large corpus compute it on a random sample of a few thousand points. On real text embeddings the absolute values are far below those of the toy data in Build it small, because topics overlap and the dimensions are many; use it to compare values of $k$, not as a grade.

### When to use HDBSCAN instead

k-means assumes you know $k$, that clusters are roughly spherical and similar in size, and that every point belongs somewhere. When those fail (an unknown number of topics, dense cores with sparse fringes, and a lot of junk that belongs to nothing) a density-based method is the right tool. HDBSCAN defines the core distance of a point, $\mathrm{core}_m(x)$, as the distance to its $m$-th nearest neighbour, a local density estimate, and the mutual reachability distance

$$
d_{\mathrm{mr}}(a, b) = \max\big(\mathrm{core}_m(a),\ \mathrm{core}_m(b),\ d(a, b)\big),
$$

which pushes sparse points apart while leaving dense regions alone. It builds the minimum spanning tree of the complete graph under $d_{\mathrm{mr}}$, which is equivalent to single-linkage clustering as a threshold $\lambda = 1/d$ sweeps from 0 upward, then condenses the resulting hierarchy by treating any split that produces a child smaller than `min_cluster_size` as points falling out of the parent rather than as a new cluster. Each condensed cluster gets a stability, $\sum_{x \in C}(\lambda_x - \lambda_{\text{birth}}(C))$, the total persistence of its members, and the final flat clustering picks the set of non-overlapping clusters with maximal total stability. Points that never joined a stable cluster get the label $-1$, noise. The parameter that most people set wrong is `min_samples` ($m$): it is the density estimate, and raising it makes more of the corpus noise. Use HDBSCAN when you are exploring, when you want junk flagged, and when the number of clusters is a question rather than a choice; use k-means when you need a partition of everything into a fixed number of parts, which is what balancing and deduplication need.

### Dimensionality reduction for looking

PCA centers the data, forms the covariance $\Sigma = \frac{1}{n}X^\top X$, and projects onto the top $r$ eigenvectors $U_r$: $Y = X U_r$. The fraction of variance kept is $\sum_{i \le r}\lambda_i / \sum_i \lambda_i$. It is linear and it preserves the large-scale structure in the least-squares sense; there is also a precise link to k-means, since the continuous relaxation of the k-means assignment problem is solved by the top $k - 1$ principal components, so running k-means in a PCA space of a few dozen dimensions loses little. The cluster station does exactly this: the assignment runs in the full 48-dimensional space and PCA to two dimensions is used only to draw the points, centroids, and topic colours.

UMAP is not linear. It builds a $k$-nearest-neighbour graph and assigns each edge a fuzzy membership $v_{j|i} = \exp(-(d(x_i, x_j) - \rho_i)/\sigma_i)$, where $\rho_i$ is the distance to $i$'s nearest neighbour and $\sigma_i$ is chosen per point so that $\sum_j v_{j|i} = \log_2 k$. It symmetrizes to $v_{ij} = v_{j|i} + v_{i|j} - v_{j|i}v_{i|j}$, places the points in two dimensions with memberships $w_{ij} = (1 + a\|y_i - y_j\|^{2b})^{-1}$, and minimizes the binary cross-entropy between $v$ and $w$ by stochastic gradient descent with negative sampling for the repulsive term. Read the definition of $\sigma_i$ again: every point is normalized to have the same total membership to its neighbours, which means UMAP deliberately erases density differences, and the negative sampling means that distances between clusters in the picture are whatever the optimizer found convenient, not distances in the data. The islands you see are shaped by `n_neighbors` and `min_dist` as much as by the corpus. It is an excellent picture and a poor coordinate system: cluster in the original space or in a PCA space of tens of dimensions, then colour the UMAP plot by those labels to check them. If you find yourself running HDBSCAN on a two-dimensional UMAP output because it gives crisper clusters, understand that the crispness came from the hyperparameters, and verify every cluster you keep with nearest-neighbour purity in the original space.

### Approximate nearest neighbours, and when brute force is fine

Exact search over $n$ unit vectors of dimension $d$ is one matrix-vector product, $nd$ multiply-adds per query, or $Q n d$ for a batch of $Q$ queries as a single matmul. Worked example, with stated assumptions: $n = 10^6$, $d = 768$, float16 storage is $1.5$ GB, which fits in the 5090's 32 GB with room to spare; each query costs $7.7 \times 10^8$ multiply-adds, about $1.5$ GFLOP; a batch of 1,000 queries is $1.5$ TFLOP, and at an assumed sustained 50 TFLOP/s for a tall-skinny fp16 matmul that is about 30 milliseconds for the batch. A single query is bound by memory bandwidth instead, since the whole matrix must be read once: at an assumed 1.8 TB/s that is under a millisecond. Brute force on the GPU is therefore exact, simple, and fast up to several million vectors; you need approximate search when you are on CPU, when the index exceeds device memory, or when you need many single-query lookups with tight latency from a process that cannot hold the matrix.

HNSW builds a graph in which each vector links to about $M$ neighbours, arranged in layers like a skip list: the top layers hold a sparse sample for coarse navigation and the bottom layer holds everything. A query starts at the top, greedily walks to the nearest node, and descends, keeping a candidate list of size `efSearch` at the bottom. Recall rises with `efSearch` and with the build parameter `efConstruction`; memory is the vectors plus roughly $2M$ links of 4 bytes per node at the base layer; deletions are awkward. IVF partitions the vectors with k-means into $n_{\text{list}}$ cells and searches only the $n_{\text{probe}}$ cells whose centroids are nearest to the query, at a cost of $n_{\text{list}} d$ for the centroids plus $n_{\text{probe}}\, (n / n_{\text{list}})\, d$ for the cells. Setting the derivative with respect to $n_{\text{list}}$ to zero gives the cost-optimal $n_{\text{list}} = \sqrt{n_{\text{probe}}\, n}$, which is the origin of the "about $\sqrt{n}$ lists" rule. IVF loses recall on queries near a cell boundary, whose true neighbours sit in the unprobed next cell; raising $n_{\text{probe}}$ is the cure. Product quantization compresses each vector into $m$ one-byte codes by k-means in $m$ subspaces, cutting memory by a factor near $4d/m$ at a recall cost that you measure. Whatever index you build, measure its recall@10 against brute force on a sample of real queries; it is the only honest number an approximate index has.

### Hybrid retrieval: BM25 plus vectors, fused by rank

Embeddings blur exact strings: a paper identifier, a rare method name, or an author's surname can be retrieved by a lexical index and missed by a vector one, while a paraphrased question is the other way around. BM25 scores a document $D$ against query terms $t \in q$ by

$$
\mathrm{BM25}(q, D) = \sum_{t \in q} \mathrm{IDF}(t)\, \frac{f(t, D)\,(k_1 + 1)}{f(t, D) + k_1\big(1 - b + b\, |D| / \mathrm{avgdl}\big)},
\qquad
\mathrm{IDF}(t) = \ln \frac{N - n_t + 0.5}{n_t + 0.5} + 1,
$$

where $f(t, D)$ is the term frequency, $|D|$ the document length, $\mathrm{avgdl}$ the mean length, $N$ the number of documents, $n_t$ the number containing $t$, and the constants $k_1 \approx 1.2$ to $2.0$ and $b = 0.75$ control term-frequency saturation and length normalization (the $+1$ in the IDF is the variant that keeps common terms from going negative). BM25 scores and cosine scores live on different scales, so combining them by addition needs calibration you do not have. Reciprocal rank fusion avoids scores entirely:

$$
\mathrm{RRF}(D) = \sum_{r \in \text{rankers}} \frac{1}{\kappa + \mathrm{rank}_r(D)},
$$

with $\kappa = 60$ in the original paper and documents absent from a ranker's list contributing nothing. The constant keeps a single rank-1 result from dominating: a document ranked 1 by one ranker and 5 by the other scores $1/61 + 1/65 = 0.0318$, while a document ranked 2 by both scores $2/62 = 0.0323$ and wins. Consistent agreement beats one strong vote, which is what you want from two rankers that fail differently.

### Chunking documents for retrieval

An embedding model with mean pooling produces one vector per input, so a 15,000-token paper embedded whole gives you a vector that is the average of everything in it, and no query about one section will match it well. You split the text into chunks of $\ell$ tokens with an overlap of $o$ so that a sentence cut by a boundary appears intact in one of the two neighbours; the number of chunks is about $\lceil (T - o)/(\ell - o) \rceil$, and the overlap costs a fraction $o/(\ell - o)$ of extra embedding work. Shorter chunks are more precise and lose context; longer chunks carry context and dilute the average. Two practices make more difference than the exact $\ell$: split on structure (headings, paragraphs) rather than at fixed token counts when the document has structure, and prepend a short header to each chunk (paper title, section name) so that the chunk's vector knows where it came from. Retrieval returns chunks, and you aggregate to documents by taking the maximum chunk score per document (or a small sum of the top few) so that one paper does not fill the whole result list.

### Clusters for corpus balancing and deduplication

Lab 01 removed exact and near-exact duplicates with hashing and set mixture weights per source. Embedding clusters extend both. For deduplication, semantic near-duplicates (the same paper from two sources with different extraction noise, a preprint and its published version, a tutorial copied with light edits) have cosine similarity well above what unrelated texts reach but will not hash to the same bucket. Comparing all pairs costs $n^2$; clustering first and comparing only within clusters costs $\sum_c n_c^2$, which for $k$ balanced clusters is $n^2/k$, and you drop one member of each pair above a threshold you choose by inspecting pairs near it. For balancing, cluster counts $n_c$ tell you where the corpus is concentrated; sampling each document with weight proportional to $n_{c(i)}^{-\alpha}$ for $\alpha \in [0, 1]$ interpolates between the natural distribution ($\alpha = 0$) and a flat one over clusters ($\alpha = 1$), and a per-cluster cap does the same more bluntly. The mixture-weight decision of the pretrain and midtrain stations (Labs 02 and 03) can then be made per cluster rather than per source, which is how a crawl gets its junk regions dropped and its thin regions upweighted before tokenization.

## Build it small

The snippet implements k-means++ seeding, Lloyd's iterations with the inertia printed each step, and silhouette, on Gaussian blobs in 16 dimensions with a known number of clusters.

```python
# k-means++ seeding, Lloyd's algorithm, inertia, and silhouette on Gaussian blobs (CPU, a few seconds)
import torch
torch.manual_seed(0)
K_TRUE, D, N = 5, 16, 1500
centers = torch.randn(K_TRUE, D) * 3.0
y = torch.randint(K_TRUE, (N,))
X = centers[y] + torch.randn(N, D)               # blobs with unit variance around each center

def kpp_init(X, k):
    """k-means++: first center uniform, each next one sampled with probability prop. to D(x)^2."""
    C = [X[torch.randint(len(X), (1,))].squeeze(0)]
    for _ in range(1, k):
        d2 = torch.cdist(X, torch.stack(C)).min(dim=1).values ** 2
        C.append(X[torch.multinomial(d2 / d2.sum(), 1)].squeeze(0))
    return torch.stack(C)

def lloyd(X, C, iters=50, verbose=False):
    for t in range(iters):
        a = torch.cdist(X, C).argmin(dim=1)                     # assignment step
        inertia = ((X - C[a]) ** 2).sum().item()
        if verbose: print(f"  iter {t:2d}  inertia {inertia:9.1f}")
        newC = torch.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(len(C))])
        if torch.allclose(newC, C): break                       # update step reached a fixed point
        C = newC
    return C, a, inertia

def silhouette(X, a):
    Dm = torch.cdist(X, X); k = int(a.max()) + 1; s = torch.zeros(len(X))
    for i in range(len(X)):
        own = a == a[i]; own[i] = False
        ai = Dm[i, own].mean() if own.any() else torch.tensor(0.0)
        bi = min(Dm[i, a == j].mean() for j in set(range(k)) - {int(a[i])} if (a == j).any())
        s[i] = (bi - ai) / torch.maximum(ai, bi)
    return s.mean().item()

def purity(a, y):
    return sum(torch.bincount(y[a == j]).max().item() for j in a.unique()) / len(y)

print("Lloyd's iterations for k=5 (inertia never increases):")
C, a, _ = lloyd(X, kpp_init(X, 5), verbose=True)
print(f"purity vs true labels: {purity(a, y):.3f}")
print("\nchoosing k: inertia always falls; silhouette peaks at the true k")
for k in [2, 3, 4, 5, 6, 8, 10]:
    best = min((lloyd(X, kpp_init(X, k)) for _ in range(3)), key=lambda r: r[2])   # 3 restarts
    print(f"  k={k:2d}  inertia {best[2]:9.1f}  silhouette {silhouette(X, best[1]):.3f}")
```

Expected output (seed 0, CPU; exact values vary by PyTorch version, the pattern does not):

```
Lloyd's iterations for k=5 (inertia never increases):
  iter  0  inertia   50861.6
  iter  1  inertia   23983.9
purity vs true labels: 1.000

choosing k: inertia always falls; silhouette peaks at the true k
  k= 2  inertia  148036.3  silhouette 0.392
  k= 3  inertia   89836.3  silhouette 0.493
  k= 4  inertia   51596.1  silhouette 0.561
  k= 5  inertia   23983.9  silhouette 0.655
  k= 6  inertia   23691.0  silhouette 0.545
  k= 8  inertia   23152.3  silhouette 0.293
  k=10  inertia   22712.0  silhouette 0.159
```

Three things to read off. The blobs are well separated (centers spread with standard deviation 3 against unit noise), so k-means++ seeds one center per blob and Lloyd converges in two iterations to purity 1.0; with random seeding you will see runs that need many iterations and end with two centers sharing a blob. Inertia at $k = 5$ is close to the expected within-blob variance, $N \cdot D = 24{,}000$ for unit-variance noise in 16 dimensions, and beyond the true $k$ it keeps falling but only slightly, which is the elbow; notice how much clearer the silhouette peak is than the bend. Replace the blobs with the Lab 07 embeddings of your papers and the silhouette values will be a fraction of these, since real topics overlap; the peak, if any, is still informative. In the cluster station you will watch the same two steps on the encoder's embeddings, with the purity against the hidden topics reported after each iteration.

## Build it real

The recipe is `recipes/index_vault.py`. It turns the PDFs and markdown notes in your vault into a searchable hybrid index and, optionally, a clustering with named clusters and a duplicate report. The heart of it is short enough to show; this is the worked example that indexes a folder of PDFs' extracted text, with PyMuPDF for extraction, sentence-transformers for vectors, FAISS for the vector index, and `rank_bm25` for the lexical side.

```python
import glob, re, numpy as np, fitz, faiss                      # fitz is PyMuPDF
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def extract(path):                                             # page texts, one string per page
    return [page.get_text() for page in fitz.open(path)]

def chunk(words, size=200, overlap=30):
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(1, len(words) - overlap), step)]

chunks, meta = [], []                                          # meta: (file, page) per chunk
for path in sorted(glob.glob("/Users/ajinkya/Cortex/**/*.pdf", recursive=True)):
    title = fitz.open(path).metadata.get("title") or path.split("/")[-1]
    for p, text in enumerate(extract(path)):
        for c in chunk(text.split()):
            if sum(ch.isalpha() for ch in c) / max(1, len(c)) > 0.6:   # drop tables, references noise
                chunks.append(f"{title} | page {p+1}\n{c}"); meta.append((path, p + 1))

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
E = model.encode(["search_document: " + c for c in chunks], batch_size=64,
                 normalize_embeddings=True, show_progress_bar=True).astype("float32")
index = faiss.IndexFlatIP(E.shape[1]); index.add(E)           # exact inner product = cosine
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
bm25 = BM25Okapi([tok(c) for c in chunks])

def search(query, k=10, kappa=60, cand=100):
    q = model.encode(["search_query: " + query], normalize_embeddings=True).astype("float32")
    _, vec_ids = index.search(q, cand)                         # ranked chunk ids from vectors
    bm_ids = np.argsort(-bm25.get_scores(tok(query)))[:cand]   # ranked chunk ids from BM25
    fused = {}
    for ranked in (vec_ids[0], bm_ids):
        for rank, i in enumerate(ranked, start=1):
            fused[int(i)] = fused.get(int(i), 0.0) + 1.0 / (kappa + rank)
    best_per_doc = {}                                          # aggregate chunks to documents
    for i, s in sorted(fused.items(), key=lambda kv: -kv[1]):
        best_per_doc.setdefault(meta[i][0], (s, meta[i][1], chunks[i][:120]))
    return list(best_per_doc.items())[:k]

for path, (score, page, snippet) in search("orthogonal matching pursuit recovery guarantees"):
    print(f"{score:.4f}  {path.split('/')[-1]}  p.{page}  {repr(snippet)}")
```

Data. Everything under `~/Cortex` that ends in `.pdf` or `.md`; markdown files are chunked directly. Chunks are 200 words with 30 words of overlap, each prefixed with the title and page so that the vector knows its source, and chunks whose alphabetic fraction is below 0.6 (tables, reference lists, extraction garbage) are dropped before embedding. The script records the source file, page, and character offsets for every chunk.

Model. The Lab 07 fine-tuned model if you pass `--model runs/embed/<timestamp>/final`, otherwise `nomic-ai/nomic-embed-text-v1.5` with the `search_document: ` and `search_query: ` prefixes; `--dim 256` stores Matryoshka-truncated, renormalized vectors instead of the full 768.

Index. Exact inner product by default (`IndexFlatIP`), which is the right choice at vault scale; `--hnsw M=32,ef=128` or `--ivf nlist=1024,nprobe=16` builds an approximate index and, either way, the script reports recall@10 of the approximate index against the exact one on 200 sampled queries. The vector matrix is saved in float16 with a JSONL of chunk metadata so that any other process (the Cortex app's chat tools, for one) can load it without FAISS.

Clustering and data work. `--cluster 40` runs spherical k-means with k-means++ and ten restarts on the chunk vectors, aggregates to documents by majority chunk label, and prints for each cluster its size, its silhouette on a 3,000-point sample, and the five highest-TF-IDF terms of its members so you can name it. `--dedup 0.92` reports pairs of chunks within a cluster whose cosine exceeds the threshold, grouped by document, so you can see the preprint-versus-published pairs and the copied notes. `--umap` writes two-dimensional coordinates coloured by cluster to an HTML scatter for looking, and nothing downstream reads those coordinates. `--hdbscan min_cluster_size=15,min_samples=5` runs the density-based alternative on a 50-dimensional PCA of the vectors and reports the noise fraction alongside the clusters.

Arguments: `--vault ~/Cortex`, `--out index/vault`, `--model`, `--dim`, `--chunk 200 --overlap 30`, `--hnsw`, `--ivf`, `--cluster K`, `--dedup THRESH`, `--hdbscan`, `--umap`, `--query "..."` (search the existing index and print fused, document-aggregated results with page numbers), `--eval queries.jsonl` (recall@10, MRR@10 for vector-only, BM25-only, and fused, over a file of query and expected-file pairs), and `--rebuild`.

What to watch in the logs. Extraction prints pages per second and the fraction of chunks dropped by the alphabetic filter; above 30 percent dropped means a folder of scanned PDFs that need OCR, not a bad threshold. Embedding prints chunks per second. Indexing prints the approximate-versus-exact recall if you asked for an approximate index. Clustering prints inertia per restart, and you want the best of ten to be within a percent or two of the others; a wide spread means the cluster structure is weak or $k$ is too large.

How long it takes. Assume 2,000 PDFs of 15 pages at 500 tokens per page, which is 15 million tokens, roughly 11 million words, and about 65,000 chunks of 200 words with the 30-word overlap (one chunk per 170-word step). An encoder forward pass costs about $2P$ FLOPs per token, so embedding is $2 \times 1.37 \times 10^{8} \times 1.5 \times 10^{7} \approx 4 \times 10^{15}$ FLOPs, well under a minute at an assumed 100 TFLOP/s and a few minutes once batching and padding overhead are counted. PDF text extraction runs on the CPU and will dominate on a first build, at a few pages per second per core. The vector matrix is $65{,}000 \times 768 \times 2$ bytes, about 100 MB in float16, so exact search is a single small matmul, and k-means with ten restarts on 65,000 points takes seconds on the GPU.

## How it goes wrong

One giant cluster and a tail of tiny ones. The vectors were not normalized, or a few directions of large variance dominate the Euclidean distance (embedding spaces from contrastive training are often anisotropic, with a strong common component). Normalize to unit norm and use spherical k-means; if the imbalance persists, subtract the mean vector before clustering and check whether the silhouette improves.

Different seeds give visibly different clusters. Lloyd converged to different local optima. Use k-means++ with restarts, keep the run with the lowest inertia, and report the adjusted Rand index between runs; if it is low, the data does not support $k$ clusters at that $k$ and you should choose $k$ by silhouette or use a smaller one.

No elbow anywhere in the inertia curve. There rarely is one on text embeddings, because clusters overlap and inertia declines smoothly. Stop looking for it; choose $k$ by silhouette on a sample, or, for balancing and dedup, by the purpose (a few hundred clusters for a multi-million-document corpus, a few dozen for a vault) since those uses need a partition rather than the "true" $k$.

HDBSCAN calls most of the corpus noise. `min_samples` is too high for the density of your data, or you ran it on 768 raw dimensions, where nearest-neighbour distances concentrate and every point looks like a sparse fringe. Reduce with PCA to around 50 dimensions first, lower `min_samples`, and check the condensed tree for whether the clusters you expected were condensed away.

The UMAP plot shows crisp islands, but nearest-neighbour purity in the original space says they are not separated. The islands are an artifact of `n_neighbors` and `min_dist`. Do not cluster the picture; label points from a clustering in the original or PCA space and use the picture only to check that the labels form contiguous regions.

HNSW recall against brute force is 0.6 when you expected 0.95. Either `efSearch` is too small for the requested $k$, or you built an L2 index over unnormalized vectors when you wanted cosine. Normalize before indexing, use the inner-product metric, and raise `efSearch` until measured recall crosses your target; that measurement, not the defaults, is the setting.

Hybrid results are worse than vector-only. The BM25 side was built on text that was never lowercased or tokenized consistently with the queries, or the chunks are so different in length that BM25's length normalization is doing the ranking. Evaluate each ranker alone first with `--eval`; fusion cannot rescue a broken component, and RRF only helps when both lists are individually sensible.

The index is full of garbage snippets: ligature fragments, reference lists, two-column text interleaved line by line. Extraction problems, not retrieval ones. Filter chunks by alphabetic ratio, drop everything after a "References" heading, and extract with page blocks in reading order rather than raw text where the layout is two-column.

## Measure it

Clustering has no ground truth, so measure it three ways. Against labels you do have (folder names, arXiv categories in the PDF metadata, the topic tags in your notes): purity and normalized mutual information between clusters and labels, computed on documents rather than chunks. Against itself: the adjusted Rand index between the best two of ten restarts, which should be high if the structure is real. Against the eye: the five top-TF-IDF terms per cluster should read as a topic to you, and the UMAP plot coloured by cluster should show regions rather than confetti. Good means a silhouette peak at the chosen $k$, restart agreement well above what random labels would give, and cluster names you could use in a table of contents.

Retrieval you measure with the queries from Lab 07 (title and body-sentence queries with a known target paper) via `--eval`: recall@10 and MRR@10 for vector-only, BM25-only, and fused. Fused should be at least as good as the better component on your query set, and usually better on the mixed set because the two components fail on different queries; if it is worse, one component is broken. For an approximate index, recall@10 against brute force on 200 real queries should be at or above 0.95 before you accept the speed; report the `efSearch` or `nprobe` that achieves it and the query latency at that setting. For deduplication, report the fraction of documents removed at the threshold and hand-check twenty pairs just above and twenty just below it; the threshold is right when the pairs above are ones you would remove and the pairs below are ones you would keep.

## Exercises

1. Show that for a fixed set of points the mean minimizes the sum of squared distances, and that the same is not true for the sum of unsquared distances. Check: the gradient of the squared sum is $-2\sum(x_i - c)$; for unsquared distances the minimizer is the geometric median, which has no closed form.
2. For unit vectors $x$ and an arbitrary center $c$, expand $\|x - c\|^2$ and explain why Euclidean k-means on normalized data with unnormalized centers is not exactly spherical k-means. Check: $\|x - c\|^2 = 1 + \|c\|^2 - 2x^\top c$, and the $\|c\|^2$ term differs across clusters, favouring centers with small norm.
3. Derive the cost-optimal number of IVF lists as a function of $n$ and $n_{\text{probe}}$, then compute it for $n = 10^7$ and $n_{\text{probe}} = 32$. Answer: $n_{\text{list}} = \sqrt{n_{\text{probe}}\, n} \approx 17{,}900$.
4. Implement mini-batch k-means in the snippet (update each center towards the mean of its batch members with a per-center step size $1/\text{count}_j$) and compare inertia with full Lloyd after twenty passes. Check: within a few percent on the blobs.
5. Compute the RRF scores with $\kappa = 60$ for a document ranked 1 and 20 by the two rankers and a document ranked 4 and 4. Answer: $1/61 + 1/80 = 0.0289$ versus $2/64 = 0.0313$; the consistent document wins.
6. Build the vault index with `--hnsw` at `efSearch` in $\{16, 32, 64, 128, 256\}$ and plot recall@10 against brute force. Check: recall is monotone in `efSearch`; report the smallest value that reaches 0.95 and its latency.

## Test yourself

1. Give the two-step argument for why Lloyd's algorithm never increases inertia, and then explain why it can still stop at a solution twice as bad as the optimum.

<details><summary>Answer</summary>
The assignment step minimizes $J$ over assignments for fixed centers, pointwise, and the update step minimizes $J$ over centers for a fixed assignment, since the mean is the unique minimizer of a sum of squared distances. Each step is a partial minimization, so $J$ is non-increasing and, with finitely many assignments, terminates. But the alternation only guarantees a fixed point where neither step alone can improve; two centers sharing one true cluster while another true cluster is split between distant centers is such a fixed point, and moving one center across would require passing through worse configurations. That is why seeding and restarts matter and why the k-means++ bound is stated for the seeding, not for Lloyd.
</details>

2. Spot the bug: `C = X[torch.randperm(len(X))[:k]]` followed by Lloyd. It runs and gives a low inertia on most seeds. What does k-means++ change, and what is the guarantee?

<details><summary>Answer</summary>
Uniform random seeds are proportional to density, so a dense region gets several seeds and a small distant cluster often gets none, and Lloyd cannot move a center across empty space to fix it. k-means++ samples each next seed with probability proportional to the squared distance to the nearest existing seed, so far-away groups are almost always seeded. The guarantee is $\mathbb{E}[J] \le 8(\ln k + 2)\,J_{\mathrm{OPT}}$ for the seeding alone, in expectation over the sampling; no such bound exists for uniform seeds.
</details>

3. A colleague reports a silhouette of 0.12 on clustered paper embeddings and concludes the clustering failed. What is wrong with the conclusion, and what should they report instead?

<details><summary>Answer</summary>
Silhouette measures how much closer a point is to its own cluster than to the nearest other one. In hundreds of dimensions with overlapping topics, within- and between-cluster distances differ by a small relative margin for almost every point, so absolute silhouettes are low for any clustering, including a good one. The value is informative relative to other $k$ on the same data, not on an absolute scale. Report the silhouette curve over $k$, the restart agreement (adjusted Rand index), and purity against whatever labels exist.
</details>

4. Estimate, with stated assumptions, when exact search on the 5090 stops being adequate: memory and per-query cost for 5 million 768-dimensional vectors, batched and single-query.

<details><summary>Answer</summary>
Memory: $5 \times 10^6 \times 768 \times 2$ bytes is 7.7 GB in float16, which fits in 32 GB. Per query, $5 \times 10^6 \times 768 = 3.8 \times 10^9$ multiply-adds, about 7.7 GFLOP; a batch of 1,000 queries is 7.7 TFLOP, about 150 ms at an assumed 50 TFLOP/s. A single query must stream the whole matrix through the memory system once, about 7.7 GB at an assumed 1.8 TB/s, so roughly 4 ms. Exact search is adequate here. It stops being adequate when the matrix no longer fits (tens of millions of vectors), when the queries arrive one at a time from a CPU process without the GPU, or when many single queries per second exceed what streaming the matrix each time allows.
</details>

5. What breaks if you cluster in the two-dimensional UMAP space and then use the cluster sizes to set corpus-balancing weights?

<details><summary>Answer</summary>
Two things. The per-point normalization in UMAP's membership function equalizes local density, so regions that are dense in the data are spread out in the plot and cluster sizes in the plot no longer reflect how much of the corpus lives there. And the boundaries between islands are set by `n_neighbors` and `min_dist`, so the partition itself is a function of hyperparameters. The weights you derive would upweight and downweight the wrong regions. Cluster in the original or PCA space, count there, and use the plot only to look.
</details>

6. Spot the bug in this fusion: `fused[d] += 1 / (60 + score_r(d))` where `score_r` is the ranker's score.

<details><summary>Answer</summary>
RRF is a function of rank, not score. Using scores puts BM25 values (unbounded, often in the tens) and cosine values (in $[-1, 1]$) into the same formula, so the vector ranker's contribution is nearly constant across documents and BM25 decides everything. Sort each list, use the position as the rank, and let documents missing from a list contribute nothing for that list.
</details>

7. With chunks of 256 tokens and 32 tokens of overlap, how many chunks does a 15,000-token paper produce, and what fraction of embedding work does the overlap cost?

<details><summary>Answer</summary>
Step size is $256 - 32 = 224$, so about $\lceil (15{,}000 - 32)/224 \rceil = 67$ chunks against $\lceil 15{,}000/256 \rceil = 59$ without overlap. The overlap costs $32/224 \approx 14$ percent extra tokens through the encoder.
</details>

8. Why can within-cluster deduplication miss near-duplicate pairs, and what are two cheap fixes?

<details><summary>Answer</summary>
Two near-identical documents can fall on opposite sides of a cluster boundary, especially when they sit near a boundary to begin with, and the within-cluster pass never compares them. Assign each document to its two nearest centers and compare within both, or run a second pass with a different seed (a different partition) and take the union of detected pairs. Both keep the cost near $n^2/k$ rather than $n^2$.
</details>

9. In HDBSCAN, which of `min_cluster_size` and `min_samples` controls how much of the data becomes noise, and why do people confuse them?

<details><summary>Answer</summary>
`min_samples` sets the core distance (the distance to the $m$-th neighbour), which is the density estimate; raising it inflates mutual reachability distances for everything not deeply inside a dense region and turns fringes into noise. `min_cluster_size` only controls which splits in the hierarchy count as new clusters versus points falling out of a parent. People confuse them because the library defaults `min_samples` to `min_cluster_size` when it is not set, so changing the one they know about silently changes the other.
</details>

10. Does an IVF index with $n_{\text{probe}} = 1$ ever return a wrong nearest neighbour for a query that lies exactly at a centroid? What about a query on a Voronoi boundary between two cells?

<details><summary>Answer</summary>
At a centroid the query's cell is the one whose points are, on average, nearest, but the exact nearest neighbour can still be a point just inside a neighbouring cell that happens to be closer than any point in the query's own cell; nothing forbids it, though it is unlikely. On a boundary the failure is common: the true neighbour is equally likely to be in either cell, and with one probe you search only one. This is the mechanism of IVF's recall loss, and raising $n_{\text{probe}}$ trades cost for coverage of the neighbouring cells.
</details>

## What will change, what will not

The alternating-minimization argument for Lloyd, the $D^2$ seeding bound, the monotonicity of inertia in $k$, and the silhouette definition are mathematics; they will read the same in five years. So will the reason to measure any approximate index against brute force, and the reason to fuse rankers by rank rather than by score.

The BM25 formula is old and stable, and the observation that lexical and dense retrieval fail on different queries has survived several generations of embedding models. What is likely to change is the second stage: rerankers that read the query and candidate together, and late-interaction models that keep one vector per token, were already displacing simple fusion for the final ordering at the time of writing. The two-stage shape, cheap recall then expensive precision, is the durable part.

Brute force on a single GPU covers more each year, because memory and bandwidth grow while your vault does not. The formulas for cost and memory are what you keep; the point at which HNSW or IVF becomes necessary moves.

HDBSCAN and UMAP are specific algorithms with specific hyperparameters. The ideas underneath them, density-based clustering with a noise label and neighbourhood-preserving embeddings that discard global distances, will outlive their names, and so will the warning that a picture produced by an optimizer is not a coordinate system.

FAISS class names, PyMuPDF's API, `rank_bm25`, the chunk sizes, and the `efSearch` values in this chapter are tooling. When they change, the checks in Measure it (recall against exact search, per-component evaluation before fusion, hand-checked pairs around a dedup threshold) are how you re-tune the replacements.

## Read next

1. k-means++: The Advantages of Careful Seeding, Arthur, 2007. The $D^2$ sampling rule and its $O(\log k)$ approximation guarantee.
2. Density-Based Clustering Based on Hierarchical Density Estimates, Campello, 2013. HDBSCAN's mutual reachability, condensed tree, and stability selection.
3. UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, McInnes, 2018. The fuzzy graph construction and the cross-entropy objective, and the source of the caveats about interpreting its output.
4. Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs, Malkov, 2018. HNSW and the meaning of $M$, `efConstruction`, and `efSearch`.
5. The Probabilistic Relevance Framework: BM25 and Beyond, Robertson, 2009. Where the BM25 formula comes from and what its constants mean.
6. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods, Cormack, 2009. The fusion rule and the constant 60.
7. SemDeDup: Data-efficient Learning at Web-scale through Semantic Deduplication, Abbas, 2023. Cluster-then-compare deduplication on embeddings at pretraining scale, which is the Lab 01 connection.
8. K-means Clustering via Principal Component Analysis, Ding, 2004. The link between the top principal components and the relaxed k-means solution that justifies clustering in a PCA space.
