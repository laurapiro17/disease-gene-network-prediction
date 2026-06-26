"""The learned alternative: represent every gene as a vector, then train a
classifier to recognise "looks like an Alzheimer gene".

Where RWR propagates labels across the raw graph, this learns a geometry first.
We take the leading eigenvectors of the normalised graph Laplacian — a spectral
embedding — so that genes sitting in the same densely-wired neighbourhood land
near each other in R^d. A logistic regression then learns, from the *training*
seeds only, the region of that space where disease genes live, and scores the
rest of the genome by how far in they fall.

Spectral embedding is deterministic and pure scipy/sklearn: no node2vec random
walks, no GPU, no extra dependency. It is the light way to put a genuinely
*trained* model next to the propagation baseline.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from . import config as C


def spectral_embedding(adj: sp.csr_matrix, dim: int = C.EMBED_DIM,
                       random_state: int = C.RANDOM_STATE) -> np.ndarray:
    """Smallest non-trivial eigenvectors of the symmetric normalised Laplacian.

    Cached per graph in-process via the lru cache on `embed_network`; here we
    just do the linear algebra. Returns an (n, dim) array.
    """
    n = adj.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv_sqrt = sp.diags(1.0 / np.sqrt(deg))
    norm_adj = (dinv_sqrt @ adj @ dinv_sqrt).tocsr()  # = I - L_sym

    # Leading eigenvectors of the normalised adjacency are the smallest of the
    # normalised Laplacian — same embedding, far faster and more stable than
    # asking eigsh for the smallest-magnitude end of L.
    k = min(dim + 1, n - 1)
    rng = np.random.default_rng(random_state)
    v0 = rng.standard_normal(n)
    vals, vecs = eigsh(norm_adj, k=k, which="LA", v0=v0, tol=1e-4)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order[1:]]  # drop the leading (near-constant) eigenvector
    return vecs


def embed_predict(embedding: np.ndarray, seed_idx: np.ndarray,
                  train_mask_seed: np.ndarray) -> np.ndarray:
    """Train a random forest on the embedding and score every gene.

    Positives are the *training* seeds; negatives are a random sample of
    non-seed genes. A linear classifier fails here — disease genes occupy
    several disconnected regions of the spectral space, so we need a model that
    can carve out non-convex regions. The forest scores every node by its
    estimated probability of being a disease gene.
    """
    n = embedding.shape[0]
    X = StandardScaler().fit_transform(embedding)

    # Negatives: 20 per positive. Many more than that just dilutes the signal
    # the forest sees; far fewer overfits the handful of seeds.
    rng = np.random.default_rng(C.RANDOM_STATE)
    neg_pool = np.where(~train_mask_seed)[0]
    neg = rng.choice(neg_pool, size=min(len(neg_pool), 20 * len(seed_idx)),
                     replace=False)

    train_idx = np.concatenate([seed_idx, neg])
    y = np.zeros(len(train_idx), dtype=int)
    y[: len(seed_idx)] = 1

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=C.RANDOM_STATE, n_jobs=-1)
    clf.fit(X[train_idx], y)
    return clf.predict_proba(X)[:, 1]
