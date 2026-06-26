"""Random walk with restart (RWR) — the network-medicine workhorse for disease
gene prioritisation.

Idea: drop a walker on the known disease genes. At every step it either hops to
a neighbour (weighted by interaction confidence) or teleports back to the seeds
with probability `restart`. Where the walker spends its time is the score: genes
the seeds reach often, through many short paths, score high. It is "guilt by
association" done properly, accounting for the whole graph rather than just
direct neighbours.

No training, no parameters fit from labels — the score is a fixed-point of the
graph and the seed vector. That makes it the honest baseline the learned model
has to beat.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from . import config as C


def _column_normalise(adj: sp.csr_matrix) -> sp.csr_matrix:
    """Column-stochastic transition matrix W: W[i, j] = P(i <- j)."""
    deg = np.asarray(adj.sum(axis=0)).ravel()
    deg[deg == 0] = 1.0
    dinv = sp.diags(1.0 / deg)
    return (adj @ dinv).tocsr()


def random_walk_with_restart(
    adj: sp.csr_matrix,
    seed_idx: np.ndarray,
    restart: float = C.RWR_RESTART,
    tol: float = 1e-8,
    max_iter: int = 200,
) -> np.ndarray:
    """Stationary visiting probability of a restart walk seeded at `seed_idx`.

    Returns a length-n score vector that sums to 1.
    """
    n = adj.shape[0]
    W = _column_normalise(adj)

    p0 = np.zeros(n)
    p0[seed_idx] = 1.0 / len(seed_idx)
    p = p0.copy()
    for _ in range(max_iter):
        p_next = (1 - restart) * (W @ p) + restart * p0
        if np.linalg.norm(p_next - p, 1) < tol:
            p = p_next
            break
        p = p_next
    return p


def score(net, seed_idx: np.ndarray) -> np.ndarray:
    """Convenience wrapper: RWR scores for every node given training seeds."""
    return random_walk_with_restart(net.adjacency, seed_idx)
