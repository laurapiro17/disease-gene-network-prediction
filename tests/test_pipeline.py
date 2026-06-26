"""Fast, download-free tests on a synthetic graph.

These check the algorithms — propagation, embedding, conformal coverage — on a
small planted-community network, so CI never downloads STRING. The reported
Alzheimer numbers come from running ``src.experiment`` on the real interactome,
not from here.
"""
import numpy as np
import scipy.sparse as sp

from src import conformal
from src.data import Network
from src.embedding import embed_predict, spectral_embedding
from src.experiment import _recovery_at_k
from src.propagation import random_walk_with_restart


def _planted_network(n_comm=60, n_other=140, p_in=0.25, p_out=0.01, seed=0):
    """Two blocks: a dense 'disease' community plus a looser background. Seeds
    are a subset of the community, so a good method should score the rest of the
    community above the background."""
    rng = np.random.default_rng(seed)
    n = n_comm + n_other
    comm = np.arange(n_comm)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            same = (i < n_comm) and (j < n_comm)
            if rng.random() < (p_in if same else p_out):
                A[i, j] = A[j, i] = 1.0
    adj = sp.csr_matrix(A)
    nodes = [f"G{i}" for i in range(n)]
    index = {g: i for i, g in enumerate(nodes)}
    seed_mask = np.zeros(n, dtype=bool)
    seed_mask[comm[:20]] = True          # first 20 of the community are known
    return Network(nodes, index, adj, seed_mask), comm


# --- propagation ---------------------------------------------------------
def test_rwr_is_a_distribution():
    net, _ = _planted_network()
    seed_idx = np.where(net.seed_mask)[0]
    p = random_walk_with_restart(net.adjacency, seed_idx)
    assert np.isclose(p.sum(), 1.0)
    assert (p >= 0).all()


def test_rwr_scores_community_above_background():
    net, comm = _planted_network()
    seed_idx = np.where(net.seed_mask)[0]
    p = random_walk_with_restart(net.adjacency, seed_idx)
    hidden = comm[20:]                              # community, not seeded
    background = np.arange(net.n)[~np.isin(np.arange(net.n), comm)]
    assert p[hidden].mean() > p[background].mean()


# --- embedding -----------------------------------------------------------
def test_spectral_embedding_shape():
    net, _ = _planted_network()
    emb = spectral_embedding(net.adjacency, dim=16)
    assert emb.shape == (net.n, 16)
    assert np.isfinite(emb).all()


def test_embed_predict_ranks_hidden_community_high():
    net, comm = _planted_network()
    emb = spectral_embedding(net.adjacency, dim=16)
    seed_idx = np.where(net.seed_mask)[0]
    scores = embed_predict(emb, seed_idx, net.seed_mask)
    hidden = comm[20:]
    background = np.arange(net.n)[~np.isin(np.arange(net.n), comm)]
    assert scores[hidden].mean() > scores[background].mean()


# --- conformal -----------------------------------------------------------
def test_conformal_threshold_monotone_in_alpha():
    rng = np.random.default_rng(0)
    cal = rng.random(50)
    # smaller alpha (more coverage) must not give a higher threshold
    assert conformal.conformal_threshold(cal, 0.05) <= conformal.conformal_threshold(cal, 0.30)


def test_conformal_coverage_is_valid():
    """Threshold from a calibration sample should cover >= 1 - alpha of fresh
    positives, averaged over many draws (the marginal guarantee)."""
    rng = np.random.default_rng(1)
    alpha = 0.1
    covered = []
    for _ in range(300):
        cal = rng.normal(1.0, 1.0, size=40)
        tst = rng.normal(1.0, 1.0, size=40)
        tau = conformal.conformal_threshold(cal, alpha)
        covered.append(conformal.empirical_coverage(tst, tau))
    assert np.mean(covered) >= 1 - alpha - 0.03


def test_candidate_set_excludes_training_seeds():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    exclude = np.array([True, False, False, False])
    cand = conformal.candidate_set(scores, threshold=0.65, exclude=exclude)
    assert 0 not in cand and 1 in cand


# --- metric --------------------------------------------------------------
def test_recovery_at_k_counts_hidden_in_top():
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.7])
    hidden = np.array([2, 4])          # scores 0.8, 0.7
    train_seeds = np.array([1])        # top scorer, excluded
    # ranking minus seed 1: [2 (0.8), 4 (0.7), 3? no 0.2...] -> top2 = {2,4}
    assert _recovery_at_k(scores, hidden, train_seeds, k=2) == 1.0
    assert _recovery_at_k(scores, hidden, train_seeds, k=1) == 0.5
