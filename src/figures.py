"""Three figures, written to assets/.

1. model_comparison  — RWR vs the learned model on the recovery metrics.
2. conformal_tradeoff — the honest cost of a coverage guarantee: how big the
   candidate set has to grow as you demand more coverage.
3. disease_module    — the picture from the brief, with real data: known
   Alzheimer genes (red) and the network's top new predictions (blue), drawn on
   their shared STRING interactions.
"""
from __future__ import annotations

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx           # noqa: E402

from . import config as C        # noqa: E402
from . import conformal, embedding, propagation  # noqa: E402
from .data import build_network  # noqa: E402

RED, BLUE, GREY = "#C0392B", "#2E74B5", "#B8B8B8"


def _scores(net, embed):
    seed_idx = np.where(net.seed_mask)[0]
    rwr = propagation.score(net, seed_idx)
    spec = embedding.embed_predict(embed, seed_idx, net.seed_mask)
    return seed_idx, rwr, spec


def model_comparison(cv: dict, path):
    labels = ["AUROC", "AUPRC", "recovery@100"]
    keys = ["auroc", "auprc", "recovery@100"]
    rwr = [cv["rwr"][k]["mean"] for k in keys]
    rwr_e = [cv["rwr"][k]["std"] for k in keys]
    spec = [cv["spectral"][k]["mean"] for k in keys]
    spec_e = [cv["spectral"][k]["std"] for k in keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.bar(x - w / 2, rwr, w, yerr=rwr_e, capsize=3, color=BLUE,
           label="RWR (network propagation)")
    ax.bar(x + w / 2, spec, w, yerr=spec_e, capsize=3, color=RED,
           label="spectral + random forest")
    ax.set_xticks(x, labels)
    ax.set_ylabel("score (5-fold mean)")
    ax.set_title("Recovering hidden Alzheimer genes")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def conformal_tradeoff(net, embed, path):
    seed_idx = np.where(net.seed_mask)[0]
    rng = np.random.default_rng(C.RANDOM_STATE)
    alphas = np.linspace(0.05, 0.6, 12)
    targets = 1 - alphas

    curves = {"rwr": [], "spectral": []}
    for a in alphas:
        sizes = {"rwr": [], "spectral": []}
        for _ in range(10):
            perm = rng.permutation(seed_idx)
            n_tr = len(perm) // 2
            train_idx = perm[:n_tr]
            calib_idx = perm[n_tr:]
            tsm = np.zeros(net.n, dtype=bool); tsm[train_idx] = True
            s_rwr = propagation.score(net, train_idx)
            s_spec = embedding.embed_predict(embed, train_idx, tsm)
            for name, s in (("rwr", s_rwr), ("spectral", s_spec)):
                tau = conformal.conformal_threshold(s[calib_idx], a)
                sizes[name].append(len(conformal.candidate_set(s, tau, exclude=tsm)))
        curves["rwr"].append(np.mean(sizes["rwr"]))
        curves["spectral"].append(np.mean(sizes["spectral"]))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(targets, curves["rwr"], "-o", color=BLUE, ms=4,
            label="RWR (network propagation)")
    ax.plot(targets, curves["spectral"], "-o", color=RED, ms=4,
            label="spectral + random forest")
    ax.set_xlabel("target coverage  (1 − α)")
    ax.set_ylabel("candidate-set size (genes)")
    ax.set_title("The price of a guarantee")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def disease_module(net, embed, path, n_pred: int = 15):
    """The brief's figure with real data: top novel predictions (blue) and the
    *known* Alzheimer genes they directly touch (red), drawn on their shared
    STRING edges. We show only seeds adjacent to a prediction and keep the
    largest connected component, so the module is legible instead of a cloud of
    unconnected dots."""
    seed_idx, rwr, spec = _scores(net, embed)

    def ranks(s):
        r = np.empty(net.n); r[np.argsort(s)[::-1]] = np.arange(net.n); return r
    combined = ranks(rwr) + ranks(spec)
    combined[net.seed_mask] = np.inf
    pred_idx = np.argsort(combined)[:n_pred]

    # seeds that directly interact with at least one prediction
    adj = net.adjacency
    seed_set = set(seed_idx.tolist())
    neigh_seeds = set()
    for p in pred_idx:
        row = adj.getrow(p).indices
        neigh_seeds.update(s for s in row if s in seed_set)
    keep = np.array(sorted(set(pred_idx.tolist()) | neigh_seeds))

    sub = adj[keep][:, keep].tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(len(keep)))
    for i, j, w in zip(sub.row, sub.col, sub.data):
        if i < j:
            G.add_edge(i, j, weight=w)
    if G.number_of_edges():
        giant = max(nx.connected_components(G), key=len)
        G = G.subgraph(giant).copy()

    local = list(G.nodes())
    is_pred = {i: keep[i] in set(pred_idx.tolist()) for i in local}
    labels = {i: net.nodes[keep[i]] for i in local}

    pos = nx.spring_layout(G, seed=C.RANDOM_STATE, k=1.4, iterations=400)
    fig, ax = plt.subplots(figsize=(10.0, 8.5))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=GREY, width=0.7, alpha=0.55)
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=[i for i in local if not is_pred[i]],
        node_color=RED, node_size=1100, label="known Alzheimer gene")
    nx.draw_networkx_nodes(
        G, pos, ax=ax, nodelist=[i for i in local if is_pred[i]],
        node_color=BLUE, node_size=1100, label="predicted gene")
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7,
                            font_color="white", font_weight="bold")
    ax.legend(frameon=False, fontsize=11, loc="lower left", scatterpoints=1)
    ax.set_title("Alzheimer disease module — known genes and new predictions",
                 fontsize=13)
    ax.axis("off")
    ax.margins(0.12)  # keep edge labels off the axes boundary
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def make_all():
    import json
    net = build_network()
    embed = embedding.spectral_embedding(net.adjacency)
    with open(C.RESULTS_DIR / "metrics.json") as fh:
        cv = json.load(fh)["cross_validation"]
    C.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    model_comparison(cv, C.ASSETS_DIR / "model_comparison.png")
    conformal_tradeoff(net, embed, C.ASSETS_DIR / "conformal_tradeoff.png")
    disease_module(net, embed, C.ASSETS_DIR / "disease_module.png")
    print("wrote 3 figures to", C.ASSETS_DIR)


if __name__ == "__main__":
    make_all()
