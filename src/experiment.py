"""The experiment, end to end.

The honest test of a disease-gene predictor is not "does it score the known
genes highly" — it was handed those. It is "if we hide some known genes, can it
find them again among 15,000 candidates it has never been told about?" So we run
stratified k-fold over the seed set: each fold hides a fifth of the Alzheimer
genes, trains on the rest, and scores the whole genome. The hidden genes are the
positives we hope to recover; every other gene is a negative.

Both models — random-walk propagation and the spectral+logistic learner — see
the *same* folds, so the comparison is fair. On top of the ranking we run
split-conformal calibration and report whether the promised coverage holds.

Finally we retrain on all 80 seeds and write out the top novel candidates: genes
the network nominates as Alzheimer-associated that are not yet in the seed set.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold

from . import config as C
from . import conformal, embedding, propagation
from .data import build_network


def _recovery_at_k(scores: np.ndarray, hidden_idx: np.ndarray,
                   train_seed_idx: np.ndarray, k: int) -> float:
    """Fraction of hidden seeds that land in the top-k of the genome, after
    removing the training seeds (which trivially top the list)."""
    order = np.argsort(scores)[::-1]
    mask = ~np.isin(order, train_seed_idx)
    ranked = order[mask][:k]
    return float(np.isin(hidden_idx, ranked).mean())


def _model_scores(net, embed, train_idx):
    """Both models scored on the same training seeds."""
    train_seed_mask = np.zeros(net.n, dtype=bool)
    train_seed_mask[train_idx] = True
    return {
        "rwr": propagation.score(net, train_idx),
        "spectral": embedding.embed_predict(embed, train_idx, train_seed_mask),
    }


def cross_validate(net, embed: np.ndarray) -> dict:
    """k-fold seed recovery for both models on identical folds."""
    seed_idx = np.where(net.seed_mask)[0]
    kf = KFold(n_splits=C.N_SPLITS, shuffle=True, random_state=C.RANDOM_STATE)

    metrics = {m: {"auroc": [], "auprc": [], "recovery@100": []}
               for m in ("rwr", "spectral")}

    for train_pos, test_pos in kf.split(seed_idx):
        train_idx, hidden_idx = seed_idx[train_pos], seed_idx[test_pos]

        # positives = hidden seeds, negatives = genes that are not seeds at all;
        # training seeds are excluded so they cannot inflate the score.
        eval_mask = np.ones(net.n, dtype=bool)
        eval_mask[train_idx] = False
        y = np.zeros(net.n, dtype=int)
        y[hidden_idx] = 1

        for name, s in _model_scores(net, embed, train_idx).items():
            metrics[name]["auroc"].append(roc_auc_score(y[eval_mask], s[eval_mask]))
            metrics[name]["auprc"].append(
                average_precision_score(y[eval_mask], s[eval_mask]))
            metrics[name]["recovery@100"].append(
                _recovery_at_k(s, hidden_idx, train_idx, 100))

    def _agg(d):
        return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in d.items()}

    return {"rwr": _agg(metrics["rwr"]), "spectral": _agg(metrics["spectral"])}


def conformal_eval(net, embed: np.ndarray, repeats: int = 20) -> dict:
    """Does the conformal candidate set keep its coverage promise?

    Each repeat splits the seeds three ways — train / calibration / test. The
    threshold is fixed on the calibration genes alone; coverage is then measured
    on the never-seen test genes. A valid procedure should land near the target
    1 - alpha. We also report how large the resulting candidate set is, because
    a guarantee is only useful if the set is small enough to act on.
    """
    seed_idx = np.where(net.seed_mask)[0]
    rng = np.random.default_rng(C.RANDOM_STATE)
    out = {m: {"coverage": [], "set_size": []} for m in ("rwr", "spectral")}

    for _ in range(repeats):
        perm = rng.permutation(seed_idx)
        n_tr = len(perm) // 2
        n_cal = (len(perm) - n_tr) // 2
        train_idx = perm[:n_tr]
        calib_idx = perm[n_tr:n_tr + n_cal]
        test_idx = perm[n_tr + n_cal:]

        train_seed_mask = np.zeros(net.n, dtype=bool)
        train_seed_mask[train_idx] = True

        for name, s in _model_scores(net, embed, train_idx).items():
            tau = conformal.conformal_threshold(s[calib_idx], C.CONFORMAL_ALPHA)
            out[name]["coverage"].append(
                conformal.empirical_coverage(s[test_idx], tau))
            cand = conformal.candidate_set(s, tau, exclude=train_seed_mask)
            out[name]["set_size"].append(len(cand))

    return {
        "target_coverage": 1 - C.CONFORMAL_ALPHA,
        "repeats": repeats,
        "rwr": {"coverage": float(np.mean(out["rwr"]["coverage"])),
                "candidate_set_size": float(np.mean(out["rwr"]["set_size"]))},
        "spectral": {"coverage": float(np.mean(out["spectral"]["coverage"])),
                     "candidate_set_size": float(np.mean(out["spectral"]["set_size"]))},
    }


def discover(net, embed: np.ndarray, top: int = 25) -> list[dict]:
    """Retrain on every seed and return the highest-scoring non-seed genes,
    where both models agree (rank product), as the discovery shortlist."""
    seed_idx = np.where(net.seed_mask)[0]
    rwr = propagation.score(net, seed_idx)
    spec = embedding.embed_predict(embed, seed_idx, net.seed_mask)

    def ranks(s):
        r = np.empty(net.n)
        r[np.argsort(s)[::-1]] = np.arange(net.n)
        return r

    combined = ranks(rwr) + ranks(spec)        # lower = better in both
    combined[net.seed_mask] = np.inf            # drop known seeds
    best = np.argsort(combined)[:top]
    return [
        {"gene": net.nodes[i],
         "rwr_rank": int(ranks(rwr)[i]) + 1,
         "spectral_rank": int(ranks(spec)[i]) + 1}
        for i in best
    ]


def run() -> dict:
    t0 = time.time()
    net = build_network()
    embed = embedding.spectral_embedding(net.adjacency)

    result = {
        "graph": {
            "disease": C.DISEASE_NAME,
            "nodes": net.n,
            "edges": int((net.adjacency > 0).nnz // 2),
            "seed_genes": net.n_seeds,
        },
        "cross_validation": cross_validate(net, embed),
        "conformal": conformal_eval(net, embed),
        "discoveries": discover(net, embed),
        "runtime_seconds": round(time.time() - t0, 1),
    }

    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.RESULTS_DIR / "metrics.json", "w") as fh:
        json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    out = run()
    cv = out["cross_validation"]
    print(f"\nAlzheimer disease-gene recovery  ({out['graph']['seed_genes']} seeds, "
          f"{out['graph']['nodes']} genes)\n")
    print(f"{'model':<12}{'AUROC':>14}{'AUPRC':>14}{'recovery@100':>16}")
    for m in ("rwr", "spectral"):
        a, p, r = cv[m]["auroc"], cv[m]["auprc"], cv[m]["recovery@100"]
        print(f"{m:<12}{a['mean']:>8.3f}±{a['std']:.3f}"
              f"{p['mean']:>8.3f}±{p['std']:.3f}"
              f"{r['mean']:>10.3f}±{r['std']:.3f}")
    co = out["conformal"]
    print(f"\nconformal target coverage {co['target_coverage']:.0%}:")
    for m in ("rwr", "spectral"):
        print(f"  {m:<10} coverage {co[m]['coverage']:.0%}, "
              f"candidate set ~{co[m]['candidate_set_size']:.0f} genes")
    print("\ntop novel candidates:",
          ", ".join(d["gene"] for d in out["discoveries"][:10]))
