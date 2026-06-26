"""From a ranking to a guarantee.

A score that orders genes is useful, but a reviewer's real question is "where do
I draw the line, and what does the line promise?" Split-conformal prediction
answers exactly that. We hold out a calibration set of *known* Alzheimer genes,
look at the scores the model gives them, and pick a threshold low enough that the
calibration positives almost all clear it. By exchangeability, a future genuine
disease gene then lands in the candidate set with probability at least 1 - alpha
— a finite-sample guarantee that needs no assumption about the score's
distribution.

This is the honest counterpart to a top-k list: instead of "here are 50 guesses",
it is "this set is built to contain 90% of real disease genes", and the set's
size is itself a readout of how confident the model can afford to be.
"""
from __future__ import annotations

import numpy as np


def conformal_threshold(calib_scores: np.ndarray, alpha: float) -> float:
    """One-sided split-conformal threshold for the positive class.

    `calib_scores` are the model scores on held-out genes that are known to be
    disease genes. We want the candidate set {gene : score >= tau} to capture a
    fraction >= 1 - alpha of such genes. The valid finite-sample choice is the
    rank-(floor(alpha (n+1))) smallest calibration score.
    """
    c = np.sort(np.asarray(calib_scores, dtype=float))
    n = len(c)
    rank = int(np.floor(alpha * (n + 1)))
    if rank < 1:
        return -np.inf          # too few calibration points: keep everything
    return c[rank - 1]


def candidate_set(scores: np.ndarray, threshold: float,
                  exclude: np.ndarray | None = None) -> np.ndarray:
    """Indices of genes whose score clears the conformal threshold.

    `exclude` (e.g. the training seeds) is dropped so the set is the *new*
    candidates a researcher would actually follow up.
    """
    keep = scores >= threshold
    if exclude is not None:
        keep = keep & ~exclude
    return np.where(keep)[0]


def empirical_coverage(test_pos_scores: np.ndarray, threshold: float) -> float:
    """Fraction of held-out true disease genes the set actually captures."""
    if len(test_pos_scores) == 0:
        return float("nan")
    return float(np.mean(test_pos_scores >= threshold))
