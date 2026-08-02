"""Standard EER computation (used for all Claim 1/2/3 metrics)."""
from __future__ import annotations

import numpy as np


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """labels: 1 = genuine (bona fide), 0 = spoofed. scores: higher = more genuine.

    EER = point where false acceptance rate (spoof accepted as genuine) equals
    false rejection rate (genuine rejected as spoof).
    """
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]
    n_genuine = labels.sum()
    n_spoof = len(labels) - n_genuine
    if n_genuine == 0 or n_spoof == 0:
        return float("nan")

    thresholds = np.unique(scores)
    fars, frrs = [], []
    for t in thresholds:
        accepted = scores >= t
        far = ((accepted) & (labels == 0)).sum() / n_spoof
        frr = ((~accepted) & (labels == 1)).sum() / n_genuine
        fars.append(far)
        frrs.append(frr)
    fars = np.array(fars)
    frrs = np.array(frrs)
    diffs = np.abs(fars - frrs)
    idx = int(np.argmin(diffs))
    eer = (fars[idx] + frrs[idx]) / 2.0
    return float(eer * 100.0)  # percent
