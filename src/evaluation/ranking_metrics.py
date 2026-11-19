"""Pointwise ranking metrics (AUC / LogLoss / Accuracy).

The validation labels used here combine one held-out positive with sampled
implicit negatives.  AUC is therefore a **sampled** ranking diagnostic, not a
real-exposure CTR metric.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def evaluate_binary(
    labels: Sequence[int] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Compute AUC, LogLoss and accuracy for binary predictions."""
    y_true = np.asarray(labels, dtype=np.int64)
    y_prob = np.asarray(probabilities, dtype=np.float64)
    if y_true.shape != y_prob.shape:
        raise ValueError("labels and probabilities must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty evaluation set")
    result: dict[str, float] = {
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, (y_prob >= 0.5).astype(int))),
    }
    if np.unique(y_true).size == 2:
        result["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        result["auc"] = float("nan")
    return result


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    values = np.asarray(relevances[:k], dtype=np.float64)
    if values.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, values.size + 2))
    return float(np.sum(values / discounts))


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg <= 0:
        return 0.0
    return dcg_at_k(relevances, k) / ideal_dcg
