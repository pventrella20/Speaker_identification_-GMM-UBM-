"""Metrics/plotting, isolated from orchestration logic (main.py mixed them together).

Fixes one real bug: `main.cmatrix_display` was called with
`confusion_matrix(SR.y_true, SR.y_predict, SR.speakers_label)`. In scikit-learn
`confusion_matrix(y_true, y_pred, *, labels=None, sample_weight=None, ...)` --
`labels` is keyword-only (the `*`). Passing it positionally raises a `TypeError`
on any reasonably modern scikit-learn. Here it's passed as `labels=...`.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def build_confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def display_confusion_matrix(cm: np.ndarray, labels: list[str], accuracy: float, figsize=(10, 7)) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sn

    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    plt.figure(figsize=figsize)
    plt.title(f"Confusion Matrix\nAccuracy: {accuracy:.3f}")
    sn.heatmap(df_cm, cmap="rocket_r", annot=True, fmt="g")
    plt.ylabel("True speaker")
    plt.xlabel("Predicted speaker")
    plt.subplots_adjust(bottom=0.155, right=0.924)
    plt.show()
