"""
Weighted, imbalance-aware metrics and reporting utilities used in both objectives.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report, confusion_matrix
)


def pr_auc_weighted(y_true, y_prob, w):
    """Weighted PR-AUC (a.k.a. Average Precision) using survey weights."""
    return float(average_precision_score(y_true, y_prob, sample_weight=w))


def roc_auc_weighted(y_true, y_prob, w):
    """Weighted ROC-AUC using survey weights."""
    return float(roc_auc_score(y_true, y_prob, sample_weight=w))


def best_f1_threshold_from_pr(y_true, y_prob, w):
    """Pick the single threshold that maximizes weighted F1 on validation."""
    p, r, t = precision_recall_curve(y_true, y_prob, sample_weight=w)
    f1 = 2 * p * r / (p + r + 1e-12)
    idx = np.nanargmax(f1[:-1])
    return float(t[idx]), float(f1[idx])


def eval_at_threshold(y_true, y_prob, w, thr):
    """Return a dict of weighted precision/recall/F1, plus confusion matrix."""
    y_pred = (y_prob >= thr).astype(int)
    # Classification report is convenient but we compute the essentials directly
    cm = confusion_matrix(y_true, y_pred, sample_weight=w)
    tn, fp, fn, tp = cm.ravel()
    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return dict(precision=float(prec), recall=float(rec), f1=float(f1), cm=cm)