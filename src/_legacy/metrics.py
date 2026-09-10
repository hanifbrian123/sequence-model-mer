"""Standard CASME II metrics computed on POOLED LOSO predictions.

UF1  = macro-averaged F1 (unweighted F1)
UAR  = macro-averaged recall (unweighted average recall / balanced accuracy)
ACC  = overall accuracy
"""
import numpy as np
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix


def compute_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))
    uf1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    uar = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "UF1": float(uf1),
        "UAR": float(uar),
        "ACC": float(acc),
        "per_class_f1": [float(x) for x in per_class_f1],
        "per_class_recall": [float(x) for x in per_class_recall],
        "confusion_matrix": cm.tolist(),
        "n": int(len(y_true)),
    }
