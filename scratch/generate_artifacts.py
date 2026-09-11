"""
generate_artifacts.py — Generate ROC curves, training curves, and detailed evaluation metrics
for the top champion models in CASME II.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ["happiness", "disgust", "repression", "surprise", "others"]
CLASS_COLORS = ["#2ecc71", "#e74c3c", "#9b59b6", "#3498db", "#e67e22"]

TARGET_RUNS = [
    "188_r3d_au05_full_s42_v2_dev_loso_all_p5",
    "017_r3d",
    "182_r3d_focus_region_full_s42_v2_dev_loso_all_p5",
    "203_expD_flow_strain_v2_dev_loso_all_p5",
    "163_r3d_regionattn_full_s42_v2_dev_loso_dev_p5",
    "116_g2_disgust_others_s123_v2_dev_p5",
]

def generate_roc_curve(run_dir, run_name):
    probs_path = os.path.join(run_dir, "probs.npz")
    if not os.path.exists(probs_path):
        print(f"Skipping ROC for {run_name}: probs.npz not found")
        return None

    data = np.load(probs_path)
    probs = data["probs"]
    labels = data["label"]
    num_classes = probs.shape[1]
    class_names = CLASS_NAMES[:num_classes]

    labels_bin = label_binarize(labels, classes=list(range(num_classes)))
    if num_classes == 2:
        labels_bin = np.hstack([1 - labels_bin, labels_bin])

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(num_classes):
        # Handle cases where a class might have no true instances in a test subset
        if np.sum(labels_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            roc_auc[i] = 0.0

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes = [i for i in range(num_classes) if i in fpr]
    for i in valid_classes:
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(valid_classes)
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Weighted-average
    roc_auc["weighted"] = roc_auc_score(labels_bin, probs, average="weighted", multi_class="ovr")

    # Plot ROC curve
    plt.figure(figsize=(8, 7), dpi=300)
    plt.plot(fpr["macro"], tpr["macro"],
             label=f"Macro-average (AUC = {roc_auc['macro']:.4f})",
             color="#1f77b4", linestyle="--", linewidth=2.5)
    plt.plot(fpr["micro"], tpr["micro"],
             label=f"Micro-average (AUC = {roc_auc['micro']:.4f})",
             color="#17becf", linestyle=":", linewidth=2)

    for i in range(num_classes):
        if i in fpr:
            plt.plot(fpr[i], tpr[i],
                     color=CLASS_COLORS[i % len(CLASS_COLORS)],
                     linewidth=1.8,
                     label=f"{class_names[i].capitalize()} (AUC = {roc_auc[i]:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6, label="Random Guess (AUC = 0.5000)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold")
    plt.title(f"ROC Curves — {run_name}", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(run_dir, "roc_curve.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[OK] Generated ROC curve: {out_path} (Macro-AUC: {roc_auc['macro']:.4f})")
    return roc_auc


def generate_training_curve(run_dir, run_name):
    history_path = os.path.join(run_dir, "epoch_history.jsonl")
    if not os.path.exists(history_path):
        print(f"Skipping Training Curve for {run_name}: epoch_history.jsonl not found")
        return

    records = []
    with open(history_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    if "epoch" not in df.columns:
        print(f"Skipping Training Curve for {run_name}: no 'epoch' column")
        return

    # Group by epoch across all folds
    grouped = df.groupby("epoch")
    epochs = np.array(sorted(df["epoch"].unique()))

    train_loss_mean = grouped["train_loss"].mean()
    train_loss_std = grouped["train_loss"].std().fillna(0)
    val_loss_mean = grouped["val_loss"].mean()
    val_loss_std = grouped["val_loss"].std().fillna(0)

    train_acc_mean = grouped["train_acc"].mean() * 100
    train_acc_std = grouped["train_acc"].std().fillna(0) * 100
    val_acc_mean = grouped["val_acc"].mean() * 100
    val_acc_std = grouped["val_acc"].std().fillna(0) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # Subplot 1: Loss
    ax1.plot(epochs, train_loss_mean, color="#2980b9", linewidth=2, label="Train Loss (Mean)")
    ax1.fill_between(epochs, train_loss_mean - train_loss_std, train_loss_mean + train_loss_std,
                     color="#2980b9", alpha=0.15, label="±1 Std Dev")
    ax1.plot(epochs, val_loss_mean, color="#e74c3c", linewidth=2, linestyle="--", label="Val Loss (Mean)")
    ax1.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std,
                     color="#e74c3c", alpha=0.15)
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Loss (Cross-Entropy)", fontsize=11, fontweight="bold")
    ax1.set_title("Training & Validation Loss Convergence", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", fontsize=9.5)

    # Subplot 2: Accuracy
    ax2.plot(epochs, train_acc_mean, color="#27ae60", linewidth=2, label="Train Accuracy (Mean)")
    ax2.fill_between(epochs, train_acc_mean - train_acc_std, train_acc_mean + train_acc_std,
                     color="#27ae60", alpha=0.15, label="±1 Std Dev")
    ax2.plot(epochs, val_acc_mean, color="#8e44ad", linewidth=2, linestyle="--", label="Val Accuracy (Mean)")
    ax2.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std,
                     color="#8e44ad", alpha=0.15)
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Training & Validation Accuracy Progression", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=9.5)

    plt.suptitle(f"Learning Curves (Across All Folds) — {run_name}", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out_path = os.path.join(run_dir, "curve_training.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated Training curve: {out_path}")


def compute_detailed_metrics(run_dir, run_name):
    preds_path = os.path.join(run_dir, "predictions.csv")
    if not os.path.exists(preds_path):
        return None

    df = pd.read_csv(preds_path)
    y_true = df["label"].values
    y_pred = df["pred"].values
    num_classes = len(np.unique(np.concatenate([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_prec = np.mean(prec)
    macro_rec = np.mean(rec)
    macro_f1 = np.mean(f1)
    acc = np.mean(y_true == y_pred)

    # Specificity per class: TN / (TN + FP)
    specificities = []
    for i in range(len(cm)):
        tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_spec = np.mean(specificities)

    return {
        "run": run_name,
        "n_samples": len(df),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_specificity": macro_spec,
        "per_class_f1": f1.tolist(),
        "per_class_rec": rec.tolist(),
        "per_class_prec": prec.tolist(),
        "per_class_spec": specificities,
    }


def main():
    print("=== Generating Visual Artifacts & Detailed Metrics for Top Runs ===")
    summary_table = []
    for r in TARGET_RUNS:
        run_dir = os.path.join("runs", r)
        print(f"\nProcessing {r}...")
        roc_dict = generate_roc_curve(run_dir, r)
        generate_training_curve(run_dir, r)
        metrics = compute_detailed_metrics(run_dir, r)
        if metrics and roc_dict:
            metrics["macro_auc"] = roc_dict.get("macro", 0.0)
            metrics["weighted_auc"] = roc_dict.get("weighted", 0.0)
            metrics["per_class_auc"] = [roc_dict.get(i, 0.0) for i in range(len(CLASS_NAMES))]
            summary_table.append(metrics)

    # Save summary table as json for report generation
    with open("scratch/top_runs_detailed_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)
    print("\nSaved detailed metrics to scratch/top_runs_detailed_metrics.json")

if __name__ == "__main__":
    main()
