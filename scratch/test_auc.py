import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

data = np.load('runs/188_r3d_au05_full_s42_v2_dev_loso_all_p5/probs.npz')
probs = data['probs'] # (N, 5)
labels = data['label'] # (N,)
num_classes = probs.shape[1]

# Binarize labels
from sklearn.preprocessing import label_binarize
labels_bin = label_binarize(labels, classes=list(range(num_classes)))

macro_auc = roc_auc_score(labels_bin, probs, average='macro', multi_class='ovr')
micro_auc = roc_auc_score(labels_bin, probs, average='micro', multi_class='ovr')
weighted_auc = roc_auc_score(labels_bin, probs, average='weighted', multi_class='ovr')

print(f"Run 188 AUC (Macro): {macro_auc:.4f}")
print(f"Run 188 AUC (Micro): {micro_auc:.4f}")
print(f"Run 188 AUC (Weighted): {weighted_auc:.4f}")

for c in range(num_classes):
    c_auc = roc_auc_score(labels_bin[:, c], probs[:, c])
    print(f"  Class {c} AUC: {c_auc:.4f}")
