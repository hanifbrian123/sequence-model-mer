"""Late-fusion of saved LOSO probabilities (experiments/*/probs.npz).

Averages per-sample softmax probs across several runs (diverse models fuse well
even when individually weaker). Can also search for the best subset.

Usage:
  python src/fuse.py --runs iter_14_r3d iter_13_4ch iter_16_mag_r3d
  python src/fuse.py --search iter_09 iter_10 ... --max_k 4
"""
import os, sys, json, argparse
import numpy as np
from itertools import combinations
from sklearn.metrics import f1_score, recall_score, accuracy_score, confusion_matrix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = ['happiness', 'disgust', 'repression', 'surprise', 'others']


def load(run):
    z = np.load(os.path.join(REPO, 'experiments', run, 'probs.npz'), allow_pickle=True)
    keys = [str(k) for k in z['keys']]
    return dict(zip(keys, z['probs'])), dict(zip(keys, z['label']))


def fuse(runs, weights=None):
    probs, labels = {}, {}
    for r in runs:
        p, l = load(r)
        probs[r] = p; labels[r] = l
    common = None
    for r in runs:
        s = set(probs[r].keys())
        common = s if common is None else common & s
    common = sorted(common)
    y = np.array([labels[runs[0]][k] for k in common])
    w = weights or [1.0] * len(runs)
    P = sum(wi * np.stack([probs[r][k] for k in common]) for wi, r in zip(w, runs)) / sum(w)
    yp = P.argmax(1)
    return y, yp, P, common


def metrics(y, yp):
    labels = list(range(len(CLS)))
    return (f1_score(y, yp, labels=labels, average='macro', zero_division=0),
            recall_score(y, yp, labels=labels, average='macro', zero_division=0),
            accuracy_score(y, yp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=None)
    ap.add_argument('--search', nargs='+', default=None)
    ap.add_argument('--max_k', type=int, default=4)
    ap.add_argument('--save', default=None, help='save fused probs.npz to experiments/<name>')
    args = ap.parse_args()

    if args.search:
        pool = [r for r in args.search
                if os.path.exists(os.path.join(REPO, 'experiments', r, 'probs.npz'))]
        results = []
        for k in range(1, args.max_k + 1):
            for combo in combinations(pool, k):
                y, yp, _, _ = fuse(list(combo))
                u, ua, a = metrics(y, yp)
                results.append((u, a, ua, combo))
        results.sort(reverse=True)
        print(f"pool={pool}\ntop-12 by UF1:")
        for u, a, ua, c in results[:12]:
            print(f"  UF1={u:.4f} UAR={ua:.4f} ACC={a:.4f}  {'+'.join(c)}")
        return

    runs = args.runs
    y, yp, P, common = fuse(runs)
    u, ua, a = metrics(y, yp)
    print(f"FUSION of {runs}")
    print(f"n={len(common)}  UF1={u:.4f}  UAR={ua:.4f}  ACC={a:.4f}")
    cm = confusion_matrix(y, yp, labels=list(range(len(CLS))))
    print("confusion (rows=true):")
    print("        " + " ".join(f"{c[:6]:>6}" for c in CLS))
    for i, row in enumerate(cm):
        print(f"{CLS[i][:7]:>7} " + " ".join(f"{v:6d}" for v in row))
    if args.save:
        d = os.path.join(REPO, 'experiments', args.save)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, 'probs.npz'),
                 keys=np.array(common), label=y, probs=P.astype(np.float32))
        json.dump({'runs': runs, 'UF1': u, 'UAR': ua, 'ACC': a},
                  open(os.path.join(d, 'summary.json'), 'w'), indent=2)
        print(f"saved fused -> {d}")


if __name__ == '__main__':
    main()
