"""Two-stage LOSO for the 'others' bottleneck.

Stage 1 (binary): others (label4) vs expression (labels 0-3 collapsed).
Stage 2 (4-class): among expressions {happiness,disgust,repression,surprise},
                   trained ONLY on non-others samples.
Combine per test sample:
   p5[0:4] = P(expression) * P_stage2[0:4]
   p5[4]   = P(others)
This decouples the semantic grab-bag 'others' from the fine expression decision.

Reuses engine.train_fold. Usage:
  python src/run_twostage.py --config configs/iter_14_r3d.json [--max_folds N] [--tag smoke]
"""
import os, sys, json, time, argparse
from datetime import datetime
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import train_fold
from metrics import compute_metrics
from run_experiment import DEFAULTS, load_config, preload_arrays, plot_confusion

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = ['happiness', 'disgust', 'repression', 'surprise', 'others']


def relabel(samples, mode):
    out = []
    for s in samples:
        d = dict(s)
        if mode == 'binary':
            d['label'] = 1 if s['label'] == 4 else 0
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--max_folds', type=int, default=0)
    ap.add_argument('--tag', default='')
    args = ap.parse_args()

    cfg = load_config(args.config)
    name = 'twostage_' + os.path.splitext(os.path.basename(args.config))[0]
    if args.tag:
        name += f'_{args.tag}'
    exp = os.path.join(REPO, 'experiments', name)
    os.makedirs(exp, exist_ok=True)
    logf = open(os.path.join(exp, 'run.log'), 'w', encoding='utf-8')

    def log(m):
        print(m); logf.write(m + '\n'); logf.flush()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log(f'=== TWO-STAGE {name} ===  {datetime.now().isoformat()}  device={device}')
    manifest = pd.read_csv(os.path.join(REPO, cfg['cache_dir'], cfg.get('manifest_name', 'manifest.csv')))
    arrays = preload_arrays(os.path.join(REPO, cfg['cache_dir']), manifest)
    samples = [dict(key=r['key'], subject=int(r['subject']), label=int(r['label']),
                    apex_pos=max(int(r['apex']) - int(r['onset']), -1)) for _, r in manifest.iterrows()]
    subjects = sorted(set(s['subject'] for s in samples))
    if args.max_folds > 0:
        subjects = subjects[:args.max_folds]

    pooled_true, pooled_pred, pooled_key, pooled_subj, pooled_probs = [], [], [], [], []
    t0 = time.time()
    for fi, subj in enumerate(subjects):
        val_s = [s for s in samples if s['subject'] == subj]
        tr_s = [s for s in samples if s['subject'] != subj]
        ft = time.time()

        # stage 1: binary others-vs-expression
        p1, _ = train_fold(relabel(tr_s, 'binary'), relabel(val_s, 'binary'),
                           arrays, cfg, 2, device, lambda m: None)
        # stage 2: 4-class among expressions, trained on non-others only.
        # val labels clamped to 0-3 so the internal val-loss monitor doesn't index
        # out of range (label 4). Only stage-2 PROBS are used downstream, not this label.
        tr_expr = [s for s in tr_s if s['label'] != 4]
        val_s2 = [dict(s, label=min(s['label'], 3)) for s in val_s]
        p2, _ = train_fold(tr_expr, val_s2, arrays, cfg, 4, device, lambda m: None)

        # combine -> 5-class probs
        p_others = p1[:, 1]
        p_expr = p1[:, 0]
        p5 = np.zeros((len(val_s), 5), dtype=np.float32)
        p5[:, 0:4] = p_expr[:, None] * p2
        p5[:, 4] = p_others
        preds = p5.argmax(1)
        for s, pr, pp in zip(val_s, p5, preds):
            pooled_true.append(s['label']); pooled_pred.append(int(pp))
            pooled_key.append(s['key']); pooled_subj.append(subj); pooled_probs.append(pr.tolist())
        acc = float(np.mean([int(p) == s['label'] for s, p in zip(val_s, preds)]))
        log(f'fold {fi+1}/{len(subjects)} sub{subj:02d} acc={acc:.3f} ({time.time()-ft:.0f}s) '
            f'| pooled acc={np.mean(np.array(pooled_true)==np.array(pooled_pred)):.3f}')

    m = compute_metrics(pooled_true, pooled_pred, 5)
    log(f'\n=== TWO-STAGE FINAL ===\nUF1 = {m["UF1"]:.4f}\nUAR = {m["UAR"]:.4f}\nACC = {m["ACC"]:.4f}')
    log('per-class F1: ' + ', '.join(f'{c}={v:.3f}' for c, v in zip(CLS, m['per_class_f1'])))
    for i, row in enumerate(m['confusion_matrix']):
        log(f'{CLS[i][:7]:>7} ' + ' '.join(f'{v:6d}' for v in row))

    np.savez(os.path.join(exp, 'probs.npz'), keys=np.array(pooled_key),
             label=np.array(pooled_true), probs=np.array(pooled_probs, dtype=np.float32))
    pd.DataFrame({'key': pooled_key, 'subject': pooled_subj, 'label': pooled_true,
                  'pred': pooled_pred}).to_csv(os.path.join(exp, 'predictions.csv'), index=False)
    plot_confusion(m['confusion_matrix'], CLS, os.path.join(exp, 'confusion_matrix.png'))
    elapsed = round(time.time() - t0, 1)
    json.dump({'name': name, 'UF1': m['UF1'], 'UAR': m['UAR'], 'ACC': m['ACC'],
               'per_class_f1': m['per_class_f1'], 'elapsed_sec': elapsed, 'config': cfg},
              open(os.path.join(exp, 'summary.json'), 'w'), indent=2)
    row = {'name': name, 'time': datetime.now().isoformat(), 'UF1': round(m['UF1'], 4),
           'UAR': round(m['UAR'], 4), 'ACC': round(m['ACC'], 4), 'n_folds': len(subjects),
           'backbone': cfg['backbone'], 'T': cfg['T'], 'img_size': cfg['img_size'],
           'input_mode': 'twostage', 'epochs': cfg['epochs'], 'lr': cfg['lr'],
           'elapsed_sec': elapsed, 'index': cfg['index'], 'num_classes': 5}
    led = os.path.join(REPO, 'results_ledger.csv')
    pd.DataFrame([row]).to_csv(led, mode='a', header=not os.path.exists(led), index=False)
    log(f'\nsaved -> {exp}  elapsed {elapsed}s')
    logf.close()


if __name__ == '__main__':
    main()
