"""LOSO runner for the apex optical-flow shallow-net (STSTNet-family).

Fast (tiny net, single flow image/sample). Saves probs.npz for fusion with the
sequence models. Usage:
  python src/run_apex.py --config configs/apex_ststnet.json [--max_folds N] [--tag smoke]
"""
import os, sys, json, time, argparse
from datetime import datetime
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apex_dataset import ApexFlowDataset
from shallow_models import build_shallow
from metrics import compute_metrics
from engine import class_weights, FocalLoss
from run_experiment import preload_arrays, plot_confusion

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLS = ['happiness', 'disgust', 'repression', 'surprise', 'others']

DEFAULTS = dict(seed=42, img_size=64, crop=112, base_size=128, flow_clip=3.0,
                strain_clip=0.3, shallow="ststnet", in_channels=3, dropout=0.5,
                epochs=60, lr=1e-3, weight_decay=1e-3, batch_size=32,
                class_weighting=True, label_smoothing=0.05, hflip=True, loss="ce",
                focal_gamma=2.0, tta=10, eval_last_k=5,
                cache_dir="cache/flow128", manifest_name="manifest.csv",
                index="cache/index_emotion.csv",
                class_names=CLS)


def load_config(p):
    cfg = dict(DEFAULTS); cfg.update(json.load(open(p))); return cfg


def loader(samples, arrays, cfg, train, shuffle=None):
    ds = ApexFlowDataset(samples, arrays, cfg, train)
    return DataLoader(ds, batch_size=cfg["batch_size"],
                      shuffle=train if shuffle is None else shuffle, num_workers=0)


@torch.no_grad()
def predict(model, samples, arrays, cfg, device, tta):
    model.eval(); probs = None
    for _ in range(max(1, tta)):
        ps = []
        for x, _ in loader(samples, arrays, cfg, train=(tta > 1), shuffle=False):
            x = x.to(device)
            ps.append(torch.softmax(model(x).float(), 1).cpu().numpy())
        p = np.concatenate(ps, 0)
        probs = p if probs is None else probs + p
    return probs / max(1, tta)


def train_fold(tr, va, arrays, cfg, ncls, device):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    model = build_shallow(cfg, ncls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    w = class_weights(tr, ncls, device) if cfg.get("class_weighting", True) else None
    crit = FocalLoss(weight=w, gamma=cfg["focal_gamma"]) if cfg.get("loss") == "focal" \
        else nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    tl = loader(tr, arrays, cfg, train=True)
    acc_p, n = None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        sched.step()
        if ep >= cfg["epochs"] - cfg["eval_last_k"]:
            p = predict(model, va, arrays, cfg, device, cfg["tta"])
            acc_p = p if acc_p is None else acc_p + p; n += 1
    return acc_p / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_folds", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    name = os.path.splitext(os.path.basename(args.config))[0] + (f"_{args.tag}" if args.tag else "")
    exp = os.path.join(REPO, "experiments", name); os.makedirs(exp, exist_ok=True)
    logf = open(os.path.join(exp, "run.log"), "w", encoding="utf-8")

    def log(m):
        print(m); logf.write(m + "\n"); logf.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"=== APEX {name} === {datetime.now().isoformat()} device={device}")
    log("config:\n" + json.dumps(cfg, indent=2))
    ncls = len(cfg["class_names"])
    man = pd.read_csv(os.path.join(REPO, cfg["cache_dir"], cfg["manifest_name"]))
    arrays = preload_arrays(os.path.join(REPO, cfg["cache_dir"]), man)
    samples = [dict(key=r["key"], subject=int(r["subject"]), label=int(r["label"]),
                    apex_pos=max(int(r["apex"]) - int(r["onset"]), -1)) for _, r in man.iterrows()]
    subjects = sorted(set(s["subject"] for s in samples))
    if args.max_folds > 0:
        subjects = subjects[:args.max_folds]

    yt, yp, keys, subj, probs = [], [], [], [], []
    t0 = time.time()
    for fi, sub in enumerate(subjects):
        va = [s for s in samples if s["subject"] == sub]
        tr = [s for s in samples if s["subject"] != sub]
        p = train_fold(tr, va, arrays, cfg, ncls, device)
        pr = p.argmax(1)
        for s, pp, prob in zip(va, pr, p):
            yt.append(s["label"]); yp.append(int(pp)); keys.append(s["key"])
            subj.append(sub); probs.append(prob.tolist())
        acc = float(np.mean([int(a) == s["label"] for s, a in zip(va, pr)]))
        log(f"fold {fi+1}/{len(subjects)} sub{sub:02d} acc={acc:.3f} "
            f"| pooled acc={np.mean(np.array(yt)==np.array(yp)):.3f} ({time.time()-t0:.0f}s)")

    m = compute_metrics(yt, yp, ncls)
    log(f"\n=== APEX FINAL ===\nUF1 = {m['UF1']:.4f}\nUAR = {m['UAR']:.4f}\nACC = {m['ACC']:.4f}")
    log("per-class F1: " + ", ".join(f"{c}={v:.3f}" for c, v in zip(CLS, m["per_class_f1"])))
    for i, row in enumerate(m["confusion_matrix"]):
        log(f"{CLS[i][:7]:>7} " + " ".join(f"{v:6d}" for v in row))
    np.savez(os.path.join(exp, "probs.npz"), keys=np.array(keys), label=np.array(yt),
             probs=np.array(probs, dtype=np.float32))
    pd.DataFrame({"key": keys, "subject": subj, "label": yt, "pred": yp}
                ).to_csv(os.path.join(exp, "predictions.csv"), index=False)
    plot_confusion(m["confusion_matrix"], CLS, os.path.join(exp, "confusion_matrix.png"))
    el = round(time.time() - t0, 1)
    json.dump({"name": name, **{k: m[k] for k in ("UF1", "UAR", "ACC")},
               "per_class_f1": m["per_class_f1"], "elapsed_sec": el, "config": cfg},
              open(os.path.join(exp, "summary.json"), "w"), indent=2)
    led = os.path.join(REPO, "results_ledger.csv")
    row = {"name": name, "time": datetime.now().isoformat(), "UF1": round(m["UF1"], 4),
           "UAR": round(m["UAR"], 4), "ACC": round(m["ACC"], 4), "n_folds": len(subjects),
           "backbone": cfg["shallow"], "T": 1, "img_size": cfg["img_size"],
           "input_mode": "apex_flow", "epochs": cfg["epochs"], "lr": cfg["lr"],
           "elapsed_sec": el, "index": cfg["index"], "num_classes": ncls}
    pd.DataFrame([row]).to_csv(led, mode="a", header=not os.path.exists(led), index=False)
    log(f"\nsaved -> {exp}  elapsed {el}s")
    logf.close()


if __name__ == "__main__":
    main()
