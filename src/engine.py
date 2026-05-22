"""Training/eval for a single LOSO fold."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import SeqDataset, TwoStreamDataset
from models import build_model


def move_to(x, device):
    """Move a tensor or a list/tuple of tensors to device."""
    if isinstance(x, (list, tuple)):
        return [t.to(device, non_blocking=True) for t in x]
    return x.to(device, non_blocking=True)


def batch_size_of(x):
    return x[0].size(0) if isinstance(x, (list, tuple)) else x.size(0)


def make_loader(samples, arrays, cfg, train, shuffle=None):
    """`train` controls dataset augmentation; `shuffle` controls loader ordering.
    For TTA we want augmentation (train=True) but NO shuffle (preserve sample order)."""
    if shuffle is None:
        shuffle = train
    if cfg.get("modality") == "two_stream":
        arrays_a, arrays_b = arrays   # tuple (flow, appearance)
        ds = TwoStreamDataset(samples, arrays_a, arrays_b, cfg, train)
    else:
        ds = SeqDataset(samples, arrays, cfg, train)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      num_workers=0, drop_last=False, pin_memory=True)


def class_weights(train_samples, num_classes, device, cap=10.0):
    """Inverse-frequency class weights, capped to avoid pathological weights for
    ultra-rare classes (e.g. a 1-sample class -> 36x weight wrecks training)."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in train_samples:
        counts[s["label"]] += 1
    counts = np.maximum(counts, 1)
    w = counts.sum() / (num_classes * counts)   # inverse frequency
    w = np.clip(w, 1.0 / cap, cap)              # cap extreme weights
    return torch.tensor(w, dtype=torch.float32, device=device)


class EMA:
    """Exponential moving average of model weights (params + BN buffers).
    On tiny, high-variance data the per-epoch val curve is noisy and the SGD
    iterate wanders around a flat basin; averaging the tail weights lands in a
    flatter minimum that generalizes better (SWA/EMA). Cheap: one shadow copy,
    no extra forward passes during training."""
    def __init__(self, model, decay=0.998):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1.0 - self.decay)
            else:                       # int buffers (e.g. num_batches_tracked)
                self.shadow[k] = v.detach().clone().float()

    @torch.no_grad()
    def reset(self, model):
        """Re-seed the shadow from current weights (call after warmup so the
        random-init phase does not pollute the average)."""
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k in msd:
            msd[k].copy_(self.shadow[k].to(msd[k].dtype))


def _freeze_bn(model):
    """Put all BatchNorm layers in eval mode (use pretrained running stats, stop
    updating them). Standard small-batch transfer trick: with batch_size=6 the
    per-step BN statistics are noisy and hurt generalization."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


@torch.no_grad()
def adapt_bn_stats(model, samples, arrays, cfg, device, passes=3, momentum=0.1):
    """Test-time BatchNorm adaptation (transductive, LABEL-FREE). Forwards the
    held-out subject's clips in BN-train mode a few times so BN running stats
    drift from the train-subjects' distribution toward the test subject's,
    starting from the trained stats (momentum blend). Attacks LOSO subject-shift
    (45% of error concentrates in 3 subjects). Uses NO test labels — legitimate
    for a deployed app that calibrates to the user's own frames."""
    model.eval()                               # dropout off
    bns = [m for m in model.modules()
           if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))]
    old_mom = [m.momentum for m in bns]
    for m in bns:
        m.momentum = momentum
        m.train()                              # let BN update running stats only
    loader = make_loader(samples, arrays, cfg, train=False, shuffle=False)
    for _ in range(max(1, passes)):
        for x, _ in loader:
            x = move_to(x, device)
            with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
                model(x)
    for m, mom in zip(bns, old_mom):
        m.momentum = mom                       # restore; predict() sets eval -> uses adapted stats


class FocalLoss(nn.Module):
    """Multiclass focal loss with optional per-class weight (alpha).
    FL = weight[y] * (1-p_y)^gamma * (-log p_y). Down-weights easy samples,
    focusing on hard/rare classes (e.g. the confusable 'others')."""
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        # label_smoothing accepted for API symmetry but not applied (focal+LS rarely combined)

    def forward(self, logits, target):
        import torch.nn.functional as F
        logp = F.log_softmax(logits, dim=1)
        ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")
        pt = logp.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        return (((1.0 - pt) ** self.gamma) * ce).mean()


@torch.no_grad()
def predict(model, samples, arrays, cfg, device, tta=1):
    """Return averaged softmax probs (N, num_classes) over `tta` passes."""
    model.eval()
    probs = None
    for _ in range(max(1, tta)):
        # TTA: augment (train-mode dataset) but keep sample order (shuffle=False)
        loader = make_loader(samples, arrays, cfg, train=(tta > 1), shuffle=False)
        batch_probs = []
        for x, _ in loader:
            x = move_to(x, device)
            with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
                out = model(x)
            batch_probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        p = np.concatenate(batch_probs, axis=0)
        probs = p if probs is None else probs + p
    return probs / max(1, tta)


@torch.no_grad()
def evaluate_val(model, samples, arrays, cfg, device):
    """Single deterministic pass over val set. Returns (probs, val_loss, val_acc).
    val_loss uses plain CE (no class weight / no label smoothing) as a clean
    generalization signal for monitoring only (not used for model selection)."""
    model.eval()
    loader = make_loader(samples, arrays, cfg, train=False)
    ce = nn.CrossEntropyLoss(reduction="sum")
    probs, loss_sum, correct, tot = [], 0.0, 0, 0
    for x, y in loader:
        x = move_to(x, device)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
            out = model(x)
        out = out.float()
        loss_sum += ce(out, y).item()
        correct += (out.argmax(1) == y).sum().item()
        tot += batch_size_of(x)
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0), loss_sum / max(1, tot), correct / max(1, tot)


def train_fold(train_samples, val_samples, arrays, cfg, num_classes, device, log_fn):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build_model(cfg, num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    epochs = cfg["epochs"]
    # snapshot ensembling: cyclic cosine-with-restarts LR; each cycle converges to
    # a distinct good minimum -> several DECORRELATED-but-strong members from ONE
    # run (unlike plain multi-seed averaging, which regressed toward the mean here).
    snap_cycles = int(cfg.get("snapshot_cycles", 0))
    if snap_cycles > 1:
        cyc_len = max(1, epochs // snap_cycles)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cyc_len)
    else:
        cyc_len = None
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights(train_samples, num_classes, device) if cfg.get("class_weighting", True) else None
    if cfg.get("loss", "ce") == "focal":
        criterion = FocalLoss(weight=w, gamma=cfg.get("focal_gamma", 2.0),
                              label_smoothing=cfg.get("label_smoothing", 0.0))
    else:
        criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.get("label_smoothing", 0.0))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))

    train_loader = make_loader(train_samples, arrays, cfg, train=True)
    history = []
    eval_last_k = cfg.get("eval_last_k", 1)
    prob_accum = None
    n_accum = 0

    ema_decay = float(cfg.get("ema", 0.0))
    ema_start = int(cfg.get("ema_start_epoch", max(1, epochs // 5)))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    freeze_bn = bool(cfg.get("freeze_bn", False))

    for ep in range(epochs):
        model.train()
        if freeze_bn:
            _freeze_bn(model)
        tot, correct, loss_sum = 0, 0, 0.0
        mixup_a = cfg.get("mixup", 0.0)
        for x, y in train_loader:
            x = move_to(x, device)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            # mixup (single-stream tensor only) — regularizes small data
            do_mix = mixup_a > 0 and not isinstance(x, (list, tuple))
            if do_mix:
                lam = float(np.random.beta(mixup_a, mixup_a))
                perm = torch.randperm(x.size(0), device=device)
                x = lam * x + (1.0 - lam) * x[perm]
                y2 = y[perm]
            with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
                out = model(x)
                loss = (lam * criterion(out, y) + (1 - lam) * criterion(out, y2)) if do_mix \
                    else criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None and ep >= ema_start:
                ema.update(model)
            bs = batch_size_of(x)
            loss_sum += loss.item() * bs
            correct += (out.argmax(1) == y).sum().item()
            tot += bs
        sched.step()
        if ema is not None and ep == ema_start - 1:
            ema.reset(model)   # re-seed shadow at end of warmup (drop init phase)
        tr_loss = loss_sum / max(1, tot)
        tr_acc = correct / max(1, tot)

        # per-epoch val/test evaluation (monitoring trend; not used for selection)
        val_probs_ep, val_loss, val_acc = evaluate_val(model, val_samples, arrays, cfg, device)
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "lr": opt.param_groups[0]["lr"]})
        log_fn(f"      ep{ep:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
               f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

        # collect predictions: at each cycle end (snapshot mode) or last-k epochs
        collect = ((ep + 1) % cyc_len == 0) if snap_cycles > 1 else (ep >= epochs - eval_last_k)
        if collect:
            bn_adapt = bool(cfg.get("test_bn_adapt", False))
            # backup weights+buffers if we will mutate them (EMA swap and/or BN adapt)
            need_restore = (ema is not None) or bn_adapt
            backup = ({k: v.detach().clone() for k, v in model.state_dict().items()}
                      if need_restore else None)
            if ema is not None:
                ema.copy_to(model)             # evaluate EMA-averaged weights (flatter minimum)
            if bn_adapt:                        # label-free test-time BN adaptation
                adapt_bn_stats(model, val_samples, arrays, cfg, device,
                               passes=cfg.get("bn_adapt_passes", 3),
                               momentum=cfg.get("bn_adapt_momentum", 0.1))
            if cfg.get("tta", 1) > 1:
                p = predict(model, val_samples, arrays, cfg, device, tta=cfg["tta"])
            elif need_restore:
                p, _, _ = evaluate_val(model, val_samples, arrays, cfg, device)
            else:
                p = val_probs_ep  # reuse deterministic pass
            if need_restore:
                model.load_state_dict(backup)  # undo EMA/BN-adapt before continued training
            prob_accum = p if prob_accum is None else prob_accum + p
            n_accum += 1

    val_probs = prob_accum / max(1, n_accum)
    del model
    torch.cuda.empty_cache()
    return val_probs, history
