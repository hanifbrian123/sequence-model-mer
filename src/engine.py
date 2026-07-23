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


def make_loader(samples, arrays, cfg, train, shuffle=None, view=None):
    """Build a loader with independent augmentation, ordering, and eval view.

    ``train`` enables stochastic training augmentation only. Deterministic TTA
    is expressed by ``view`` while keeping ``train=False`` so random erase,
    color jitter, random crop, and random temporal jitter cannot leak into eval.
    """
    if shuffle is None:
        shuffle = train
    if cfg.get("modality") == "two_stream":
        arrays_a, arrays_b = arrays   # tuple (flow, appearance)
        ds = TwoStreamDataset(samples, arrays_a, arrays_b, cfg, train, view=view)
    else:
        ds = SeqDataset(samples, arrays, cfg, train, view=view)
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


def build_criterion(samples, cfg, num_classes, device):
    """Build the configured loss for both LOSO and train-on-all deployment."""
    weight = class_weights(samples, num_classes, device) \
        if cfg.get("class_weighting", True) else None
    if cfg.get("loss", "ce") == "focal":
        return FocalLoss(weight=weight, gamma=cfg.get("focal_gamma", 2.0),
                         label_smoothing=cfg.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(weight=weight,
                               label_smoothing=cfg.get("label_smoothing", 0.0))


def configure_training(model, cfg):
    """Create optimizer/scheduler and adaptive-finetuning state once.

    Keeping this shared prevents train_final from silently changing the recipe
    used during LOSO (the previous implementation always used plain CE/AdamW).
    """
    epochs = int(cfg["epochs"])
    is_vivit = cfg.get("backbone") == "vivit"
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    freeze_epochs = int(cfg.get("freeze_epochs", 0))
    if is_vivit:
        model.set_backbone_requires_grad(True)
        param_groups = model.llrd_param_groups(
            base_lr=cfg["lr"], head_lr=cfg.get("head_lr", cfg["lr"]),
            decay=cfg.get("llrd_decay", 0.75), weight_decay=cfg["weight_decay"])
        optimizer = torch.optim.AdamW(param_groups)
        if freeze_epochs > 0:
            model.set_backbone_requires_grad(False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                      weight_decay=cfg["weight_decay"])

    snapshot_cycles = int(cfg.get("snapshot_cycles", 0))
    if snapshot_cycles > 1:
        cycle_length = max(1, epochs // snapshot_cycles)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cycle_length)
    elif warmup_epochs > 0:
        cycle_length = None
        warm = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs - warmup_epochs))
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warm, cosine], milestones=[warmup_epochs])
    else:
        cycle_length = None
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs)
    return {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "is_vivit": is_vivit,
        "freeze_epochs": freeze_epochs,
        "snapshot_cycles": snapshot_cycles,
        "cycle_length": cycle_length,
    }


def train_one_epoch(model, loader, criterion, scaler, cfg, device, epoch,
                    training, ema=None, ema_start=0, log_fn=None):
    """Shared training epoch used by LOSO and final artifact training."""
    if training["is_vivit"] and training["freeze_epochs"] > 0 \
            and epoch == training["freeze_epochs"]:
        model.set_backbone_requires_grad(True)
        if log_fn is not None:
            log_fn(f"      [unfreeze] backbone unfrozen at epoch {epoch}")
    model.train()
    if cfg.get("freeze_bn", False):
        _freeze_bn(model)

    optimizer = training["optimizer"]
    accum = max(1, int(cfg.get("grad_accum", 1)))
    mixup_alpha = float(cfg.get("mixup", 0.0))
    total = correct = 0
    loss_sum = 0.0
    num_batches = len(loader)
    optimizer.zero_grad()
    for batch_index, (x, y) in enumerate(loader):
        x = move_to(x, device)
        y = y.to(device, non_blocking=True)
        do_mixup = mixup_alpha > 0 and not isinstance(x, (list, tuple))
        if do_mixup:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            permutation = torch.randperm(x.size(0), device=device)
            x = lam * x + (1.0 - lam) * x[permutation]
            y_second = y[permutation]
        with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
            output = model(x)
            loss = (lam * criterion(output, y)
                    + (1.0 - lam) * criterion(output, y_second)) \
                if do_mixup else criterion(output, y)
        scaler.scale(loss / accum).backward()
        if (batch_index + 1) % accum == 0 or (batch_index + 1) == num_batches:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema is not None and epoch >= ema_start:
                ema.update(model)
        batch_size = batch_size_of(x)
        loss_sum += loss.item() * batch_size
        correct += (output.argmax(1) == y).sum().item()
        total += batch_size
    return loss_sum / max(1, total), correct / max(1, total)


def deterministic_tta_views(cfg, tta=1):
    """Return fixed, reproducible eval views.

    Configs may provide an explicit ``tta_views`` list. Otherwise the legacy
    integer TTA count selects from a fixed bank of spatial/temporal views. This
    makes reruns comparable and, crucially, keeps evaluation out of training
    augmentation mode.
    """
    requested = max(1, int(tta))
    explicit = cfg.get("tta_views")
    if explicit is not None:
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("tta_views must be a non-empty list of view dicts")
        if not all(isinstance(v, dict) for v in explicit):
            raise ValueError("every tta_views entry must be a dict")
        return [dict(v) for v in explicit[:requested]]
    bank = [
        {"crop": "center", "flip": False, "temporal_phase": 0.50},
        {"crop": "center", "flip": True,  "temporal_phase": 0.50},
        {"crop": "top_left", "flip": False, "temporal_phase": 0.25},
        {"crop": "top_right", "flip": True, "temporal_phase": 0.25},
        {"crop": "bottom_left", "flip": False, "temporal_phase": 0.75},
        {"crop": "bottom_right", "flip": True, "temporal_phase": 0.75},
        {"crop": "top_left", "flip": True, "temporal_phase": 0.75},
        {"crop": "top_right", "flip": False, "temporal_phase": 0.75},
        {"crop": "bottom_left", "flip": True, "temporal_phase": 0.25},
        {"crop": "bottom_right", "flip": False, "temporal_phase": 0.25},
    ]
    if requested <= len(bank):
        return bank[:requested]
    return [dict(bank[i % len(bank)]) for i in range(requested)]


@torch.no_grad()
def predict(model, samples, arrays, cfg, device, tta=1):
    """Return averaged softmax probs over deterministic TTA views."""
    model.eval()
    probs = None
    views = deterministic_tta_views(cfg, tta)
    for view in views:
        loader = make_loader(samples, arrays, cfg, train=False, shuffle=False, view=view)
        batch_probs = []
        for x, _ in loader:
            x = move_to(x, device)
            with torch.autocast(device_type="cuda", enabled=cfg.get("amp", True)):
                out = model(x)
            batch_probs.append(torch.softmax(out.float(), dim=1).cpu().numpy())
        p = np.concatenate(batch_probs, axis=0)
        probs = p if probs is None else probs + p
    return probs / len(views)


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


def train_fold(train_samples, val_samples, arrays, cfg, num_classes, device, log_fn,
               snapshot_fn=None):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    model = build_model(cfg, num_classes).to(device)

    epochs = cfg["epochs"]
    training = configure_training(model, cfg)
    opt = training["optimizer"]
    sched = training["scheduler"]
    snap_cycles = training["snapshot_cycles"]
    cyc_len = training["cycle_length"]
    criterion = build_criterion(train_samples, cfg, num_classes, device)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.get("amp", True))

    train_loader = make_loader(train_samples, arrays, cfg, train=True)
    history = []
    eval_last_k = cfg.get("eval_last_k", 1)
    prob_accum = None
    n_accum = 0
    monitor_val = bool(cfg.get("monitor_val", True))

    ema_decay = float(cfg.get("ema", 0.0))
    ema_start = int(cfg.get("ema_start_epoch", max(1, epochs // 5)))
    ema = EMA(model, ema_decay) if ema_decay > 0 else None
    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, scaler, cfg, device, ep, training,
            ema=ema, ema_start=ema_start, log_fn=log_fn)
        sched.step()
        if ema is not None and ep == ema_start - 1:
            ema.reset(model)   # re-seed shadow at end of warmup (drop init phase)
        # Validation monitoring is useful inside the development protocol, but
        # disabled for a locked audit split so its labels are not repeatedly
        # exposed while training. Snapshot epochs remain fixed by config.
        if monitor_val:
            val_probs_ep, val_loss, val_acc = evaluate_val(
                model, val_samples, arrays, cfg, device)
        else:
            val_probs_ep, val_loss, val_acc = None, None, None
        history.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc,
                        "val_loss": val_loss, "val_acc": val_acc,
                        "lr": opt.param_groups[0]["lr"]})
        if monitor_val:
            log_fn(f"      ep{ep:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
                   f"| val_loss={val_loss:.4f} val_acc={val_acc:.3f}")
        else:
            log_fn(f"      ep{ep:02d} train_loss={tr_loss:.4f} train_acc={tr_acc:.3f}")

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
            elif need_restore or val_probs_ep is None:
                p, _, _ = evaluate_val(model, val_samples, arrays, cfg, device)
            else:
                p = val_probs_ep  # reuse deterministic pass
            if snapshot_fn is not None:
                snapshot_fn(ep, {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                })
            if need_restore:
                model.load_state_dict(backup)  # undo EMA/BN-adapt before continued training
            prob_accum = p if prob_accum is None else prob_accum + p
            n_accum += 1

    val_probs = prob_accum / max(1, n_accum)
    del model
    torch.cuda.empty_cache()
    return val_probs, history
