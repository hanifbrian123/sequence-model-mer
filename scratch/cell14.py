def create_fold_model():
    """Membuat instance CNNTemporalViViT baru di setiap fold LOSO."""
    model = CNNTemporalViViT(
        feature_dim=feat_dim,
        num_classes=NUM_CLASSES,
        num_frames=NUM_FRAMES,
        d_model=512,
        nhead=8,
        num_layers=4,
        dropout=DROPOUT,
    ).to(DEVICE)
    return model

def train_one_epoch(model, dataloader, criterion, optimizer, scaler=None, grad_accum_steps=1, use_amp=False):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad()

    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if use_amp and scaler is not None:
            with autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                outputs = model(inputs)
                loss = criterion(outputs, labels) / grad_accum_steps
            scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels) / grad_accum_steps
            loss.backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps
        all_preds.extend(outputs.argmax(dim=1).detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = running_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1

@torch.no_grad()
def evaluate_subject(model, dataloader, criterion, use_amp=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for inputs, labels in dataloader:
        inputs = inputs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda" if "cuda" in str(DEVICE) else "cpu"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        running_loss += loss.item()
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = outputs.argmax(dim=1).cpu().numpy()

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(len(dataloader), 1)
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1, all_preds, all_labels, all_probs

print("✅ Model Factory & Training Functions Siap!")
