import torch
import torch.nn as nn


def compute_pos_weight(loader, device):
    """
    Walk the training data once to compute the empirical positive-class weight.
    pos_weight = n_negative / n_positive (per BCEWithLogitsLoss convention).
    Falls back to 10.0 if the data gives extreme values.
    """
    n_pos = n_neg = 0
    for batch in loader:
        labels = batch["hal_label"].view(-1)
        n_pos += (labels > 0.5).sum().item()
        n_neg += (labels < 0.5).sum().item()
    if n_pos == 0:
        return torch.tensor([10.0]).to(device)
    weight = n_neg / n_pos
    # Clamp to [2, 20] — avoids degenerate regimes
    weight = max(2.0, min(20.0, weight))
    return torch.tensor([weight]).to(device)


def train_epoch(model, loader, optimizer, scheduler, pos_weight, grad_accum_steps=4):
    """
    Train for one epoch.

    Args
    ────
    model            : AgentSightModel
    loader           : DataLoader (batch_size=1, one trajectory per item)
    optimizer        : AdamW
    scheduler        : LR scheduler (stepped once per *gradient update*, not per batch)
    pos_weight       : tensor([float]) for BCEWithLogitsLoss
    grad_accum_steps : accumulate gradients over N trajectories before stepping
                       (effective batch size = grad_accum_steps)
    """
    model.train()
    device = next(model.parameters()).device

    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    total_loss = 0.0
    optimizer.zero_grad()

    for step_idx, batch in enumerate(loader):
        input_ids      = batch["input_ids"].squeeze(0).to(device)
        attention_mask = batch["attention_mask"].squeeze(0).to(device)
        hal_labels     = batch["hal_label"].squeeze(0).float().to(device)

        # Guard against out-of-vocabulary tokens (rare but seen in practice)
        vocab_size = model.encoder.config.vocab_size
        input_ids  = torch.clamp(input_ids, 0, vocab_size - 1)

        hal_logits = model(input_ids, attention_mask)

        loss = bce(hal_logits, hal_labels) / grad_accum_steps
        loss.backward()

        total_loss += loss.item() * grad_accum_steps   # un-scale for logging

        # Gradient update every grad_accum_steps trajectories
        if (step_idx + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

    # Flush remaining gradients
    if len(loader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader) if loader else 0.0
