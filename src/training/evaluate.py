import torch
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score


# ─────────────────────────────────────────────────────────────────────────────
# Official metric definitions — match AgentHallu paper (Equations 6-11)
# ─────────────────────────────────────────────────────────────────────────────

def step_localization_accuracy(samples, step_preds):
    """
    AgentHallu Eq. 11:
      Acc_step = |{i ∈ Hhal : pred_step(i) == true_step(i)}| / |Hhal|

    Denominator is ALL hallucinated trajectories (regardless of whether
    the model's judgment prediction was correct).  Do NOT restrict to
    TP-only — that would be a lenient variant not comparable to the paper.
    """
    correct = total = 0
    for s, pred_step in zip(samples, step_preds):
        is_hal = s.get("is_hallucination", False)
        if isinstance(is_hal, str):
            is_hal = is_hal.lower() == "true"
        if not is_hal:
            continue
        total += 1
        true_step = s.get("hallucination_step")
        if true_step is not None:
            true_step = int(true_step)
        if pred_step == true_step:
            correct += 1
    return correct / total if total else 0.0


def tune_threshold(model, val_samples, preprocessor, thresholds=None):
    """
    Find the decision threshold on the validation set that maximises
    macro-F1.  Returns the best threshold and the corresponding metrics.
    """
    if thresholds is None:
        thresholds = np.arange(0.2, 0.8, 0.02)

    model.eval()
    device = next(model.parameters()).device

    # Collect max-prob per trajectory
    max_probs = []
    hal_true  = []

    with torch.no_grad():
        for sample in val_samples:
            is_hal = sample.get("is_hallucination", False)
            if isinstance(is_hal, str):
                is_hal = is_hal.lower() == "true"
            hal_true.append(int(is_hal))

            try:
                steps = preprocessor.encode_trajectory(sample)
            except Exception:
                steps = []

            if not steps:
                max_probs.append(0.0)
                continue

            ids  = torch.stack([s["encoding"]["input_ids"].squeeze(0)  for s in steps]).to(device)
            mask = torch.stack([s["encoding"]["attention_mask"].squeeze(0) for s in steps]).to(device)
            vocab_size = model.encoder.config.vocab_size
            ids = torch.clamp(ids, 0, vocab_size - 1)

            logits = model(ids, mask)
            probs  = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]
            max_probs.append(max(probs))

    best_f1 = -1.0
    best_thr = 0.5
    for thr in thresholds:
        preds = [1 if p > thr else 0 for p in max_probs]
        f1 = f1_score(hal_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_thr = float(thr)

    return best_thr, best_f1


def evaluate(model, test_samples, preprocessor, threshold=0.5):
    """
    Full evaluation: judgment (binary) + step localisation.
    Uses the supplied threshold (tune it on val, never on test).
    """
    model.eval()
    device = next(model.parameters()).device

    hal_preds  = []
    hal_true   = []
    step_preds = []

    with torch.no_grad():
        for sample in test_samples:
            is_hal = sample.get("is_hallucination", False)
            if isinstance(is_hal, str):
                is_hal = is_hal.lower() == "true"
            hal_true.append(int(is_hal))

            try:
                steps = preprocessor.encode_trajectory(sample)
            except Exception:
                steps = []

            if not steps:
                hal_preds.append(0)
                step_preds.append(None)
                continue

            ids  = torch.stack([s["encoding"]["input_ids"].squeeze(0)  for s in steps]).to(device)
            mask = torch.stack([s["encoding"]["attention_mask"].squeeze(0) for s in steps]).to(device)
            vocab_size = model.encoder.config.vocab_size
            ids = torch.clamp(ids, 0, vocab_size - 1)

            logits = model(ids, mask)
            probs  = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]

            max_prob     = max(probs)
            max_prob_idx = probs.index(max_prob)

            pred_is_hal = max_prob > threshold
            hal_preds.append(int(pred_is_hal))

            pred_step = steps[max_prob_idx]["step_idx"] if pred_is_hal else None
            step_preds.append(pred_step)

    step_acc = step_localization_accuracy(test_samples, step_preds)
    macro_f1 = f1_score(hal_true, hal_preds, average="macro",    zero_division=0)
    macro_rec = recall_score(hal_true, hal_preds, average="macro", zero_division=0)
    macro_pre = precision_score(hal_true, hal_preds, average="macro", zero_division=0)

    return {
        "step_acc":         step_acc,
        "judgment_f1":      macro_f1,
        "judgment_recall":  macro_rec,
        "judgment_precision": macro_pre,
        "threshold":        threshold,
        "n_samples":        len(test_samples),
    }
