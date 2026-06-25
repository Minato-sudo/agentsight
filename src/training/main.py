"""
AgentSight — full training entry point.

Key design decisions
────────────────────
* AdamW with cosine-annealing LR schedule (linear warmup for 10% of steps).
* Gradient accumulation over grad_accum_steps=4 trajectories (effective batch ~4).
* pos_weight computed dynamically from the training split ratio.
* WeightedRandomSampler ensures each epoch sees balanced class distribution.
* Decision threshold tuned on val set after training (never on test).
* Best model selected by val step_localization_accuracy (primary metric).
* Test set NEVER evaluated during training — locked behind hash verification.

TEST SET INTEGRITY SEAL — DO NOT MODIFY
  sha256sum data/splits/test.json
  9604aae8eb5aec4ae666cfbe3053910f0570a807a4fa5515223dbca1aa66a7d8
test.json is LOCKED until a final deliberate single run for the paper.
"""
import os
import sys
import json
import argparse

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.preprocessor import StepPreprocessor
from src.data.dataset       import get_dataloader, AgentTrajectoryDataset
from src.models.agentsight  import AgentSightModel
from src.training.train     import train_epoch, compute_pos_weight
from src.training.evaluate  import evaluate, tune_threshold


def main():
    parser = argparse.ArgumentParser(description="Train AgentSight Hallucination Detector")
    parser.add_argument("--epochs",           type=int,   default=50,   help="Maximum training epochs")
    parser.add_argument("--lr",               type=float, default=3e-5, help="Peak learning rate for AdamW")
    parser.add_argument("--grad_accum",       type=int,   default=4,    help="Gradient accumulation steps")
    parser.add_argument("--patience",         type=int,   default=15,   help="Early stopping patience (epochs)")
    parser.add_argument("--max_len",          type=int,   default=512,  help="Tokenizer max sequence length")
    parser.add_argument("--max_steps",        type=int,   default=20,   help="Max trajectory steps (centred truncation)")
    parser.add_argument("--weight_decay",     type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup_ratio",     type=float, default=0.10, help="Fraction of steps for LR warmup")
    parser.add_argument("--no_weighted_sampler", action="store_true",  help="Disable WeightedRandomSampler")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits_dir = os.path.join(project_root, "data", "splits")

    # ── Preprocessor & data ───────────────────────────────────────────────────
    print("Loading tokenizer …")
    preprocessor = StepPreprocessor(max_len=args.max_len)

    use_sampler = not args.no_weighted_sampler
    print(f"Building dataloaders (WeightedRandomSampler={use_sampler}) …")
    train_loader = get_dataloader(
        os.path.join(splits_dir, "train.json"),
        preprocessor,
        batch_size=1,
        shuffle=True,
        use_weighted_sampler=use_sampler,
    )
    val_loader = get_dataloader(
        os.path.join(splits_dir, "val.json"),
        preprocessor,
        batch_size=1,
        shuffle=False,
    )

    with open(os.path.join(splits_dir, "val.json")) as f:
        val_samples = json.load(f)

    print("Initialising AgentSightModel …")
    model = AgentSightModel()
    model.to(device)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    # Separate LoRA params from the rest to avoid weight-decaying bias/LN terms
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(param_groups, lr=args.lr)

    total_updates = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps  = int(total_updates * args.warmup_ratio)

    # Linear warmup then cosine decay implemented manually via LambdaLR
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_updates - warmup_steps)
        import math
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Dynamic pos_weight ────────────────────────────────────────────────────
    print("Computing class weights from training data …")
    pos_weight = compute_pos_weight(train_loader, device)
    print(f"  pos_weight = {pos_weight.item():.2f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    weights_path       = os.path.join(project_root, "src", "models", "best_agentsight.pth")
    best_val_step_acc  = 0.0
    best_threshold     = 0.5
    epochs_no_improve  = 0

    print(f"\nStarting training for up to {args.epochs} epochs "
          f"(patience={args.patience}) …\n")

    for epoch in range(1, args.epochs + 1):
        print(f"── Epoch {epoch}/{args.epochs} ──────────────────────────────")

        avg_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            pos_weight, grad_accum_steps=args.grad_accum,
        )
        print(f"  Train loss : {avg_loss:.4f}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Tune threshold on val every 5 epochs (cheap) or when we might improve
        if epoch % 5 == 0 or epoch <= 5:
            thr, thr_f1 = tune_threshold(model, val_samples, preprocessor)
            print(f"  Val threshold tuning → thr={thr:.2f}  macro-F1={thr_f1*100:.1f}%")
        else:
            thr = best_threshold

        metrics = evaluate(model, val_samples, preprocessor, threshold=thr)
        print(
            f"  Val step-acc : {metrics['step_acc']*100:.1f}%  |  "
            f"F1 : {metrics['judgment_f1']*100:.1f}%  |  "
            f"Recall : {metrics['judgment_recall']*100:.1f}%  |  "
            f"Precision : {metrics['judgment_precision']*100:.1f}%"
        )

        if metrics["step_acc"] > best_val_step_acc:
            best_val_step_acc = metrics["step_acc"]
            best_threshold    = thr
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
            # Save the best threshold alongside weights so run_test.py can load it
            meta = {"threshold": best_threshold, "val_step_acc": best_val_step_acc,
                    "val_f1": metrics["judgment_f1"], "epoch": epoch}
            with open(weights_path.replace(".pth", "_meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            print(f"  [✓] New best model saved  (step-acc={best_val_step_acc*100:.1f}%,  "
                  f"thr={best_threshold:.2f})")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve}/{args.patience} epochs.")
            if epochs_no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    print(f"\n── Training complete ──────────────────────────────────────────")
    print(f"Best val step-acc : {best_val_step_acc*100:.1f}%  at threshold {best_threshold:.2f}")
    print(f"Weights saved to  : {weights_path}")


if __name__ == "__main__":
    main()
