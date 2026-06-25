import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import re
import os
import json


REFERENCE_BASELINES = {
    "Random":         {"f1": 48.5, "step_acc": 8.7,  "color": "gray",   "ls": ":"},
    "Open-src avg":   {"f1": 45.8, "step_acc": 10.9, "color": "orange", "ls": "-."},
    "Gemini-2.5-Pro": {"f1": 64.6, "step_acc": 41.1, "color": "purple", "ls": "--"},
}


def parse_log(log_path):
    """Parse training log into lists of (epoch, train_loss, val_f1, val_step_acc)."""
    with open(log_path) as f:
        content = f.read()

    epochs      = []
    train_losses = []
    val_f1s     = []
    val_accs    = []

    blocks = re.split(r'── Epoch (\d+)/\d+ ──+', content)
    # blocks[0] = preamble, then alternating: epoch_num, block_content
    for i in range(1, len(blocks) - 1, 2):
        ep = int(blocks[i])
        blk = blocks[i + 1]

        loss_m = re.search(r'Train loss\s*:\s*([0-9.]+)', blk)
        f1_m   = re.search(r'F1\s*:\s*([0-9.]+)%', blk)
        acc_m  = re.search(r'step-acc\s*:\s*([0-9.]+)%', blk, re.IGNORECASE)

        if loss_m and f1_m and acc_m:
            epochs.append(ep)
            train_losses.append(float(loss_m.group(1)))
            val_f1s.append(float(f1_m.group(1)))
            val_accs.append(float(acc_m.group(1)))

    return epochs, train_losses, val_f1s, val_accs


def plot_learning_curves(log_path, output_path, meta_path=None):
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return

    epochs, train_losses, val_f1s, val_accs = parse_log(log_path)
    if not epochs:
        print("Could not parse any epoch data from the log.")
        return

    best_epoch = None
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        best_epoch = meta.get("epoch")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("AgentSight — Training Curves", fontsize=14, fontweight="bold")

    # ── Top panel: training loss ──────────────────────────────────────────────
    ax1.plot(epochs, train_losses, color="tab:red", marker="o", ms=3,
             label="Train loss (BCE)")
    ax1.set_ylabel("Train Loss", fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    if best_epoch:
        ax1.axvline(best_epoch, color="green", lw=1.5, ls="--", label=f"Best epoch ({best_epoch})")
    ax1.legend(fontsize=9)

    # ── Bottom panel: val metrics + baselines ─────────────────────────────────
    ax2.plot(epochs, val_accs, color="tab:blue", marker="s", ms=3,
             label="Val Step-Acc (%)")
    ax2.plot(epochs, val_f1s,  color="tab:green", marker="^", ms=3, ls="--",
             label="Val Macro-F1 (%)")

    for name, ref in REFERENCE_BASELINES.items():
        ax2.axhline(ref["step_acc"], color=ref["color"], ls=ref["ls"], lw=1.5,
                    label=f"{name} Step-Acc ({ref['step_acc']}%)")

    if best_epoch:
        ax2.axvline(best_epoch, color="green", lw=1.5, ls="--")

    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Metric (%)", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved → {output_path}")


if __name__ == "__main__":
    root     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    log_file = os.path.join(root, "full_training_log.txt")
    meta     = os.path.join(root, "src", "models", "best_agentsight_meta.json")
    out_png  = os.path.join(root, "notebooks", "training_curve.png")
    plot_learning_curves(log_file, out_png, meta_path=meta)
