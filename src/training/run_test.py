"""
Final test-set evaluation.

RULES
─────
1. Run this script EXACTLY ONCE — the run that produces the numbers for the paper.
2. Before running, verify the SHA-256 of data/splits/test.json matches the seal below.
3. Load the best threshold from best_agentsight_meta.json — do NOT re-tune on test.

SHA-256 seal for data/splits/test.json:
  9604aae8eb5aec4ae666cfbe3053910f0570a807a4fa5515223dbca1aa66a7d8
"""
import os
import sys
import json
import hashlib
import torch

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

from src.models.agentsight import AgentSightModel
from src.data.preprocessor  import StepPreprocessor
from src.training.evaluate  import evaluate, step_localization_accuracy

TEST_SHA256 = "9604aae8eb5aec4ae666cfbe3053910f0570a807a4fa5515223dbca1aa66a7d8"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_test_predictions():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. Hash verification (integrity seal) ────────────────────────────────
    test_file = os.path.join(project_root, "data", "splits", "test.json")
    actual_hash = sha256_file(test_file)
    if actual_hash != TEST_SHA256:
        raise RuntimeError(
            f"TEST SET HASH MISMATCH!\n"
            f"  Expected : {TEST_SHA256}\n"
            f"  Got      : {actual_hash}\n"
            "The test file has been modified — this run is invalid."
        )
    print(f"[✓] Test set hash verified: {actual_hash[:16]}…")

    # ── 2. Load model ─────────────────────────────────────────────────────────
    weights_path = os.path.join(project_root, "src", "models", "best_agentsight.pth")
    meta_path    = weights_path.replace(".pth", "_meta.json")

    threshold = 0.5   # fallback
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", 0.5)
        print(f"Loaded threshold from meta: {threshold:.3f}  "
              f"(val step-acc={meta.get('val_step_acc',0)*100:.1f}%, "
              f"val F1={meta.get('val_f1',0)*100:.1f}%)")
    else:
        print(f"Warning: no _meta.json found — using default threshold {threshold}")

    preprocessor = StepPreprocessor()
    model        = AgentSightModel()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # ── 3. Load test data ─────────────────────────────────────────────────────
    with open(test_file) as f:
        test_samples = json.load(f)
    print(f"Loaded {len(test_samples)} test trajectories.")

    # ── 4. Per-trajectory predictions ────────────────────────────────────────
    predictions_dump = []
    with torch.no_grad():
        for sample in test_samples:
            is_hal_true = sample.get("is_hallucination", False)
            if isinstance(is_hal_true, str):
                is_hal_true = is_hal_true.lower() == "true"
            true_step = sample.get("hallucination_step")
            if true_step is not None and is_hal_true:
                true_step = int(true_step)
            else:
                true_step = None

            try:
                steps_enc = preprocessor.encode_trajectory(sample)
            except Exception:
                steps_enc = []

            if not steps_enc:
                predictions_dump.append({
                    "trajectory_id":         sample.get("model_id", "unknown"),
                    "domain":                sample.get("question_domain", "unknown"),
                    "true_is_hallucination": is_hal_true,
                    "true_hallucination_step": true_step,
                    "pred_is_hallucination": False,
                    "pred_hallucination_step": None,
                    "max_probability":       0.0,
                    "encoding_failed":       True,
                })
                continue

            ids  = torch.stack([s["encoding"]["input_ids"].squeeze(0)  for s in steps_enc]).to(device)
            mask = torch.stack([s["encoding"]["attention_mask"].squeeze(0) for s in steps_enc]).to(device)
            vocab_size = model.encoder.config.vocab_size
            ids = torch.clamp(ids, 0, vocab_size - 1)

            logits = model(ids, mask)
            probs  = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]

            max_prob     = max(probs)
            max_idx      = probs.index(max_prob)
            pred_is_hal  = max_prob > threshold
            pred_step    = steps_enc[max_idx]["step_idx"] if pred_is_hal else None

            predictions_dump.append({
                "trajectory_id":           sample.get("model_id", "unknown"),
                "domain":                  sample.get("question_domain", "unknown"),
                "true_is_hallucination":   is_hal_true,
                "true_hallucination_step": true_step,
                "pred_is_hallucination":   pred_is_hal,
                "pred_hallucination_step": pred_step,
                "max_probability":         max_prob,
                "encoding_failed":         False,
            })

    out_file = os.path.join(project_root, "test_predictions.json")
    with open(out_file, "w") as f:
        json.dump(predictions_dump, f, indent=4)
    print(f"Saved detailed predictions → {out_file}")

    # ── 5. Formal evaluation ──────────────────────────────────────────────────
    metrics = evaluate(model, test_samples, preprocessor, threshold=threshold)

    print("\n" + "=" * 58)
    print("  FINAL TEST SET METRICS  (AgentHallu benchmark)")
    print("=" * 58)
    print(f"  Step Localization Acc  : {metrics['step_acc']*100:.1f}%")
    print(f"  Judgment Macro-F1      : {metrics['judgment_f1']*100:.1f}%")
    print(f"  Judgment Precision     : {metrics['judgment_precision']*100:.1f}%")
    print(f"  Judgment Recall        : {metrics['judgment_recall']*100:.1f}%")
    print(f"  Decision Threshold     : {threshold:.3f}")
    print(f"  N test samples         : {metrics['n_samples']}")
    print("=" * 58)

    print("\n  Reference baselines (AgentHallu paper):")
    print("    Random            — F1: 48.5%,  Step-Acc:  8.7%")
    print("    Open-source avg   — F1: 45.8%,  Step-Acc: 10.9%")
    print("    Gemini-2.5-Pro    — F1: 64.6%,  Step-Acc: 41.1%")
    print("    GPT-5             — F1: 70.2%,  Step-Acc:   n/a")
    print("=" * 58)


if __name__ == "__main__":
    generate_test_predictions()
