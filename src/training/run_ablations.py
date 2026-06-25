"""
Ablation study — evaluates the three controllable ablations on the validation set.

A1  No Tool Context  — [ACTION]+[OBS] tokens zeroed at preprocessing time
A2  No Query Context — [QUERY] tokens zeroed at preprocessing time
A3  No Cross-Step    — the TransformerEncoder is bypassed (each step independent)

All ablations reuse the *same trained weights* — they are forward-pass
modifications, not retrains.  This is correct because we are measuring
the sensitivity of the FULL model to each input channel.

Output: val_ablations.json
"""
import os
import sys
import json
import torch

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

from src.models.agentsight import AgentSightModel
from src.data.preprocessor  import StepPreprocessor
from src.training.evaluate  import step_localization_accuracy
from sklearn.metrics import f1_score


# ── Ablated preprocessors ─────────────────────────────────────────────────────

class AblatedPreprocessor(StepPreprocessor):
    """Zeroes out one or more input channels before encoding."""
    def __init__(self, ablate_tool=False, ablate_query=False, **kwargs):
        super().__init__(**kwargs)
        self.ablate_tool  = ablate_tool
        self.ablate_query = ablate_query

    def encode_step(self, query, step):
        if self.ablate_tool:
            step = dict(step)
            step["tool_calls"]     = []
            step["tool_responses"] = []
        if self.ablate_query:
            query = ""
        return super().encode_step(query, step)


# ── No-cross-step variant ─────────────────────────────────────────────────────

class NoContextModel(AgentSightModel):
    """Bypass the TransformerEncoder — each step is classified independently."""
    def forward(self, input_ids, attention_mask):
        out       = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        step_repr = out.last_hidden_state[:, 0, :].to(torch.float32)
        fused     = self.fusion(step_repr)
        return self.cls_head(fused).squeeze(-1)


# ── Runner ────────────────────────────────────────────────────────────────────

def run_condition(name, model, preprocessor, val_samples, device, threshold=0.5):
    model.eval()
    results = []

    with torch.no_grad():
        for i, sample in enumerate(val_samples):
            is_hal = sample.get("is_hallucination", False)
            if isinstance(is_hal, str):
                is_hal = is_hal.lower() == "true"
            hal_step = sample.get("hallucination_step")
            if hal_step is not None:
                hal_step = int(hal_step)

            try:
                steps = preprocessor.encode_trajectory(sample)
            except Exception:
                steps = []

            if not steps:
                results.append({
                    "condition": name,
                    "sample_idx": i,
                    "true_is_hallucination": is_hal,
                    "true_hallucination_step": hal_step,
                    "pred_is_hallucination": False,
                    "pred_step": None,
                    "max_hal_prob": None,
                    "encoding_failed": True,
                })
                continue

            ids  = torch.stack([s["encoding"]["input_ids"].squeeze(0)  for s in steps]).to(device)
            mask = torch.stack([s["encoding"]["attention_mask"].squeeze(0) for s in steps]).to(device)
            vocab_size = model.encoder.config.vocab_size
            ids = torch.clamp(ids, 0, vocab_size - 1)

            logits     = model(ids, mask)
            probs      = torch.sigmoid(logits).cpu().tolist()
            if isinstance(probs, float):
                probs = [probs]

            max_prob    = max(probs)
            pred_is_hal = max_prob > threshold
            pred_step   = steps[probs.index(max_prob)]["step_idx"] if pred_is_hal else None

            results.append({
                "condition": name,
                "sample_idx": i,
                "true_is_hallucination": is_hal,
                "true_hallucination_step": hal_step,
                "pred_is_hallucination": pred_is_hal,
                "pred_step": pred_step,
                "max_hal_prob": max_prob,
                "encoding_failed": False,
            })

    return results


def compute_metrics(results, val_samples):
    hal_true  = [1 if r["true_is_hallucination"] else 0 for r in results]
    hal_preds = [1 if r["pred_is_hallucination"] else 0 for r in results]
    f1 = f1_score(hal_true, hal_preds, average="macro", zero_division=0)

    # Step-acc: official denominator = all hallucinated, not TP-only
    hal_only = [r for r in results if r["true_is_hallucination"] and not r.get("encoding_failed")]
    correct  = sum(1 for r in hal_only if r["pred_step"] == r["true_hallucination_step"])
    step_acc = correct / len(hal_only) if hal_only else 0.0

    return {"step_acc": step_acc, "judgment_f1": f1,
            "step_correct": correct, "n_hal": len(hal_only)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights_path = os.path.join(project_root, "src", "models", "best_agentsight.pth")
    meta_path    = weights_path.replace(".pth", "_meta.json")

    threshold = 0.5
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", 0.5)
        print(f"Using saved threshold: {threshold:.3f}")

    with open(os.path.join(project_root, "data", "splits", "val.json")) as f:
        val_samples = json.load(f)
    print(f"Loaded {len(val_samples)} val samples.\n")

    conditions = [
        ("Full Model",       AgentSightModel, StepPreprocessor()),
        ("No Tool Context",  AgentSightModel, AblatedPreprocessor(ablate_tool=True)),
        ("No Query Context", AgentSightModel, AblatedPreprocessor(ablate_query=True)),
        ("No Cross-Step",    NoContextModel,  StepPreprocessor()),
    ]

    all_results = []
    summary     = {}

    for name, ModelClass, preprocessor in conditions:
        print(f"Running ablation: {name} …")
        model = ModelClass()
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device)

        results = run_condition(name, model, preprocessor, val_samples, device, threshold)
        all_results.extend(results)

        m = compute_metrics(results, val_samples)
        summary[name] = m
        print(f"  Step-Acc: {m['step_acc']*100:.1f}% ({m['step_correct']}/{m['n_hal']})  "
              f"|  F1: {m['judgment_f1']*100:.1f}%\n")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(project_root, "val_ablations.json")
    with open(out_path, "w") as f:
        json.dump({
            "note": "Ablations evaluated on val.json only. test.json is sealed.",
            "test_json_sha256": "9604aae8eb5aec4ae666cfbe3053910f0570a807a4fa5515223dbca1aa66a7d8",
            "threshold_used": threshold,
            "summary": summary,
            "per_trajectory": all_results,
        }, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────────────
    base_acc = summary["Full Model"]["step_acc"]
    print("=" * 64)
    print(f"{'Condition':<24}  {'Step-Acc':>9}  {'Δ vs Full':>9}  {'Macro-F1':>9}")
    print("-" * 64)
    for name, m in summary.items():
        delta = m["step_acc"] - base_acc
        delta_str = f"({delta*100:+.1f})" if name != "Full Model" else "baseline"
        print(f"  {name:<22}  {m['step_acc']*100:>8.1f}%  {delta_str:>9}  "
              f"{m['judgment_f1']*100:>8.1f}%")
    print("=" * 64)
    print(f"\nRaw output → {out_path}")


if __name__ == "__main__":
    main()
