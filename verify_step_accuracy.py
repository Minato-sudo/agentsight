import argparse
import json
from collections import Counter

AGENTHALLU_REFERENCE = {
    "random_judgment_f1": 48.5,
    "open_source_avg_judgment_f1": 45.8,
    "gemini_judgment_f1": 64.6,
    "gpt5_judgment_f1": 70.2,
    "random_step_acc": 8.7,
    "open_source_avg_step_acc": 10.9,
    "gemini_step_acc_sota": 41.1,
}


def load(path):
    with open(path) as f:
        return json.load(f)


def check_id_field(data):
    print("=== trajectory_id sanity check ===")
    ids = Counter(d.get("trajectory_id") for d in data)
    if len(ids) < len(data) * 0.5:
        print(
            f"  WARNING: only {len(ids)} unique values across {len(data)} "
            f"entries. This does not look like a unique per-trajectory ID — "
            f"check whether it is actually the backbone model name."
        )
        print(f"  Most common values: {ids.most_common(5)}")
    else:
        print(f"  OK: {len(ids)} unique values across {len(data)} entries.")
    print()


def judgment_metrics(data):
    tp = tn = fp = fn = 0
    for d in data:
        t, p = d["true_is_hallucination"], d["pred_is_hallucination"]
        if t and p:
            tp += 1
        elif (not t) and (not p):
            tn += 1
        elif (not t) and p:
            fp += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1_pos = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    prec_neg = tn / (tn + fn) if (tn + fn) else 0.0
    rec_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_neg = 2 * prec_neg * rec_neg / (prec_neg + rec_neg) if (prec_neg + rec_neg) else 0.0

    macro_f1 = (f1_pos + f1_neg) / 2
    acc = (tp + tn) / len(data)

    print("=== Judgment metrics (AgentHallu Eq. 6-10) ===")
    print(f"  Confusion matrix: TP={tp} FP={fp} TN={tn} FN={fn}")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(
        f"  Macro-F1:  {macro_f1*100:.1f}%   (reference -- random="
        f"{AGENTHALLU_REFERENCE['random_judgment_f1']}, open-source avg="
        f"{AGENTHALLU_REFERENCE['open_source_avg_judgment_f1']}, Gemini="
        f"{AGENTHALLU_REFERENCE['gemini_judgment_f1']}, GPT-5="
        f"{AGENTHALLU_REFERENCE['gpt5_judgment_f1']})"
    )
    print()


def step_localization_metrics(data):
    hal = [d for d in data if d["true_is_hallucination"]]
    n = len(hal)
    if n == 0:
        print("=== Step localization accuracy ===\n  No hallucinated samples found.\n")
        return

    official = sum(
        1 for d in hal if d["pred_hallucination_step"] == d["true_hallucination_step"]
    )
    tp_subset = [d for d in hal if d["pred_is_hallucination"]]
    tp_only = sum(
        1 for d in tp_subset if d["pred_hallucination_step"] == d["true_hallucination_step"]
    )
    tolerance = sum(
        1
        for d in hal
        if d["pred_hallucination_step"] is not None
        and abs(d["pred_hallucination_step"] - d["true_hallucination_step"]) <= 1
    )

    print("=== Step localization accuracy ===")
    print(f"  Total true-hallucinated samples (|Hhal|): {n}")
    print()
    print(
        f"  OFFICIAL definition (denominator = all {n} hallucinated samples, "
        f"regardless of judgment correctness) -- use THIS for any SOTA comparison:"
    )
    print(f"    {official}/{n} = {official/n*100:.1f}%")
    print()
    print("  Debugging variants only -- NOT valid for comparison against the paper:")
    if tp_subset:
        print(
            f"    TP-only (denom = {len(tp_subset)} correctly-flagged cases): "
            f"{tp_only}/{len(tp_subset)} = {tp_only/len(tp_subset)*100:.1f}%"
        )
    print(f"    +/-1 step tolerance: {tolerance}/{n} = {tolerance/n*100:.1f}%")
    print()
    print(
        f"  Reference -- random={AGENTHALLU_REFERENCE['random_step_acc']}, "
        f"open-source avg={AGENTHALLU_REFERENCE['open_source_avg_step_acc']}, "
        f"Gemini-2.5-Pro SOTA={AGENTHALLU_REFERENCE['gemini_step_acc_sota']}"
    )
    print()


def position_bias_check(data):
    hal = [d for d in data if d["true_is_hallucination"]]
    if not hal:
        return
    true_steps = Counter(d["true_hallucination_step"] for d in hal)
    pred_steps = Counter(
        d["pred_hallucination_step"] for d in hal if d["pred_hallucination_step"] is not None
    )
    total_pred = sum(pred_steps.values())

    print("=== Position bias check ===")
    true_rate = true_steps.get(1, 0) / len(hal)
    print(f"  True step==1 frequency:      {true_steps.get(1,0)}/{len(hal)} = {true_rate*100:.1f}%")
    if total_pred:
        pred_rate = pred_steps.get(1, 0) / total_pred
        print(f"  Predicted step==1 frequency: {pred_steps.get(1,0)}/{total_pred} = {pred_rate*100:.1f}%")
        gap = pred_rate - true_rate
        if gap > 0.15:
            print(
                f"  WARNING: model over-predicts step 1 by {gap*100:.1f} points "
                f"relative to the true distribution -- possible positional shortcut."
            )
        else:
            print("  No strong positional shortcut detected.")
    print()


def domain_breakdown(data):
    print("=== Step accuracy by domain ===")
    for dom in sorted(set(d["domain"] for d in data)):
        hal = [d for d in data if d["domain"] == dom and d["true_is_hallucination"]]
        if not hal:
            continue
        correct = sum(1 for d in hal if d["pred_hallucination_step"] == d["true_hallucination_step"])
        print(f"  {dom:<20s} {correct}/{len(hal)} = {correct/len(hal)*100:.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    data = load(args.predictions)
    print(f"Loaded {len(data)} test predictions from {args.predictions}\n")

    check_id_field(data)
    judgment_metrics(data)
    step_localization_metrics(data)
    position_bias_check(data)
    domain_breakdown(data)


if __name__ == "__main__":
    main()
