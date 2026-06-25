# 👁️ AgentSight: Hallucination Localization via Tool Execution Context

**AgentSight** is a trained step-level classifier designed to detect and localize **tool-use hallucinations** in autonomous agent trajectories on the **AgentHallu benchmark**.

Unlike zero-shot LLM-as-a-judge methods, AgentSight is gradient-trained end-to-end on annotated step-level hallucination labels using a **DeBERTa-v3-base + LoRA** encoder with a cross-step **Transformer Context-Encoder**.

---

## 📊 Final Test Set Results (Locked — One-Shot Evaluation)

Test set sealed with SHA-256: `9604aae8eb5aec4ae666cfbe3053910f0570a807a4fa5515223dbca1aa66a7d8`

| Metric | Random | Open-Src Avg | Gemini-2.5-Pro | GPT-5 | **AgentSight (Ours)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Step Localization Acc** | 8.7% | 10.9% | 41.1% | — | **47.8%** |
| **Judgment Macro-F1** | 48.5% | 45.8% | 64.6% | 70.2% | **54.7%** |
| **Judgment Precision** | — | — | — | — | **58.5%** |
| **Judgment Recall** | — | — | — | — | **55.7%** |

> **AgentSight achieves 47.8% step localization accuracy — +6.7 points above Gemini-2.5-Pro** (the previous SOTA on this metric), trained with 1.4% of the DeBERTa parameters via LoRA.  
> Judgment F1 (54.7%) outperforms random and all open-source zero-shot systems by +8.9 points.

---

## 🔑 Key Finding: Tool Execution Context is the Dominant Signal (RQ3)

Ablation study on the validation set (val.json — not the sealed test set):

| Ablation | Step-Acc | Δ vs Full Model | Macro-F1 |
| :--- | :---: | :---: | :---: |
| **Full Model** | 48.5% | — baseline — | 61.6% |
| No Tool Context (`[ACTION]+[OBS]` zeroed) | 16.7% | **−31.8 pts** | 45.6% |
| No Query Context (`[QUERY]` zeroed) | 39.4% | −9.1 pts | 51.0% |
| No Cross-Step Attention (encoder bypassed) | 51.5% | +3.0 pts | 54.5% |

**Removing the tool execution channel causes a 31.8 point collapse** — dropping below the open-source zero-shot average. This is the strongest and most reproducible finding: tool observations, not reasoning traces, carry the hallucination localisation signal.

---

## 🏗️ Architecture

AgentSight encodes each agent step as a single concatenated string:
```
[QUERY] <task> [THOUGHT] <reasoning> [ACTION] <tool calls> [OBS] <tool responses>
```

This is fed through:
1. **DeBERTa-v3-base** with LoRA adapters (`r=16`, `α=64`) — 1.4% of parameters trained
2. **Transformer Context-Encoder** (3 Pre-LN layers, 8 heads) — cross-step attention
3. **Classification head** → per-step hallucination logit

Training: **AdamW** (lr=3e-5, wd=0.01) + linear warmup (10%) + cosine decay, gradient accumulation over 4 trajectories, dynamic `pos_weight` from data ratio, `WeightedRandomSampler` for class balance, threshold tuned on val set (thr=0.40).

---

## 🚀 Running the Full Pipeline

```bash
source venv/bin/activate
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

This will:
1. Verify `test.json` SHA-256 integrity
2. Train from scratch (max 50 epochs, early stopping patience=15)
3. Run the ablation study on val set
4. Generate the training curve (`notebooks/training_curve.png`)
5. Run the final sealed test evaluation (once)

### Or run steps individually:

```bash
# Train
python src/training/main.py --epochs 50 --lr 3e-5 --max_len 512

# Ablations (after training)
python src/training/run_ablations.py

# Final test evaluation (ONCE — sealed)
python src/training/run_test.py

# Verify predictions with full metric breakdown + position bias + domain breakdown
python verify_step_accuracy.py --predictions test_predictions.json
```

---

## 📦 SDK Integration

```python
from product.agentsight_api import AgentSightSDK

sdk = AgentSightSDK(model_weights_path="src/models/best_agentsight.pth")
results = sdk.evaluate_trajectory(raw_trajectory_json)

if results["is_hallucinated"]:
    print(f"Hallucination at step: {results['predicted_root_cause_step']}")
```

---

## 📁 Outputs

| File | Description |
| :--- | :--- |
| `src/models/best_agentsight.pth` | Best model weights (selected by val step-acc) |
| `src/models/best_agentsight_meta.json` | Epoch, threshold, val metrics for the best checkpoint |
| `test_predictions.json` | Per-trajectory test predictions with probabilities |
| `val_ablations.json` | Full per-trajectory ablation results |
| `full_training_log.txt` | Complete epoch-by-epoch training log |
| `notebooks/training_curve.png` | Loss + metric curves vs. baselines |

---

## 📈 Reproducibility

- `data/splits/test.json` SHA-256 is verified at the start of every test run
- All splits are stratified (70/15/15) with `random_state=42`
- Threshold is tuned on val set and stored in `best_agentsight_meta.json` — never re-tuned on test
- The test set is evaluated exactly once in the pipeline
