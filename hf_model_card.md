---
language: en
license: mit
tags:
  - hallucination-detection
  - agent-trajectories
  - deberta
  - lora
  - tool-use
  - step-classification
datasets:
  - AgentHallu
metrics:
  - f1
  - accuracy
model-index:
  - name: agentsight
    results:
      - task:
          type: text-classification
          name: Hallucination Step Localisation
        dataset:
          name: AgentHallu
          type: AgentHallu-benchmark
        metrics:
          - type: accuracy
            value: 0.478
            name: Step Localisation Accuracy (test, n=67)
          - type: f1
            value: 0.547
            name: Judgment Macro-F1 (test, n=105)
---

# AgentSight — Step-Level Hallucination Classifier

**AgentSight** is the first gradient-trained step-level hallucination classifier
on the [AgentHallu benchmark](https://arxiv.org/abs/...).  
It identifies *which step* in an autonomous agent trajectory is the root cause of a hallucination.

## Results (locked test set, one-shot evaluation)

| Metric | Value | 95% CI | Note |
|---|---|---|---|
| Step Localisation Acc | **47.8%** (32/67) | [36.3%, 59.5%] | vs Gemini 41.1% — p=0.163, not stat. sig. |
| Judgment Macro-F1 | **54.7%** | — | n=105 trajectories |
| Judgment Precision | 58.5% | — | threshold=0.40 |
| Judgment Recall | 55.7% | — | |

> The +6.7 point margin over Gemini-2.5-Pro is practically meaningful but not statistically significant at n=67. The 95% CI spans 23 points. We report this honestly.

## Key Finding

Removing the `[ACTION]+[OBS]` tool-execution channel from the input drops step localisation accuracy by **31.8 percentage points** (48.5% → 16.7% on the validation set). Tool output content is the dominant localisation signal.

## Architecture

- **Backbone**: DeBERTa-v3-base with LoRA adapters (r=16, α=64) — 1.42% of parameters trained
- **Context encoder**: 3-layer Pre-LN Transformer (8 heads) over the step sequence
- **Head**: 2-layer MLP → binary hallucination logit per step
- **Input**: `[QUERY] task [THOUGHT] reasoning [ACTION] tool_calls [OBS] responses` (max 512 tokens)
- **Decision threshold**: 0.40 (tuned on validation set)

## Quick Start

```python
from agentsight_sdk import AgentMonitor

# Auto-downloads weights from this repo
monitor = AgentMonitor("YOUR_HF_USERNAME/agentsight")

result = monitor.evaluate_trajectory(raw_trajectory_dict)

print(result["is_hallucinated"])               # True / False
print(result["predicted_root_cause_step"])      # e.g. 3
print(result["step_probabilities"])             # [0.12, 0.08, 0.91, 0.04]
```

## Input Format

```json
{
  "question": "What is the capital of France?",
  "history": [
    {
      "step": 1,
      "content": "I will search for this.",
      "tool_calls": [{"name": "web_search", "arguments": {"query": "capital France"}}],
      "tool_responses": ["Paris is the capital of France."]
    }
  ]
}
```

## Training

```bash
git clone https://github.com/Minato-sudo/agentsight
cd agentsight
pip install -r requirements.txt
./run_full_pipeline.sh   # verify → train → ablate → test (one-shot)
```

## Citation

```bibtex
@misc{agentsight2026,
  title={AgentSight: Step-Level Hallucination Localisation in Autonomous Agent Trajectories via Tool-Execution Context},
  author={Namikaze, Minato},
  year={2026},
  note={Paper + code: https://github.com/Minato-sudo/agentsight}
}
```

## Files

| File | Description |
|---|---|
| `best_agentsight.pth` | PyTorch state dict — load with `AgentSightModel()` |
| `best_agentsight_meta.json` | Threshold (0.40), val metrics, best epoch |
| `tokenizer_config/` | DeBERTa-v3-base tokeniser config (max_len=512) |

## Limitations

- Evaluated on AgentHallu only — generalisation to other benchmarks unknown
- n=67 test set means wide confidence intervals
- Judgment F1 (54.7%) is below frontier models (Gemini 64.6%, GPT-5 70.2%)
