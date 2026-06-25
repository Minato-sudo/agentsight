"""
product/agentsight_api.py — Clean API wrapper around AgentMonitor.

This module is what external Python code should import.
The FastAPI server (api_server.py) also uses it.

Usage:
    from product.agentsight_api import AgentSightAPI

    api = AgentSightAPI()
    result = api.detect(query="Find files", trajectory=[...])
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of where this is imported from
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agentsight_sdk import AgentMonitor


class AgentSightAPI:
    """
    Thin wrapper around AgentMonitor with a detect() interface that accepts
    (query, trajectory) instead of a single dict — convenient for REST APIs.

    Parameters
    ----------
    weights_source : str | None
        Local directory with best_agentsight.pth, or HF repo id.
        None → auto-resolves: local weights first, then HuggingFace Hub.
    """

    def __init__(self, weights_source: str | None = None):
        self._monitor = AgentMonitor(weights_source=weights_source)

    @property
    def threshold(self) -> float:
        return self._monitor.threshold

    def detect(self, query: str, trajectory: list[dict]) -> dict:
        """
        Run step-level hallucination detection on a single trajectory.

        Parameters
        ----------
        query : str
            The original user task / question.
        trajectory : list[dict]
            List of step dicts. Each step should have:
                step           (int)  — step number
                content        (str)  — reasoning / thought
                tool_calls     (list) — [{name, arguments}]   optional
                tool_responses (list) — [str, ...]            optional

        Returns
        -------
        dict:
            is_hallucinated          bool
            predicted_root_cause_step  int | None
            step_probabilities       list[float]  — one per step
            step_analysis            list[dict]   — step-level detail
            threshold                float
        """
        sample = {"question": query, "trajectory": trajectory}
        result = self._monitor.evaluate_trajectory(sample)

        # Build the step_analysis list the dashboard expects
        probs = result.get("step_probabilities", [])
        step_analysis = []
        for i, step in enumerate(trajectory):
            prob = probs[i] if i < len(probs) else 0.0
            step_analysis.append({
                "step":                    step.get("step", i + 1),
                "hallucination_probability": round(prob, 4),
                "is_flagged":              prob > self._monitor.threshold,
                "content_preview":         str(step.get("content", ""))[:120],
                "tool_calls":              step.get("tool_calls", []),
                "tool_responses":          step.get("tool_responses", []),
            })

        return {
            "is_hallucinated":            result["is_hallucinated"],
            "predicted_root_cause_step":  result["predicted_root_cause_step"],
            "max_hallucination_prob":     round(max(probs) if probs else 0.0, 4),
            "step_probabilities":         [round(p, 4) for p in probs],
            "step_analysis":              step_analysis,
            "threshold":                  self._monitor.threshold,
            "n_steps":                    len(trajectory),
        }


# ── quick smoke test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    api = AgentSightAPI()
    print(f"Loaded. Threshold = {api.threshold}")

    sample_trajectory = [
        {
            "step": 1,
            "content": "I will search for the capital of France.",
            "tool_calls": [{"name": "web_search", "arguments": {"query": "capital of France"}}],
            "tool_responses": ["Paris is the capital of France."],
        },
        {
            "step": 2,
            "content": "Based on the search, the capital is Berlin.",
            "tool_calls": [],
            "tool_responses": [],
        },
    ]

    result = api.detect(
        query="What is the capital of France?",
        trajectory=sample_trajectory,
    )
    print(json.dumps(result, indent=2))
