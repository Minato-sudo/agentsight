"""
AgentSight — minimal inference SDK.

Loads the released weights from HuggingFace Hub (or a local path) and
exposes a single evaluate_trajectory() call that returns per-step
hallucination probabilities plus the predicted root-cause step.

Usage
-----
    from agentsight_sdk import AgentMonitor

    monitor = AgentMonitor()                        # downloads from HF Hub
    monitor = AgentMonitor("./local_weights")       # or from a local dir

    result = monitor.evaluate_trajectory(raw_json)

    print(result["is_hallucinated"])               # bool
    print(result["predicted_root_cause_step"])      # int | None
    print(result["step_probabilities"])             # list[float]
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

# ── project src on path ────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from src.models.agentsight import AgentSightModel
from src.data.preprocessor import StepPreprocessor

_HF_REPO_ID = "talha1234567/Agentic-Ai"   # HuggingFace: https://huggingface.co/talha1234567/Agentic-Ai
_DEFAULT_THRESHOLD = 0.40


class AgentMonitor:
    """
    One-call interface for evaluating a single agent trajectory.

    Parameters
    ----------
    weights_source : str | None
        Either:
        - A local directory containing ``best_agentsight.pth`` and
          ``best_agentsight_meta.json``  (checked first), OR
        - A HuggingFace repo id string like ``"username/agentsight"``.
        - None → uses the published HF repo defined in ``_HF_REPO_ID``.
    device : str | None
        ``"cuda"``, ``"cpu"``, or None (auto-detects).
    """

    def __init__(
        self,
        weights_source: str | None = None,
        device: str | None = None,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.threshold, weights_path = self._resolve_weights(weights_source)
        self.preprocessor = StepPreprocessor(max_len=512)
        self.model = AgentSightModel()
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

    # ── private helpers ────────────────────────────────────────────────────

    def _resolve_weights(self, source: str | None) -> tuple[float, str]:
        """
        Returns (threshold, local_path_to_pth_file).
        Downloads from HF Hub if no local path is found.
        """
        # 1. Check local path first
        if source and os.path.isdir(source):
            pth = os.path.join(source, "best_agentsight.pth")
            meta = os.path.join(source, "best_agentsight_meta.json")
            if os.path.exists(pth):
                thr = self._read_threshold(meta)
                return thr, pth

        # 2. Check the default model directory inside the package
        local_pth = _HERE / "src" / "models" / "best_agentsight.pth"
        local_meta = _HERE / "src" / "models" / "best_agentsight_meta.json"
        if local_pth.exists():
            thr = self._read_threshold(str(local_meta))
            return thr, str(local_pth)

        # 3. Download from HuggingFace Hub
        repo_id = source or _HF_REPO_ID
        print(f"Downloading weights from HuggingFace Hub: {repo_id} …")
        from huggingface_hub import hf_hub_download

        pth_path = hf_hub_download(repo_id=repo_id, filename="best_agentsight.pth")
        try:
            meta_path = hf_hub_download(
                repo_id=repo_id, filename="best_agentsight_meta.json"
            )
            thr = self._read_threshold(meta_path)
        except Exception:
            thr = _DEFAULT_THRESHOLD
        return thr, pth_path

    @staticmethod
    def _read_threshold(meta_path: str) -> float:
        try:
            with open(meta_path) as f:
                return float(json.load(f).get("threshold", _DEFAULT_THRESHOLD))
        except Exception:
            return _DEFAULT_THRESHOLD

    # ── public API ──────────────────────────────────────────────────────────

    def evaluate_trajectory(self, trajectory_json: dict | str) -> dict[str, Any]:
        """
        Evaluate a single agent trajectory.

        Parameters
        ----------
        trajectory_json : dict | str
            Either a dict (already parsed) or a JSON string.
            Must contain a ``"history"`` or ``"trajectory"`` key with a list
            of step dicts, and a ``"question"`` / ``"query"`` key.

        Returns
        -------
        dict with keys:
            is_hallucinated         bool
            predicted_root_cause_step   int | None
            step_probabilities      list[float]   (one per step)
            threshold               float
        """
        if isinstance(trajectory_json, str):
            trajectory_json = json.loads(trajectory_json)

        steps = self.preprocessor.encode_trajectory(trajectory_json)
        if not steps:
            return {
                "is_hallucinated": False,
                "predicted_root_cause_step": None,
                "step_probabilities": [],
                "threshold": self.threshold,
                "error": "empty_trajectory",
            }

        ids = torch.stack(
            [s["encoding"]["input_ids"].squeeze(0) for s in steps]
        ).to(self.device)
        mask = torch.stack(
            [s["encoding"]["attention_mask"].squeeze(0) for s in steps]
        ).to(self.device)

        vocab_size = self.model.encoder.config.vocab_size
        ids = torch.clamp(ids, 0, vocab_size - 1)

        with torch.no_grad():
            logits = self.model(ids, mask)
            probs = torch.sigmoid(logits).cpu().tolist()

        if isinstance(probs, float):
            probs = [probs]

        max_prob = max(probs)
        is_hal = max_prob > self.threshold
        pred_step = steps[probs.index(max_prob)]["step_idx"] if is_hal else None

        return {
            "is_hallucinated": is_hal,
            "predicted_root_cause_step": pred_step,
            "step_probabilities": probs,
            "threshold": self.threshold,
        }


# ── backward-compat alias ──────────────────────────────────────────────────
AgentSightSDK = AgentMonitor
