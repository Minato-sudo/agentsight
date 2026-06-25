"""
product/api_server.py — FastAPI backend for AgentSight.

Endpoints:
    POST /analyze        — run hallucination detection on a trajectory
    GET  /health         — liveness probe
    GET  /model/info     — model metadata (threshold, val metrics, HF link)

Start with:
    cd "/home/minato/Documents/Agentic Ai Project/agentsight"
    venv/bin/uvicorn product.api_server:app --reload --port 8000

Or use the helper script:
    ./start_server.sh
"""
from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Any

# ── path setup ─────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from product.agentsight_api import AgentSightAPI

# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="AgentSight API",
    description=(
        "Step-level hallucination detection for autonomous agent trajectories. "
        "Paper: https://github.com/Minato-sudo/agentsight | "
        "Model: https://huggingface.co/talha1234567/Agentic-Ai"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow the Vite dev server (port 5173) and any localhost port
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── lazy-loaded singleton ────────────────────────────────────────────────────
_api: AgentSightAPI | None = None
_startup_error: str | None = None


def get_api() -> AgentSightAPI:
    global _api, _startup_error
    if _startup_error:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {_startup_error}")
    if _api is None:
        raise HTTPException(status_code=503, detail="Model is still loading. Retry in a few seconds.")
    return _api


@app.on_event("startup")
async def load_model():
    global _api, _startup_error
    try:
        print("Loading AgentSight model …")
        t0 = time.time()
        _api = AgentSightAPI()
        print(f"Model ready in {time.time() - t0:.1f}s  (threshold={_api.threshold})")
    except Exception as e:
        _startup_error = str(e)
        print(f"ERROR loading model: {e}")


# ── Request / Response models ────────────────────────────────────────────────

class ToolCall(BaseModel):
    name: str = ""
    arguments: dict[str, Any] = {}


class TrajectoryStep(BaseModel):
    step: int
    content: str = ""
    tool_calls: list[ToolCall] = []
    tool_responses: list[str] = []


class AnalyzeRequest(BaseModel):
    query: str = Field(..., description="The original user task or question")
    trajectory: list[TrajectoryStep] = Field(
        ..., description="List of agent trajectory steps"
    )


class StepDetail(BaseModel):
    step: int
    hallucination_probability: float
    is_flagged: bool
    content_preview: str
    tool_calls: list[ToolCall]
    tool_responses: list[str]


class AnalyzeResponse(BaseModel):
    is_hallucinated: bool
    predicted_root_cause_step: int | None
    max_hallucination_prob: float
    step_probabilities: list[float]
    step_analysis: list[StepDetail]
    threshold: float
    n_steps: int
    processing_time_ms: float


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    """Liveness probe. Returns 200 when the model is loaded and ready."""
    api = get_api()
    return {
        "status": "ok",
        "model_loaded": True,
        "threshold": api.threshold,
    }


@app.get("/model/info", tags=["System"])
def model_info():
    """Model metadata — val metrics, links, threshold."""
    meta_path = _ROOT / "src" / "models" / "best_agentsight_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return {
        "model_name":    "AgentSight",
        "architecture":  "DeBERTa-v3-base + LoRA (r=16) + 3-layer Transformer",
        "trainable_params": "2,654,208 (1.42%)",
        "threshold":     meta.get("threshold", 0.40),
        "val_step_acc":  meta.get("val_step_acc", None),
        "val_f1":        meta.get("val_f1", None),
        "best_epoch":    meta.get("epoch", None),
        "test_step_acc": 0.478,
        "test_f1":       0.547,
        "test_ci":       "[36.3%, 59.5%]",
        "github":        "https://github.com/Minato-sudo/agentsight",
        "huggingface":   "https://huggingface.co/talha1234567/Agentic-Ai",
    }


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Detection"])
def analyze(request: AnalyzeRequest):
    """
    Run step-level hallucination detection on an agent trajectory.

    Returns per-step hallucination probabilities and the predicted root-cause step.
    """
    api = get_api()
    t0 = time.time()

    # Convert Pydantic models → plain dicts for the SDK
    trajectory_dicts = [
        {
            "step":           s.step,
            "content":        s.content,
            "tool_calls":     [{"name": tc.name, "arguments": tc.arguments} for tc in s.tool_calls],
            "tool_responses": s.tool_responses,
        }
        for s in request.trajectory
    ]

    try:
        result = api.detect(query=request.query, trajectory=trajectory_dicts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

    elapsed_ms = (time.time() - t0) * 1000

    return {
        **result,
        "processing_time_ms": round(elapsed_ms, 1),
    }


# ── Dev entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("product.api_server:app", host="0.0.0.0", port=8000, reload=True)
