"""
AgentSight Hallucination Guard.

A context manager that wraps agent execution, monitors steps, and can
interrupt execution if a hallucination is detected.
"""
from contextlib import contextmanager
from typing import Callable, Optional, Dict, Any, List

import sys
from pathlib import Path
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agentsight_sdk import AgentMonitor


class HallucinationDetected(Exception):
    """Exception raised when a hallucination exceeds the threshold."""
    pass


class HallucinationGuard:
    def __init__(
        self, 
        threshold: float = 0.40, 
        on_flag: str = "warn", 
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the guard.
        
        Args:
            threshold: Probability above which a step is flagged.
            on_flag: "warn" (print), "raise" (exception), "retry" (not fully impl), "pause".
            callback: Optional function called with result when flagged.
        """
        self.monitor = AgentMonitor()
        self.monitor.threshold = threshold
        self.on_flag = on_flag
        self.callback = callback
        
        self.flagged_steps = []
        self._trajectory = []
        self._query = "Unknown query"
        self._current_step_num = 1

    def set_query(self, query: str):
        """Set the initial task/query for context."""
        self._query = query

    def check_step(self, thought: str, action: str, observation: str, arguments: Dict[str, Any] = None):
        """
        Add a step and evaluate it immediately.
        """
        if arguments is None:
            arguments = {}
            
        self._trajectory.append({
            "step": self._current_step_num,
            "content": thought,
            "tool_calls": [{"name": action, "arguments": arguments}],
            "tool_responses": [observation]
        })
        
        result = self.monitor.evaluate_trajectory({
            "question": self._query,
            "trajectory": self._trajectory
        })
        
        if result.get("is_hallucinated"):
            self.flagged_steps.append(result)
            if self.callback:
                self.callback(result)
                
            if self.on_flag == "raise":
                root_step = result.get("predicted_root_cause_step", self._current_step_num)
                prob = result.get("max_hallucination_prob", 0.0)
                raise HallucinationDetected(
                    f"Hallucination root cause detected at Step {root_step} "
                    f"(prob={prob:.2f})"
                )
            elif self.on_flag == "warn":
                root_step = result.get("predicted_root_cause_step", self._current_step_num)
                prob = result.get("max_hallucination_prob", 0.0)
                print(f"[AgentSight Guard] ⚠️ Warning: Hallucination detected (Root Step {root_step}, prob={prob:.2f})")
                
        self._current_step_num += 1
        return result

    @property
    def was_flagged(self) -> bool:
        return len(self.flagged_steps) > 0

    @property
    def flagged_step(self) -> Optional[int]:
        if self.flagged_steps:
            return self.flagged_steps[-1].get("predicted_root_cause_step")
        return None


@contextmanager
def guard(threshold: float = 0.40, on_flag: str = "warn", callback: Optional[Callable] = None):
    """
    Context manager for wrapping agent execution.
    
    Example:
        with agentsight.guard(threshold=0.40, on_flag="raise") as monitor:
            monitor.set_query("Summarize Apple Q3")
            # inside your agent loop
            monitor.check_step(thought, tool, obs)
    """
    g = HallucinationGuard(threshold, on_flag, callback)
    yield g
