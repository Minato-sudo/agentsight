"""
LangChain integration for AgentSight.

Allows you to monitor LangChain agents for hallucinations in real-time.
"""
from typing import Any, Dict, List, Optional
try:
    from langchain.callbacks.base import BaseCallbackHandler
    from langchain.schema import AgentAction, AgentFinish
except ImportError:
    BaseCallbackHandler = object
    AgentAction = None
    AgentFinish = None

from agentsight_sdk import AgentMonitor


class AgentSightCallback(BaseCallbackHandler):
    """
    A LangChain callback handler that streams execution steps to AgentSight
    for step-level hallucination detection.
    """

    def __init__(self, threshold: float = 0.40):
        if BaseCallbackHandler is object:
            raise ImportError("Please install langchain to use this callback: pip install langchain")
        
        self.monitor = AgentMonitor()
        self.monitor.threshold = threshold
        
        # State for the current trajectory
        self._trajectory = []
        self._current_step_num = 1
        self._current_query = "Unknown query"

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """Capture the initial user query if present."""
        if "input" in inputs:
            self._current_query = str(inputs["input"])
        elif "question" in inputs:
            self._current_query = str(inputs["question"])

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        """Record an action being taken."""
        self._trajectory.append({
            "step": self._current_step_num,
            "content": action.log.split("Action:")[0].strip(),
            "tool_calls": [{
                "name": action.tool,
                "arguments": action.tool_input if isinstance(action.tool_input, dict) else {"input": action.tool_input}
            }],
            "tool_responses": []
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> Any:
        """Record the output of the tool."""
        if self._trajectory:
            self._trajectory[-1]["tool_responses"].append(str(output))
            
            # Evaluate after tool finishes
            result = self.monitor.evaluate_trajectory({
                "question": self._current_query,
                "trajectory": self._trajectory
            })
            
            if result.get("is_hallucinated"):
                prob = result.get("max_hallucination_prob", 0.0)
                root_step = result.get("predicted_root_cause_step", self._current_step_num)
                print(f"\n[AgentSight] ⚠️ Hallucination Warning: Step {root_step} flagged (prob={prob:.2f})")
            
            self._current_step_num += 1

    def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> Any:
        """Final evaluation when the agent finishes."""
        result = self.monitor.evaluate_trajectory({
            "question": self._current_query,
            "trajectory": self._trajectory
        })
        
        if result.get("is_hallucinated"):
            root_step = result.get("predicted_root_cause_step")
            print(f"\n[AgentSight] 🚨 Final Verdict: Hallucination Detected (Root Cause: Step {root_step})")
        else:
            print("\n[AgentSight] ✅ Final Verdict: Trajectory Clean")
