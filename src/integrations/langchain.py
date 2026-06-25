from typing import Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

import os
import sys

# Ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.monitor import AgentMonitor

class AgentSightCallback(BaseCallbackHandler):
    """
    Callback handler for LangChain that integrates AgentSight
    to monitor trajectories for hallucinations in real time.
    """
    def __init__(self):
        self.monitor = AgentMonitor()

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when chain starts to capture the initial query."""
        # Typically the main input key is 'input' or similar
        query = inputs.get("input", str(inputs))
        self.monitor.set_query(query)

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action. We save the action to be added with observation later."""
        self.current_thought = action.log if hasattr(action, 'log') else ""
        self.current_action = action.tool

    def on_tool_end(
        self,
        output: Any,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool ends running to add the full step."""
        thought = getattr(self, "current_thought", "")
        action = getattr(self, "current_action", "")
        observation = str(output)
        
        self.monitor.add_step(
            thought=thought,
            action=action,
            observation=observation
        )

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run when agent ends."""
        report = self.monitor.get_report()
        report.summary()
