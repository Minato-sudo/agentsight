import os
import sys
from typing import List, Dict, Optional

# Ensure imports work
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from product.agentsight_api import AgentSightDetector

class StepResult:
    def __init__(self, step_index: int, hallucination_prob: float, risk_score: float, is_flagged: bool):
        self.step_index = step_index
        self.hallucination_prob = hallucination_prob
        self.risk_score = risk_score
        self.is_flagged = is_flagged

class Report:
    def __init__(self, overall_hallucination: bool, riskiest_step: Optional[StepResult], steps: List[StepResult]):
        self.overall_hallucination = overall_hallucination
        self.riskiest_step = riskiest_step
        self.steps = steps

    def summary(self):
        print("=== AgentSight Summary ===")
        print(f"Overall Hallucination Detected: {self.overall_hallucination}")
        if self.riskiest_step:
            print(f"Riskiest Step: {self.riskiest_step.step_index} (Prob: {self.riskiest_step.hallucination_prob:.3f})")
        for step in self.steps:
            flag = "🚨" if step.is_flagged else "✅"
            print(f"Step {step.step_index}: {flag} Prob={step.hallucination_prob:.3f}, Risk={step.risk_score:.3f}")

class AgentMonitor:
    def __init__(self):
        self.detector = AgentSightDetector()
        self.query = ""
        self.steps = []
        self.step_counter = 1

    def set_query(self, query: str):
        self.query = query

    def add_step(self, thought: str, action: str, observation: str) -> StepResult:
        step_dict = {
            "step": self.step_counter,
            "content": thought,
            "tool_calls": [],
            "tool_responses": [observation]
        }
        
        # Simple parse for action if it's structured, else just treat as text
        # If action is in a format we can parse, we could, but for generic compat, put it in tool_calls
        step_dict["tool_calls"].append({"name": "AgentAction", "arguments": {"action_str": action}})

        self.steps.append(step_dict)
        
        # We can run the detector incrementally or just return a dummy result until get_report is called
        # But the PDF implies add_step returns a result immediately.
        # However, AgentSightModel takes the full sequence up to now.
        result = self.detector.detect(self.query, self.steps)
        
        # The latest step is at the end of the details array
        latest_detail = result["details"][-1] if result["details"] else {"prob": 0.0, "risk": 0.0}
        
        is_flagged = latest_detail["prob"] > 0.5
        
        step_result = StepResult(
            step_index=self.step_counter,
            hallucination_prob=latest_detail["prob"],
            risk_score=latest_detail["risk"],
            is_flagged=is_flagged
        )
        
        self.step_counter += 1
        return step_result

    def get_report(self) -> Report:
        result = self.detector.detect(self.query, self.steps)
        
        step_results = []
        riskiest_step = None
        max_prob = 0.0
        
        for i, detail in enumerate(result["details"]):
            sr = StepResult(
                step_index=detail["step"],
                hallucination_prob=detail["prob"],
                risk_score=detail["risk"],
                is_flagged=detail["prob"] > 0.5
            )
            step_results.append(sr)
            if detail["prob"] > max_prob:
                max_prob = detail["prob"]
                riskiest_step = sr
                
        return Report(
            overall_hallucination=result["is_hallucinated"],
            riskiest_step=riskiest_step,
            steps=step_results
        )
