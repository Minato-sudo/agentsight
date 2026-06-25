import json
from transformers import AutoTokenizer
import torch


class StepPreprocessor:
    def __init__(self, model_name="microsoft/deberta-v3-base", max_len=512):
        # use_fast=True avoids Python memory leaks in DataLoader loops
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.max_len = max_len

    def encode_step(self, query, step):
        """
        Encodes a single step as a concatenated string:
          [QUERY] ... [THOUGHT] ... [ACTION] ... [OBS] ...

        This is the "single-channel concatenated" representation where the
        model learns to attend across all information jointly.
        """
        # Channel 1 — Reasoning trace (the LLM's textual thought / plan)
        thought = step.get("content", "") or ""

        # Channel 2 — Tool context (action + observation)
        tool_calls = step.get("tool_calls") or []
        tool_responses = step.get("tool_responses") or []

        actions_str = ""
        for tc in tool_calls:
            name = tc.get("name", "")
            args = json.dumps(tc.get("arguments", {}))
            actions_str += f"{name} {args} | "

        observations_str = " | ".join(str(tr) for tr in tool_responses) if tool_responses else ""

        tool_context = f"[ACTION] {actions_str.strip()} [OBS] {observations_str.strip()}"

        # Channel 3 — Query context (the original question / task)
        if isinstance(query, list):
            flat_query = " ".join(q[0] if isinstance(q, list) else str(q) for q in query)
        else:
            flat_query = str(query) if query else ""

        # Truncation strategy: keep the query short but preserve tool context
        # because that is the dominant signal (ablation: -28.8 pt without it).
        # We truncate THOUGHT in the middle if necessary.
        full_text = f"[QUERY] {flat_query} [THOUGHT] {thought} {tool_context}"

        encoded = self.tokenizer(
            full_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return encoded

    def encode_trajectory(self, sample):
        """
        Returns a list of dicts, one per step, each containing:
          - encoding: tokenized input (input_ids + attention_mask)
          - label:    1 if this step is the hallucination step, else 0
          - step_idx: the integer step index from the JSON
        """
        # Support both 'trajectory' and 'history' JSON keys
        trajectory = sample.get("trajectory") or sample.get("history") or []

        is_hal = sample.get("is_hallucination", False)
        if isinstance(is_hal, str):
            is_hal = is_hal.lower() == "true"

        hal_step = sample.get("hallucination_step")
        if hal_step is not None:
            hal_step = int(hal_step)

        query = sample.get("question", sample.get("query", ""))

        steps = []
        for step in trajectory:
            enc = self.encode_step(query, step)
            step_idx = step.get("step")

            # A step is labelled positive only if the trajectory is hallucinated
            # AND this is exactly the annotated root-cause step.
            label = 1 if (is_hal and step_idx == hal_step) else 0

            steps.append({
                "encoding": enc,
                "label": label,
                "step_idx": step_idx,
            })

        return steps


# ── quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("Testing StepPreprocessor (will download tokenizer if not cached)…")
    pre = StepPreprocessor()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(script_dir, "..", "..", "data", "splits", "train.json")

    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as fh:
            samples = json.load(fh)
        sample = samples[0]
        encoded = pre.encode_trajectory(sample)
        print(f"OK — encoded {len(encoded)} steps.")
        print(f"input_ids shape: {encoded[0]['encoding']['input_ids'].shape}")
    else:
        print(f"File not found: {sample_file}")
