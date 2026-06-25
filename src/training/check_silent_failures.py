"""
Silent failure checker — outputs per-sample JSON.
"""
import os, sys, json, traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

from src.data.preprocessor import StepPreprocessor

def main():
    splits_dir = os.path.join(project_root, "data", "splits")
    with open(os.path.join(splits_dir, "val.json")) as f:
        val_samples = json.load(f)

    preprocessor = StepPreprocessor()
    results = []

    for i, sample in enumerate(val_samples):
        entry = {
            "sample_idx": i,
            "trajectory_id": sample.get("model_id", sample.get("trajectory_id", "unknown")),
            "domain": sample.get("question_domain", "unknown"),
            "is_hallucination": sample.get("is_hallucination", False),
            "n_steps_in_json": len(sample.get("trajectory", sample.get("history", []))),
            "n_steps_encoded": None,
            "encoding_exception": None,
        }
        try:
            steps = preprocessor.encode_trajectory(sample)
            entry["n_steps_encoded"] = len(steps)
            entry["encoding_exception"] = None
        except Exception:
            entry["n_steps_encoded"] = 0
            entry["encoding_exception"] = traceback.format_exc()

        results.append(entry)

    out_path = os.path.join(project_root, "val_encoding_check.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    n_failures = sum(1 for r in results if r["encoding_exception"] is not None)
    n_empty = sum(1 for r in results if r["n_steps_encoded"] == 0 and r["encoding_exception"] is None)
    print(f"Total val samples: {len(results)}")
    print(f"Encoding exceptions: {n_failures}")
    print(f"Empty trajectories (no steps, no exception): {n_empty}")
    print(f"Raw output written to: {out_path}")

if __name__ == "__main__":
    main()
