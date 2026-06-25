import json
import os
import random

def load_data(raw_dir="data/raw/"):
    all_samples = []
    for f in os.listdir(raw_dir):
        if f.endswith(".json"):
            with open(os.path.join(raw_dir, f), 'r', encoding='utf-8') as fh:
                sample = json.load(fh)
                
                # Fix types since some fields are strings in the raw JSON
                if isinstance(sample.get("is_hallucination"), str):
                    sample["is_hallucination"] = sample["is_hallucination"].lower() == "true"
                
                if sample.get("hallucination_step") is not None:
                    sample["hallucination_step"] = int(sample["hallucination_step"])
                
                # The implementation plan uses 'trajectory', but raw data has 'history'
                if "history" in sample and "trajectory" not in sample:
                    sample["trajectory"] = sample["history"]
                    
                all_samples.append(sample)
    return all_samples

def random_baseline_accuracy(samples):
    correct = 0
    hallucinated_samples = [s for s in samples if s.get("is_hallucination")]
    
    if not hallucinated_samples:
        return 0.0

    for s in hallucinated_samples:
        n_steps = len(s.get("trajectory", []))
        if n_steps == 0:
            continue
        predicted_step = random.randint(1, n_steps)
        if predicted_step == s.get("hallucination_step"):
            correct += 1
            
    return correct / len(hallucinated_samples)

if __name__ == "__main__":
    # Ensure reproducibility
    random.seed(42)
    
    # Load from the correct path relative to the root folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    data_dir = os.path.join(project_root, "data", "raw")
    
    samples = load_data(data_dir)
    print(f"Total samples loaded: {len(samples)}")
    
    hallucinated = [s for s in samples if s.get("is_hallucination")]
    clean = [s for s in samples if not s.get("is_hallucination")]
    print(f"Hallucinated: {len(hallucinated)}")
    print(f"Clean: {len(clean)}")
    
    # Run 1000 times and average
    scores = [random_baseline_accuracy(samples) for _ in range(1000)]
    avg_score = sum(scores) / len(scores) * 100
    print(f"Random baseline step localization accuracy: {avg_score:.2f}%")
