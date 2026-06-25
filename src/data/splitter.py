import os
import json
from sklearn.model_selection import train_test_split
from baseline import load_data

def create_splits(data_dir, splits_dir):
    print("Loading raw dataset...")
    all_samples = load_data(data_dir)
    
    hal = [s for s in all_samples if s.get("is_hallucination")]
    clean = [s for s in all_samples if not s.get("is_hallucination")]
    
    # Stratified split for hallucinated: 70 / 15 / 15
    hal_train, hal_tmp = train_test_split(hal, test_size=0.30, random_state=42)
    hal_val, hal_test = train_test_split(hal_tmp, test_size=0.50, random_state=42)
    
    # Stratified split for clean: 70 / 15 / 15
    cln_train, cln_tmp = train_test_split(clean, test_size=0.30, random_state=42)
    cln_val, cln_test = train_test_split(cln_tmp, test_size=0.50, random_state=42)
    
    train = hal_train + cln_train
    val = hal_val + cln_val
    test = hal_test + cln_test
    
    os.makedirs(splits_dir, exist_ok=True)
    
    with open(os.path.join(splits_dir, "train.json"), "w", encoding='utf-8') as f:
        json.dump(train, f, indent=4)
        
    with open(os.path.join(splits_dir, "val.json"), "w", encoding='utf-8') as f:
        json.dump(val, f, indent=4)
        
    with open(os.path.join(splits_dir, "test.json"), "w", encoding='utf-8') as f:
        json.dump(test, f, indent=4)
        
    print(f"Splits saved to {splits_dir}")
    print(f"Train: {len(train)} (Hal: {len(hal_train)}, Clean: {len(cln_train)})")
    print(f"Val:   {len(val)} (Hal: {len(hal_val)}, Clean: {len(cln_val)})")
    print(f"Test:  {len(test)} (Hal: {len(hal_test)}, Clean: {len(cln_test)})")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(script_dir, "..", "..")
    data_dir = os.path.join(project_root, "data", "raw")
    splits_dir = os.path.join(project_root, "data", "splits")
    
    create_splits(data_dir, splits_dir)
