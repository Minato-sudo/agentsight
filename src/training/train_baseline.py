import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import json
import argparse
from sklearn.metrics import f1_score, recall_score

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..", "..")
sys.path.insert(0, project_root)

from src.data.preprocessor import StepPreprocessor
from src.data.dataset import get_dataloader
from src.models.baseline_model import VanillaBaselineModel

def train_baseline_epoch(model, loader, optimizer):
    model.train()
    device = next(model.parameters()).device
    
    # Add pos_weight to handle trajectory-level class imbalance 
    # (Without this, it predicts 0 for everything since most trajectories are clean)
    pos_weight = torch.tensor([5.0]).to(device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    total_loss = 0
    
    for batch in loader:
        device = next(model.parameters()).device
        
        # Squeeze DataLoader dimension
        input_ids = batch["input_ids"].squeeze(0).to(device)
        attention_mask = batch["attention_mask"].squeeze(0).to(device)
        
        # Clamp input_ids
        vocab_size = model.encoder.config.vocab_size
        input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
        
        # For the baseline, we only care if the overall trajectory is hallucinated
        # hal_label shape is (N_steps,). If ANY step is hallucinated, the trajectory is 1.0.
        hal_labels = batch["hal_label"].squeeze(0).float().to(device)
        trajectory_label = hal_labels.max().unsqueeze(0) # Shape: (1,)
        
        optimizer.zero_grad()
        
        # Forward pass (Baseline predicts 1 score for the whole trajectory)
        logits = model(input_ids, attention_mask)
        
        # The model now outputs a single logit for the trajectory (shape: 1,)
        traj_logit = logits[0].unsqueeze(0)
        
        loss = bce(traj_logit, trajectory_label)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader) if len(loader) > 0 else 0

def evaluate_baseline(model, test_samples, preprocessor):
    model.eval()
    hal_preds, hal_true = [], []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for sample in test_samples:
            steps = preprocessor.encode_trajectory(sample)
            if not steps:
                continue
                
            input_ids = []
            attention_masks = []
            for step in steps:
                input_ids.append(step["encoding"]["input_ids"].squeeze(0))
                attention_masks.append(step["encoding"]["attention_mask"].squeeze(0))
                
            input_ids = torch.stack(input_ids).to(device)
            attention_masks = torch.stack(attention_masks).to(device)
            
            vocab_size = model.encoder.config.vocab_size
            input_ids = torch.clamp(input_ids, min=0, max=vocab_size - 1)
            
            logits = model(input_ids, attention_masks)
            traj_logit = logits[0]
            prob = torch.sigmoid(traj_logit).item()
            
            hal_preds.append(1 if prob > 0.5 else 0)
            hal_true.append(1 if sample.get("is_hallucination") else 0)
            
    f1 = f1_score(hal_true, hal_preds, average="macro", zero_division=0)
    rec = recall_score(hal_true, hal_preds, average="macro", zero_division=0)
    
    return {"judgment_f1": f1, "judgment_recall": rec}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4) # Higher LR since encoder is frozen
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    splits_dir = os.path.join(project_root, "data", "splits")
    preprocessor = StepPreprocessor()

    train_loader = get_dataloader(os.path.join(splits_dir, "train.json"), preprocessor, batch_size=1, shuffle=True)
    
    with open(os.path.join(splits_dir, "val.json"), "r") as f:
        val_samples = json.load(f)

    print("Initializing Vanilla Baseline Model...")
    model = VanillaBaselineModel()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_f1 = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        avg_train_loss = train_baseline_epoch(model, train_loader, optimizer)
        print(f"Train Loss: {avg_train_loss:.4f}")
        
        metrics = evaluate_baseline(model, val_samples, preprocessor)
        print(f"Validation Metrics: Judgment F1: {metrics['judgment_f1']*100:.1f}% | Judgment Recall: {metrics['judgment_recall']*100:.1f}%")
              
        if metrics['judgment_f1'] > best_val_f1:
            best_val_f1 = metrics['judgment_f1']
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(project_root, "src", "models", "baseline_model.pth"))
            print("  [*] New best baseline saved!")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

if __name__ == "__main__":
    main()
