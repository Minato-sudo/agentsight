#!/bin/bash
# AgentSight — Full Reproducible Training Pipeline
#
# Steps:
#   1. Verify test set SHA-256 has not drifted
#   2. Train the model from scratch (up to 50 epochs, early stopping)
#   3. Run ablation study on val set
#   4. Generate the training curve PNG
#   5. Run final test-set evaluation (ONCE — sealed)

set -e
cd "/home/minato/Documents/Agentic Ai Project/agentsight"
source venv/bin/activate

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " AgentSight — Full Reproducible Pipeline"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── Step 1: Integrity check ──────────────────────────────────────────────────
EXPECTED="9604aae8eb5aec4ae666cfbe3053910f0570a807a4fa5515223dbca1aa66a7d8"
ACTUAL=$(sha256sum data/splits/test.json | awk '{print $1}')
if [ "$ACTUAL" != "$EXPECTED" ]; then
  echo "ERROR: test.json SHA-256 mismatch. Aborting."
  echo "  Expected: $EXPECTED"
  echo "  Got:      $ACTUAL"
  exit 1
fi
echo "[✓] test.json integrity verified."
echo ""

# ── Step 2: Train ────────────────────────────────────────────────────────────
echo "[2/5] Training (max 50 epochs, lr=3e-5, AdamW, cosine warmup) …"
python src/training/main.py \
  --epochs 50 \
  --lr 3e-5 \
  --grad_accum 4 \
  --patience 15 \
  --max_len 512 \
  --max_steps 20 \
  --weight_decay 0.01 \
  --warmup_ratio 0.10 \
  2>&1 | tee full_training_log.txt

echo ""

# ── Step 3: Ablation study ───────────────────────────────────────────────────
echo "[3/5] Running ablation study on val set …"
python src/training/run_ablations.py 2>&1 | tee val_ablations_log.txt
echo ""

# ── Step 4: Training curve ───────────────────────────────────────────────────
echo "[4/5] Generating training curve …"
python src/training/plot_results.py
echo ""

# ── Step 5: Final test evaluation ────────────────────────────────────────────
echo "[5/5] Final test-set evaluation (ONCE — locked) …"
python src/training/run_test.py 2>&1 | tee test_eval_log.txt

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Pipeline complete.  Outputs:"
echo "   Model weights   : src/models/best_agentsight.pth"
echo "   Threshold meta  : src/models/best_agentsight_meta.json"
echo "   Test predictions: test_predictions.json"
echo "   Ablations       : val_ablations.json"
echo "   Training curve  : notebooks/training_curve.png"
echo "   Test eval log   : test_eval_log.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
