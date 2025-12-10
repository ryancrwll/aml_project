#!/bin/bash
# Script to run clean training or fine-tuning from a checkpoint.

# --- CONFIGURATION ---
DATASET_IDS="2,3,4"
EPOCHS=30  # Total epochs to run
LR=1e-4    # Learning rate for initial training
CKPT_PATH="./checkpoints/vo_model_ep20.pth" # Path to save final clean checkpoint

# --- UNCOMMENT ONE OPTION BELOW ---

# OPTION 1: CLEAN TRAINING (Start from scratch)
python train.py \
  --datasets "$DATASET_IDS" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --use-stereo --use-calib \
  --run

# OPTION 2: FINE-TUNE/RESUME (Load model weights from an existing checkpoint)
# Note: Use a smaller LR for fine-tuning (e.g., 1e-5)
# python train.py \
#   --datasets "$DATASET_IDS" \
#   --resume "$CKPT_PATH" \
#   --epochs 5 \
#   --lr 1e-5 \
#   --use-stereo --use-calib \
#   --run

# echo "Training complete. Checkpoints saved to ./checkpoints/"