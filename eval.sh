#!/bin/bash
# Script to run evaluation on dataset 1, enabling resume functionality.

# --- CONFIGURATION ---
EVAL_DATA_FILE="./data/indoor_flying3_data.hdf5"
EVAL_GT_FILE="./data/indoor_flying3_gt.hdf5"
CKPT_TO_EVALUATE="./checkpoints/vo_model_ep30.pth"
CHECKPOINT_DIR="./eval_checkpoints_flying3"
CHUNK_SIZE=1000

# --- UNCOMMENT ONE OPTION BELOW ---

# OPTION 1: CLEAN EVALUATION (Force restart: delete old chunks and re-run inference)
python main.py \
  --data-file "$EVAL_DATA_FILE" \
  --gt-file "$EVAL_GT_FILE" \
  --checkpoint "$CKPT_TO_EVALUATE" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --chunk-size "$CHUNK_SIZE" \
  --use-stereo --use-calib \
  --force \
  --run

# OPTION 2: RESUME EVALUATION (Continue from existing chunks or start fresh if none found)
# Note: This is recommended for long runs.
# python main.py \
#   --data-file "$EVAL_DATA_FILE" \
#   --gt-file "$EVAL_GT_FILE" \
#   --checkpoint "$CKPT_TO_EVALUATE" \
#   --checkpoint-dir "$CHECKPOINT_DIR" \
#   --chunk-size "$CHUNK_SIZE" \
#   --use-stereo --use-calib \
#   --enable-checkpointing \
#   --resume \
#   --run

echo "Evaluation on dataset 3 complete. Metrics printed above."