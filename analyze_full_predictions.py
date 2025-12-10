#!/usr/bin/env python3
"""
Check if the model is predicting NaN/Inf or if scale factors drift over the full trajectory.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MVSECDataset
from VOnetwork import VONet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/vo_model_ep30.pth'
TRANS_SCALE_FACTOR = 1000.0

# Load model and dataset
model = VONet(input_channels=10).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

ds = MVSECDataset(
    data_path='./data/indoor_flying3_data.hdf5',
    gt_path='./data/indoor_flying3_gt.hdf5',
    seq_len=10,
    crop_params={'H': 260, 'W': 346, 'B': 5},
    use_stereo=True,
    use_calib=True,
    calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'
)

loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)

print("=" * 80)
print("FULL DATASET PREDICTION ANALYSIS - DATASET 3")
print("=" * 80)

scale_factors = []
pred_mags = []
gt_mags = []
has_nan = False
has_inf = False

print("\nProcessing all sequences...")
with torch.no_grad():
    for i, (voxels, imu_feats, imu_ts, targets) in enumerate(tqdm(loader)):
        voxels = voxels.to(DEVICE)
        imu_feats = imu_feats.to(DEVICE)
        preds = model(voxels, imu_feats)
        
        preds_np = preds.cpu().numpy().reshape(-1, 6)
        preds_np[:, :3] = preds_np[:, :3] / TRANS_SCALE_FACTOR
        targets_np = targets.numpy().reshape(-1, 6)
        
        # Check for NaN/Inf
        if np.any(np.isnan(preds_np)):
            has_nan = True
            print(f"\n  WARNING: NaN detected at sequence {i}")
        if np.any(np.isinf(preds_np)):
            has_inf = True
            print(f"\n  WARNING: Inf detected at sequence {i}")
        
        # Compute scale factors
        pred_trans_mags = np.linalg.norm(preds_np[:, :3], axis=1)
        gt_trans_mags = np.linalg.norm(targets_np[:, :3], axis=1)
        
        valid = gt_trans_mags > 1e-6
        if valid.any():
            batch_scales = pred_trans_mags[valid] / gt_trans_mags[valid]
            scale_factors.extend(batch_scales)
            pred_mags.extend(pred_trans_mags[valid])
            gt_mags.extend(gt_trans_mags[valid])

print(f"\n--- RESULTS ---")
print(f"Sequences processed: {len(ds)}")
print(f"Valid prediction-GT pairs: {len(scale_factors)}")
print(f"Has NaN: {has_nan}")
print(f"Has Inf: {has_inf}")

if scale_factors:
    scale_factors = np.array(scale_factors)
    pred_mags = np.array(pred_mags)
    gt_mags = np.array(gt_mags)
    
    print(f"\nPrediction Magnitudes:")
    print(f"  Min: {pred_mags.min():.6e} m")
    print(f"  Max: {pred_mags.max():.6e} m")
    print(f"  Mean: {pred_mags.mean():.6e} m")
    print(f"  Median: {np.median(pred_mags):.6e} m")
    
    print(f"\nGT Magnitudes:")
    print(f"  Min: {gt_mags.min():.6e} m")
    print(f"  Max: {gt_mags.max():.6e} m")
    print(f"  Mean: {gt_mags.mean():.6e} m")
    print(f"  Median: {np.median(gt_mags):.6e} m")
    
    print(f"\nScale Factors (pred/gt):")
    print(f"  Min: {scale_factors.min():.6f}")
    print(f"  Max: {scale_factors.max():.6f}")
    print(f"  Mean: {scale_factors.mean():.6f}")
    print(f"  Median: {np.median(scale_factors):.6f}")
    print(f"  Std: {scale_factors.std():.6f}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("If mean scale factor << 1, predictions are systematically too small.")
print("If std is high, predictions are inconsistent across dataset.")
print("=" * 80)
