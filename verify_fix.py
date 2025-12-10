#!/usr/bin/env python3
"""
Verify the fix by loading actual dataset samples and checking alignment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataloader import MVSECDataset
import torch

print("=" * 80)
print("VERIFYING TIMESTAMP FIX")
print("=" * 80)

# Load dataset
ds = MVSECDataset(
    data_path='./data/indoor_flying4_data.hdf5',
    gt_path='./data/indoor_flying4_gt.hdf5',
    seq_len=5,
    crop_params={'H': 260, 'W': 346, 'B': 5},
    use_stereo=False,
    use_calib=False
)

print(f"Dataset loaded. Testing multiple samples...\n")

# Test several samples
for sample_idx in [0, 10, 50, 100, 200]:
    try:
        voxels, imu_feats, imu_ts, poses = ds[sample_idx]

        print(f"Sample {sample_idx}:")
        print(f"  Voxels shape: {voxels.shape}")
        print(f"  Poses (GT deltas) shape: {poses.shape}")
        print(f"  First pose delta: {poses[0]}")
        print(f"  First pose translation magnitude: {torch.norm(poses[0, :3]):.6e} m")
        print(f"  First pose rotation magnitude: {torch.norm(poses[0, 3:]):.6e} rad")
        print()
    except Exception as e:
        print(f"Sample {sample_idx}: ERROR - {e}")
        print()

print("=" * 80)
print("If translation magnitudes are ~5e-5 to 5e-4 m, the fix is working!")
print("=" * 80)
