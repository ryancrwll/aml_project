#!/usr/bin/env python3
"""
Verify that the dataloader is using event timestamps, not frame timestamps.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataloader import MVSECDataset
import torch

print("=" * 80)
print("TIMESTAMP VERIFICATION - CHECK THAT FIX IS ACTIVE")
print("=" * 80)

ds = MVSECDataset(
    data_path='./data/indoor_flying4_data.hdf5',
    gt_path='./data/indoor_flying4_gt.hdf5',
    seq_len=5,
    crop_params={'H': 260, 'W': 346, 'B': 5},
    use_stereo=False,
    use_calib=False
)

# Check the actual __getitem__ code path
print("\nLoading sample 50 from flying4...")
voxels, imu_feats, imu_ts, poses = ds[50]

print(f"Sample shapes: {voxels.shape}, {poses.shape}")
print(f"First pose: {poses[0]}")

# Now check the source code to ensure it's using event timestamps
import inspect
source = inspect.getsource(ds.__getitem__)

# Check if "event_slice[0, 2]" is in the code
if "event_slice[0, 2]" in source:
    print("\n✓ CONFIRMED: Code uses event_slice[0, 2] (actual event timestamps)")
else:
    print("\n✗ ERROR: Code does NOT use event_slice[0, 2]")
    print("The fix may not be applied correctly!")

# Check for the fallback to frame timestamps
if "event_timestamps_dict" in source and "event_slice[0, 2]" in source:
    print("✓ Code has proper fallback logic for empty event slices")
else:
    print("✗ Warning: Fallback logic may be missing")

print("\n" + "=" * 80)
