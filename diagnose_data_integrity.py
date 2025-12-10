#!/usr/bin/env python3
"""
STEP 1: Verify Mono Data Integrity

Load a single sequence and inspect:
1. Voxel grid timestamps (t_start, t_end)
2. GT pose delta timestamps (t0, t1)
3. Verify they match
4. Manually compute and inspect GT pose deltas
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataloader import MVSECDataset
import numpy as np

# Test on indoor_flying4 dataset (smaller, faster)
DATA_PATH = './data/indoor_flying4_data.hdf5'
GT_PATH = './data/indoor_flying4_gt.hdf5'
SEQ_IDX = 50  # Random sequence to inspect

print("=" * 80)
print("DATA INTEGRITY DIAGNOSTIC - MONO BRANCH")
print("=" * 80)
print(f"Dataset: {DATA_PATH}")
print(f"Sequence Index: {SEQ_IDX}")
print()

# Load dataset (mono, no calib)
ds = MVSECDataset(
    data_path=DATA_PATH,
    gt_path=GT_PATH,
    seq_len=5,
    crop_params={'H': 260, 'W': 346, 'B': 5},
    use_stereo=False,
    use_calib=False
)

print(f"Dataset length: {len(ds)}")
print(f"Valid index: {SEQ_IDX < len(ds)}")
print()

# Manually replicate the __getitem__ logic to inspect internals
start_frame = ds.valid_indices[SEQ_IDX]
cam = 'left'

print(f"Start frame index: {start_frame}")
print()

# Iterate through sequence and log timestamps
print("SEQUENCE STEP-BY-STEP ANALYSIS")
print("-" * 80)

for step_i in range(ds.seq_len):
    frame_idx = start_frame + step_i

    # Get event slice boundaries
    idx_start = int(ds.event_indices_dict[cam][frame_idx])
    idx_end = int(ds.event_indices_dict[cam][frame_idx + 1])

    # Get timestamps used for voxel grid
    t_start_voxel = ds.event_timestamps_dict[cam][frame_idx]
    t_end_voxel = ds.event_timestamps_dict[cam][frame_idx + 1]

    # Get event slice and extract pose timestamps
    event_slice = ds.events_dict[cam][idx_start:idx_end]

    if event_slice.shape[0] > 0:
        t0_pose = event_slice[0, 2]
        t1_pose = event_slice[-1, 2]
        num_events = event_slice.shape[0]
    else:
        t0_pose = None
        t1_pose = None
        num_events = 0

    # Compute GT pose delta
    if event_slice.shape[0] > 0:
        if ds.pose.ndim == 3 and ds.pose.shape[1:] == (4, 4):
            t0_idx = frame_idx
            t1_idx = frame_idx + 1
            gt_delta = ds._compute_relative_pose(t0_idx, t1_idx)
        else:
            gt_delta = ds._compute_relative_pose(t0_pose, t1_pose)
    else:
        gt_delta = np.zeros(6, dtype=np.float32)

    print(f"\nStep {step_i}:")
    print(f"  Frame index: {frame_idx}")
    print(f"  Event slice: indices [{idx_start}, {idx_end}), count={num_events}")
    print(f"  Voxel grid timestamps:  t_start={t_start_voxel:.6f}, t_end={t_end_voxel:.6f}")
    if t0_pose is not None:
        print(f"  Event slice timestamps: t0={t0_pose:.6f}, t1={t1_pose:.6f}")
        print(f"  Timestamp match (voxel vs events)?")
        print(f"    t_start voxel vs t0 event: {t_start_voxel:.6f} vs {t0_pose:.6f} (diff={abs(t_start_voxel - t0_pose):.6e})")
        print(f"    t_end voxel vs t1 event:   {t_end_voxel:.6f} vs {t1_pose:.6f} (diff={abs(t_end_voxel - t1_pose):.6e})")
    else:
        print(f"  [NO EVENTS IN SLICE - GT delta will be zero]")

    print(f"  GT pose delta: {gt_delta}")
    print(f"    Translation (m):  [{gt_delta[0]:.6e}, {gt_delta[1]:.6e}, {gt_delta[2]:.6e}]")
    print(f"    Rotation (rad):   [{gt_delta[3]:.6e}, {gt_delta[4]:.6e}, {gt_delta[5]:.6e}]")

    # Check magnitudes
    trans_mag = np.linalg.norm(gt_delta[:3])
    rot_mag = np.linalg.norm(gt_delta[3:])
    print(f"    Translation magnitude: {trans_mag:.6e} m")
    print(f"    Rotation magnitude: {rot_mag:.6e} rad")

print()
print("=" * 80)
print("OBSERVATIONS & CHECKS")
print("=" * 80)

# Summary checks
print("\n1. Timestamp Alignment:")
print("   - Do voxel grid timestamps (t_start, t_end) match event slice timestamps (t0, t1)?")
print("   - If NOT, your network input is misaligned with your label!")

print("\n2. GT Pose Magnitude:")
print("   - Are translation deltas in the range ±0.001 to ±0.1 m?")
print("   - Are rotation deltas in the range ±0.001 to ±0.1 rad?")
print("   - If magnitudes are too small or too large, there may be a scale issue.")

print("\n3. Pose Format:")
pose_shape = ds.pose.shape
pose_dtype = ds.pose.dtype
print(f"   - Pose array shape: {pose_shape}")
print(f"   - Pose array dtype: {pose_dtype}")
if ds.pose.ndim == 3 and ds.pose.shape[1:] == (4, 4):
    print("   - Format: (N, 4, 4) — Frame-indexed pose matrices")
elif ds.pose.ndim == 2 and ds.pose.shape[1] == 8:
    print("   - Format: (N, 8) — Unstructured [t, x, y, z, qx, qy, qz, qw]")
elif ds.pose.dtype.names:
    print(f"   - Format: Structured array with fields: {ds.pose.dtype.names}")
else:
    print(f"   - Format: UNKNOWN (unexpected)")

print("\n" + "=" * 80)
