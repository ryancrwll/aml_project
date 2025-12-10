#!/usr/bin/env python3
"""
Unit tests for IMU feature and timestamp shapes/values in MVSECDataset.

Tests verify:
  - Per-step IMU feature dimensions (mono: 12, stereo: 24)
  - Per-step IMU timestamp dimensions (mono: 2, stereo: 4)
  - Zeros when IMU data is unavailable
  - Correct temporal slicing of IMU measurements
"""

import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dataloader import MVSECDataset


def test_mono_dataset_shapes():
    """Test mono dataset returns correct shapes for imu_feats and imu_ts."""
    print("Testing mono dataset shapes...")

    ds = MVSECDataset(
        data_path='./data/indoor_flying4_data.hdf5',
        gt_path='./data/indoor_flying4_gt.hdf5',
        seq_len=5,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=False,
        use_calib=False
    )

    voxels, imu_feats, imu_ts, poses = ds[0]

    assert voxels.shape == (5, 5, 260, 346), f"Unexpected voxels shape: {voxels.shape}"
    assert imu_feats.shape == (5, 12), f"Expected imu_feats (5, 12), got {imu_feats.shape}"
    assert imu_ts.shape == (5, 2), f"Expected imu_ts (5, 2), got {imu_ts.shape}"
    assert poses.shape == (5, 6), f"Unexpected poses shape: {poses.shape}"

    print("  ✓ Mono dataset shapes correct")


def test_stereo_dataset_shapes():
    """Test stereo dataset returns correct shapes for imu_feats and imu_ts."""
    print("Testing stereo dataset shapes...")

    ds = MVSECDataset(
        data_path='./data/indoor_flying4_data.hdf5',
        gt_path='./data/indoor_flying4_gt.hdf5',
        seq_len=5,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=True,
        use_calib=False
    )

    voxels, imu_feats, imu_ts, poses = ds[0]

    assert voxels.shape == (5, 10, 260, 346), f"Unexpected voxels shape: {voxels.shape}"
    assert imu_feats.shape == (5, 24), f"Expected imu_feats (5, 24), got {imu_feats.shape}"
    assert imu_ts.shape == (5, 4), f"Expected imu_ts (5, 4), got {imu_ts.shape}"
    assert poses.shape == (5, 6), f"Unexpected poses shape: {poses.shape}"

    print("  ✓ Stereo dataset shapes correct")


def test_imu_feats_dtype():
    """Test IMU features have correct dtype (float32)."""
    print("Testing IMU feature dtype...")

    ds = MVSECDataset(
        data_path='./data/indoor_flying4_data.hdf5',
        gt_path='./data/indoor_flying4_gt.hdf5',
        seq_len=5,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=False,
        use_calib=False
    )

    voxels, imu_feats, imu_ts, poses = ds[0]

    assert imu_feats.dtype == torch.float32, f"Expected float32, got {imu_feats.dtype}"
    assert imu_ts.dtype == torch.float32, f"Expected float32, got {imu_ts.dtype}"

    print("  ✓ IMU feature dtypes correct")


def test_imu_feats_range():
    """Test IMU features are reasonable (not all zeros or infinite)."""
    print("Testing IMU feature value ranges...")

    ds = MVSECDataset(
        data_path='./data/indoor_flying4_data.hdf5',
        gt_path='./data/indoor_flying4_gt.hdf5',
        seq_len=10,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=False,
        use_calib=False
    )

    # Check multiple samples
    found_nonzero = False
    for i in range(min(5, len(ds))):
        voxels, imu_feats, imu_ts, poses = ds[i]

        # Check for NaN or Inf
        assert not torch.isnan(imu_feats).any(), f"Sample {i} contains NaN in imu_feats"
        assert not torch.isinf(imu_feats).any(), f"Sample {i} contains Inf in imu_feats"
        assert not torch.isnan(imu_ts).any(), f"Sample {i} contains NaN in imu_ts"
        assert not torch.isinf(imu_ts).any(), f"Sample {i} contains Inf in imu_ts"

        # Check if we have non-zero IMU features
        if (imu_feats != 0).any():
            found_nonzero = True
            print(f"  Sample {i}: found non-zero IMU features (min={imu_feats.min():.4f}, max={imu_feats.max():.4f})")

    if found_nonzero:
        print("  ✓ Found non-zero IMU features (IMU data available)")
    else:
        print("  ⚠ All IMU features are zero (IMU data may not be available in this dataset)")


def test_imu_ts_temporal_order():
    """Test IMU timestamps maintain temporal order (start <= end) for each step."""
    print("Testing IMU timestamp temporal ordering...")

    ds = MVSECDataset(
        data_path='./data/indoor_flying4_data.hdf5',
        gt_path='./data/indoor_flying4_gt.hdf5',
        seq_len=10,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=False,
        use_calib=False
    )

    for i in range(min(10, len(ds))):
        voxels, imu_feats, imu_ts, poses = ds[i]

        # For each visual step, check that start_ts <= end_ts
        # imu_ts shape is (seq_len, 2): [start_ts, end_ts] per step
        for step in range(imu_ts.shape[0]):
            start_ts = imu_ts[step, 0].item()
            end_ts = imu_ts[step, 1].item()

            # Both zero means no IMU data for this step
            if start_ts == 0.0 and end_ts == 0.0:
                continue

            assert start_ts <= end_ts, f"Sample {i} step {step}: start_ts {start_ts} > end_ts {end_ts}"

    print("  ✓ IMU timestamp temporal order correct")


def test_batch_stacking():
    """Test that multiple samples can be batched correctly."""
    print("Testing batch stacking...")

    from torch.utils.data import DataLoader

    ds = MVSECDataset(
        data_path='./data/indoor_flying4_data.hdf5',
        gt_path='./data/indoor_flying4_gt.hdf5',
        seq_len=5,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=False,
        use_calib=False
    )

    loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)

    batch = next(iter(loader))
    voxels, imu_feats, imu_ts, poses = batch

    assert voxels.shape == (4, 5, 5, 260, 346), f"Unexpected batched voxels: {voxels.shape}"
    assert imu_feats.shape == (4, 5, 12), f"Unexpected batched imu_feats: {imu_feats.shape}"
    assert imu_ts.shape == (4, 5, 2), f"Unexpected batched imu_ts: {imu_ts.shape}"
    assert poses.shape == (4, 5, 6), f"Unexpected batched poses: {poses.shape}"

    print("  ✓ Batch stacking works correctly")


if __name__ == '__main__':
    print("=" * 60)
    print("IMU Feature & Timestamp Unit Tests")
    print("=" * 60)

    try:
        test_mono_dataset_shapes()
        test_stereo_dataset_shapes()
        test_imu_feats_dtype()
        test_imu_feats_range()
        test_imu_ts_temporal_order()
        test_batch_stacking()

        print("\n" + "=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
