#!/usr/bin/env python3
"""
Diagnose why evaluation RMSE varies wildly between datasets.
Test per-frame error vs trajectory error.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(__file__))

from dataloader import MVSECDataset
from VOnetwork import VONet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANS_SCALE_FACTOR = 1000.0

# Load model (use latest checkpoint if available)
checkpoint_path = './checkpoints/vo_model_ep30.pth'
if not os.path.exists(checkpoint_path):
    # Try to find any checkpoint
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if checkpoints:
        checkpoint_path = os.path.join('checkpoints', sorted(checkpoints)[-1])
    else:
        print("ERROR: No checkpoint found!")
        sys.exit(1)

print(f"Using checkpoint: {checkpoint_path}\n")

model = VONet(input_channels=10).to(DEVICE)
state = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(state)
model.eval()

# Test datasets
test_datasets = [
    {'name': 'flying1', 'data': './data/indoor_flying1_data.hdf5', 'gt': './data/indoor_flying1_gt.hdf5'},
    {'name': 'flying3', 'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
    {'name': 'flying4', 'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'},
]

def convert_delta_to_matrix(delta):
    """Convert 6D delta to 4x4 transformation matrix."""
    t = delta[:3].astype(np.float64)
    r = R.from_euler('xyz', delta[3:], degrees=False)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r.as_matrix()
    T[:3, 3] = t
    return T

def compute_trajectory(pose_deltas):
    """Integrate pose deltas into absolute trajectory."""
    trajectory = [np.eye(4, dtype=np.float64)]
    current = np.eye(4, dtype=np.float64)
    for delta_T in pose_deltas:
        current = current @ delta_T
        trajectory.append(current)
    return np.array(trajectory)

def align_trajectories_umeyama(pred_xyz, gt_xyz):
    """Umeyama alignment (Sim3)."""
    N = pred_xyz.shape[0]
    if N < 3:
        return pred_xyz, 1.0
    
    centroid_pred = np.mean(pred_xyz, axis=0)
    centroid_gt = np.mean(gt_xyz, axis=0)
    
    X = pred_xyz - centroid_pred
    Y = gt_xyz - centroid_gt
    
    X_var = np.sum(X**2)
    if X_var < 1e-6:
        return pred_xyz, 1.0
    
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1, 1, d])
    R_mat = V @ D @ U.T
    
    c = np.trace(H @ R_mat) / X_var
    t = centroid_gt - c * R_mat @ centroid_pred
    
    pred_aligned = (c * R_mat @ pred_xyz.T).T + t
    return pred_aligned, c

print("=" * 100)
print("EVALUATION DIAGNOSTIC")
print("=" * 100)

for dataset_cfg in test_datasets:
    print(f"\n\nTesting {dataset_cfg['name']}:")
    print("-" * 100)
    
    ds = MVSECDataset(
        data_path=dataset_cfg['data'],
        gt_path=dataset_cfg['gt'],
        seq_len=10,
        crop_params={'H': 260, 'W': 346, 'B': 5},
        use_stereo=True,
        use_calib=True,
        calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'
    )
    
    loader = DataLoader(ds, batch_size=1, shuffle=False, drop_last=False)
    
    all_preds = []
    all_gt = []
    
    # Collect predictions
    with torch.no_grad():
        for i, (voxels, imu_feats, imu_ts, targets) in enumerate(loader):
            if i >= min(300, len(ds)):  # Limit to 300 sequences for speed
                break
            
            voxels = voxels.to(DEVICE)
            imu_feats = imu_feats.to(DEVICE)
            preds = model(voxels, imu_feats)
            
            # Descale
            preds_np = preds.cpu().numpy().reshape(-1, 6)
            preds_np[:, :3] = preds_np[:, :3] / TRANS_SCALE_FACTOR
            
            targets_np = targets.numpy().reshape(-1, 6)
            
            all_preds.append(preds_np)
            all_gt.append(targets_np)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)
    
    # Compute per-frame error
    trans_error = np.linalg.norm(all_preds[:, :3] - all_gt[:, :3], axis=1)
    rot_error = np.linalg.norm(all_preds[:, 3:] - all_gt[:, 3:], axis=1)
    
    print(f"Per-frame statistics ({all_preds.shape[0]} frames):")
    print(f"  Translation error - RMSE: {np.sqrt(np.mean(trans_error**2)):.6f}m, MAE: {np.mean(trans_error):.6f}m")
    print(f"  Rotation error - RMSE: {np.sqrt(np.mean(rot_error**2)):.6f} rad, MAE: {np.mean(rot_error):.6f} rad")
    print(f"  Translation range: [{trans_error.min():.6f}, {trans_error.max():.6f}] m")
    print(f"  Rotation range: [{rot_error.min():.6f}, {rot_error.max():.6f}] rad")
    
    # Convert to trajectory and compute trajectory RMSE
    pred_T = [convert_delta_to_matrix(d) for d in all_preds]
    gt_T = [convert_delta_to_matrix(d) for d in all_gt]
    
    pred_traj = compute_trajectory(pred_T)
    gt_traj = compute_trajectory(gt_T)
    
    pred_xyz = pred_traj[:, :3, 3].astype(np.float32)
    gt_xyz = gt_traj[:, :3, 3].astype(np.float32)
    
    # Without alignment
    unaligned_error = np.linalg.norm(pred_xyz - gt_xyz, axis=1)
    print(f"\nTrajectory error (UNALIGNED):")
    print(f"  RMSE: {np.sqrt(np.mean(unaligned_error**2)):.6f}m")
    print(f"  MAE: {np.mean(unaligned_error):.6f}m")
    
    # With alignment
    pred_xyz_aligned, scale = align_trajectories_umeyama(pred_xyz.astype(np.float64), gt_xyz.astype(np.float64))
    aligned_error = np.linalg.norm(pred_xyz_aligned - gt_xyz, axis=1)
    print(f"\nTrajectory error (ALIGNED with scale={scale:.4f}):")
    print(f"  RMSE: {np.sqrt(np.mean(aligned_error**2)):.6f}m")
    print(f"  MAE: {np.mean(aligned_error):.6f}m")
    
    # Analyze prediction scale bias
    pred_magnitudes = np.linalg.norm(np.diff(pred_xyz, axis=0), axis=1)
    gt_magnitudes = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    print(f"\nStep magnitude analysis:")
    print(f"  Pred steps - mean: {np.mean(pred_magnitudes):.6f}m, std: {np.std(pred_magnitudes):.6f}m")
    print(f"  GT steps - mean: {np.mean(gt_magnitudes):.6f}m, std: {np.std(gt_magnitudes):.6f}m")
    print(f"  Ratio (pred/gt): {np.mean(pred_magnitudes) / np.mean(gt_magnitudes):.4f}")

print("\n" + "=" * 100)
print("DIAGNOSTIC COMPLETE")
print("=" * 100)
