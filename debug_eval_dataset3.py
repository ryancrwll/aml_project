#!/usr/bin/env python3
"""
Debug evaluation pipeline step-by-step on dataset 3.
Check: model output, descaling, trajectory integration, and alignment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MVSECDataset
from VOnetwork import VONet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = './checkpoints/vo_model_ep30.pth'
TRANS_SCALE_FACTOR = 1000.0

print("=" * 80)
print("STEP-BY-STEP EVALUATION DEBUG ON DATASET 3")
print("=" * 80)

# Check if checkpoint exists
if not os.path.exists(CHECKPOINT_PATH):
    print(f"\nERROR: Checkpoint not found at {CHECKPOINT_PATH}")
    sys.exit(1)

# Load model
model = VONet(input_channels=10).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()
print(f"\n✓ Model loaded from {CHECKPOINT_PATH}")

# Load dataset 3
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
print(f"✓ Dataset loaded. Size: {len(ds)} sequences")

# Step 1: Check first batch
print(f"\n--- STEP 1: INSPECT FIRST BATCH ---")
first_batch = next(iter(loader))
voxels, imu_feats, imu_ts, targets = first_batch
print(f"Voxels shape: {voxels.shape}")
print(f"Targets (GT) shape: {targets.shape}")
print(f"First GT pose delta: {targets[0, 0, :].numpy()}")
print(f"GT translation mag: {np.linalg.norm(targets[0, 0, :3].numpy()):.6e} m")
print(f"GT rotation mag: {np.linalg.norm(targets[0, 0, 3:].numpy()):.6e} rad")

# Step 2: Model forward pass
print(f"\n--- STEP 2: MODEL FORWARD PASS ---")
voxels = voxels.to(DEVICE)
imu_feats = imu_feats.to(DEVICE)
with torch.no_grad():
    preds = model(voxels, imu_feats)
print(f"Predictions shape: {preds.shape}")
print(f"First prediction (scaled): {preds[0, 0, :].cpu().numpy()}")

# Step 3: Check descaling
print(f"\n--- STEP 3: DESCALING ---")
preds_np = preds.cpu().numpy().reshape(-1, 6)
preds_descaled = preds_np.copy()
preds_descaled[:, :3] = preds_descaled[:, :3] / TRANS_SCALE_FACTOR

print(f"First pred (before descale): {preds_np[0, :3]}")
print(f"First pred (after descale): {preds_descaled[0, :3]}")
print(f"Translation mag (descaled): {np.linalg.norm(preds_descaled[0, :3]):.6e} m")

# Step 4: Check scale factor
print(f"\n--- STEP 4: SCALE FACTOR CHECK ---")
targets_np = targets.cpu().numpy().reshape(-1, 6)
print(f"First GT (unscaled): {targets_np[0, :3]}")

# Compute mean magnitude ratio
gt_trans_mags = np.linalg.norm(targets_np[:, :3], axis=1)
pred_trans_mags = np.linalg.norm(preds_descaled[:, :3], axis=1)

valid_gt = gt_trans_mags > 1e-6
if valid_gt.any():
    scale_factors = pred_trans_mags[valid_gt] / gt_trans_mags[valid_gt]
    print(f"Scale factors (pred/gt):")
    print(f"  Min: {scale_factors.min():.4f}, Max: {scale_factors.max():.4f}")
    print(f"  Mean: {scale_factors.mean():.4f}, Median: {np.median(scale_factors):.4f}")

# Step 5: Run full inference
print(f"\n--- STEP 5: FULL INFERENCE (first 50 sequences) ---")
all_preds = []
all_gt = []

with torch.no_grad():
    for i, (voxels, imu_feats, imu_ts, targets) in enumerate(tqdm(loader)):
        if i >= 50:
            break
        
        voxels = voxels.to(DEVICE)
        imu_feats = imu_feats.to(DEVICE)
        preds = model(voxels, imu_feats)
        
        preds_np = preds.cpu().numpy().reshape(-1, 6)
        preds_np[:, :3] = preds_np[:, :3] / TRANS_SCALE_FACTOR
        
        targets_np = targets.numpy().reshape(-1, 6)
        
        all_preds.append(preds_np)
        all_gt.append(targets_np)

all_preds = np.concatenate(all_preds, axis=0)
all_gt = np.concatenate(all_gt, axis=0)

print(f"Collected {all_preds.shape[0]} poses")

# Step 6: Per-frame error
print(f"\n--- STEP 6: PER-FRAME ERRORS ---")
trans_error = np.linalg.norm(all_preds[:, :3] - all_gt[:, :3], axis=1)
rot_error = np.linalg.norm(all_preds[:, 3:] - all_gt[:, 3:], axis=1)

print(f"Translation error (m):")
print(f"  RMSE: {np.sqrt(np.mean(trans_error**2)):.6e}")
print(f"  MAE: {np.mean(np.abs(trans_error)):.6e}")
print(f"  Min: {trans_error.min():.6e}, Max: {trans_error.max():.6e}")

print(f"Rotation error (rad):")
print(f"  RMSE: {np.sqrt(np.mean(rot_error**2)):.6e}")
print(f"  MAE: {np.mean(np.abs(rot_error)):.6e}")
print(f"  Min: {rot_error.min():.6e}, Max: {rot_error.max():.6e}")

# Step 7: Trajectory integration
print(f"\n--- STEP 7: TRAJECTORY INTEGRATION ---")

def convert_delta_to_matrix(delta):
    t = delta[:3].astype(np.float64)
    r = R.from_euler('xyz', delta[3:], degrees=False)
    R_matrix = r.as_matrix().astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_matrix
    T[:3, 3] = t
    return T

def compute_trajectory(pose_deltas):
    trajectory = [np.eye(4, dtype=np.float64)]
    current_pose = np.eye(4, dtype=np.float64)
    for delta_T in pose_deltas:
        current_pose = current_pose @ delta_T
        trajectory.append(current_pose)
    return np.array(trajectory)

pred_T_deltas = [convert_delta_to_matrix(d) for d in all_preds]
gt_T_deltas = [convert_delta_to_matrix(d) for d in all_gt]

pred_trajectory = compute_trajectory(pred_T_deltas)
gt_trajectory = compute_trajectory(gt_T_deltas)

print(f"Pred trajectory shape: {pred_trajectory.shape}")
print(f"GT trajectory shape: {gt_trajectory.shape}")

pred_xyz = pred_trajectory[:, :3, 3].astype(np.float32)
gt_xyz = gt_trajectory[:, :3, 3].astype(np.float32)

# Step 8: Alignment
print(f"\n--- STEP 8: TRAJECTORY ALIGNMENT ---")

def align_trajectories_umeyama(pred_xyz, gt_xyz):
    N = pred_xyz.shape[0]
    centroid_pred = np.mean(pred_xyz, axis=0)
    centroid_gt = np.mean(gt_xyz, axis=0)
    X = pred_xyz - centroid_pred
    Y = gt_xyz - centroid_gt
    X_var_sum = np.sum(X**2)
    
    if X_var_sum < 1e-6:
        return pred_xyz, 1.0
    
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])
    R_matrix = V @ D @ U.T
    c = np.trace(H @ R_matrix) / X_var_sum
    t = centroid_gt - c * R_matrix @ centroid_pred
    
    pred_aligned = (c * R_matrix @ pred_xyz.T).T + t
    return pred_aligned, c

pred_xyz_aligned, scale = align_trajectories_umeyama(pred_xyz.astype(np.float64), gt_xyz.astype(np.float64))

print(f"Scale factor: {scale:.4f}")

# Step 9: Final RMSE
print(f"\n--- STEP 9: ALIGNED TRAJECTORY RMSE ---")
rmse = np.sqrt(np.mean(np.linalg.norm(pred_xyz_aligned - gt_xyz, axis=1)**2))
print(f"Trajectory RMSE (aligned): {rmse:.4f} m")

print("\n" + "=" * 80)
