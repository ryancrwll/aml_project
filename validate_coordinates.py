#!/usr/bin/env python3
"""
Validates that predicted and GT coordinates are in the same frame.
Run this after training to check if X and Y are swapped.
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))

from dataloader import MVSECDataset
from VOnetwork import VONet

# Configuration
DATA_FILE = './data/indoor_flying4_data.hdf5'
GT_FILE = './data/indoor_flying4_gt.hdf5'
CHECKPOINT_PATH = './checkpoints/vo_model_ep50.pth'
SEQ_LEN = 5
BATCH_SIZE = 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
OUTPUT_DIM = 6
TRANS_SCALE_FACTOR = 1000.0

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

def test_coordinates():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    device = torch.device(DEVICE)
    model = VONet(input_channels=VOXEL_PARAMS['B']).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"Model loaded from {CHECKPOINT_PATH}.")

    dataset = MVSECDataset(
        data_path=DATA_FILE,
        gt_path=GT_FILE,
        seq_len=SEQ_LEN,
        crop_params=VOXEL_PARAMS
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    all_pred_deltas = []
    all_gt_deltas = []

    print("Running inference on first 100 samples...")
    with torch.no_grad():
        for idx, (voxels, targets) in enumerate(loader):
            if idx >= 100:
                break
            if voxels.size(1) != SEQ_LEN:
                continue

            voxels = voxels.to(device)
            preds = model(voxels)

            # Rescale predictions
            preds_np = preds.cpu().numpy().reshape(-1, OUTPUT_DIM)
            preds_np[:, :3] = preds_np[:, :3] / TRANS_SCALE_FACTOR

            all_pred_deltas.append(preds_np)
            all_gt_deltas.append(targets.numpy().reshape(-1, OUTPUT_DIM))

    pred_deltas = np.concatenate(all_pred_deltas, axis=0)
    gt_deltas = np.concatenate(all_gt_deltas, axis=0)

    # Integrate trajectories
    pred_T_deltas = [convert_delta_to_matrix(d) for d in pred_deltas]
    gt_T_deltas = [convert_delta_to_matrix(d) for d in gt_deltas]

    pred_trajectory = compute_trajectory(pred_T_deltas)
    gt_trajectory = compute_trajectory(gt_T_deltas)

    pred_xyz = pred_trajectory[:, :3, 3].astype(np.float32)
    gt_xyz = gt_trajectory[:, :3, 3].astype(np.float32)

    # Compute correlation for each axis independently
    print("\n--- Coordinate Axis Analysis ---")
    for axis, name in enumerate(['X', 'Y', 'Z']):
        corr = np.corrcoef(pred_xyz[:, axis], gt_xyz[:, axis])[0, 1]
        print(f"{name} correlation (pred vs GT): {corr:.4f}")

    # Test X-Y swap
    pred_xyz_swapped = pred_xyz.copy()
    pred_xyz_swapped[:, [0, 1]] = pred_xyz_swapped[:, [1, 0]]

    for axis, name in enumerate(['X', 'Y', 'Z']):
        corr = np.corrcoef(pred_xyz_swapped[:, axis], gt_xyz[:, axis])[0, 1]
        print(f"{name} correlation (pred SWAPPED vs GT): {corr:.4f}")

    # Compute RMSE for both
    rmse_original = np.sqrt(np.mean(np.linalg.norm(pred_xyz - gt_xyz, axis=1)**2))
    rmse_swapped = np.sqrt(np.mean(np.linalg.norm(pred_xyz_swapped - gt_xyz, axis=1)**2))

    print(f"\nRMSE (original): {rmse_original:.6f}")
    print(f"RMSE (swapped):  {rmse_swapped:.6f}")

    if rmse_swapped < rmse_original * 0.95:
        print("\n⚠️  ALERT: X and Y appear to be SWAPPED in predictions!")
        print("   Fix: Swap the output order in VOnetwork.py or dataloader.py")
        pred_xyz = pred_xyz_swapped
    else:
        print("\n✓ Coordinate order appears correct.")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 2D trajectory
    ax1.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', label='GT', linewidth=2)
    ax1.plot(pred_xyz[:, 0], pred_xyz[:, 1], 'r--', label='Pred', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('2D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)
    ax1.axis('equal')

    # 3D trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], 'b-', label='GT', linewidth=2)
    ax2.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], 'r--', label='Pred', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_zlabel('Z (m)')
    ax2.set_title('3D Trajectory Comparison')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('coordinate_validation.png', dpi=100)
    print("Plot saved to coordinate_validation.png")

if __name__ == "__main__":
    test_coordinates()
