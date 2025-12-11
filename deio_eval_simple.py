"""Simple DEIO evaluation: load checkpoint, run inference, plot trajectory."""
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, os.path.dirname(__file__))

from vo_deio_net import DEIONet, DifferentiableBundleAdjustment
from deio_data import DEIODataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA_FILE = './data/indoor_flying1_data.hdf5'
GT_FILE = './data/indoor_flying1_gt.hdf5'
CHECKPOINT_PATH = './checkpoints/deio_model_ep30.pth'
SEQ_LEN = 10
BATCH_SIZE = 1
VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
USE_STEREO = False
USE_CALIB = True
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'

if not os.path.exists(CHECKPOINT_PATH):
    print(f"Checkpoint not found: {CHECKPOINT_PATH}")
    sys.exit(1)

device = torch.device(DEVICE)

# Load model
input_channels = VOXEL_PARAMS['B']
model = DEIONet(input_channels=input_channels, imu_state_dim=6).to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()

dba = DifferentiableBundleAdjustment(state_dim=16, seq_len=SEQ_LEN, dba_param_dim=model.dba_param_dim).to(device)
dba.eval()

print(f"Model loaded from {CHECKPOINT_PATH}")

# Load dataset
ds = DEIODataset(
    data_path=DATA_FILE, gt_path=GT_FILE, seq_len=SEQ_LEN,
    crop_params=VOXEL_PARAMS, use_stereo=USE_STEREO, use_calib=USE_CALIB,
    calib_path=CALIB_PATH, imu_state_dim=6, gt_state_dim=16
)
loader = DataLoader(ds, batch_size=1, shuffle=False)

print(f"Dataset loaded: {len(ds)} sequences")

# Run inference and collect positions
pred_all = []
gt_all = []

# Start trajectories at identity
pred_traj_mat = [np.eye(4, dtype=np.float64)]
gt_traj_mat = [np.eye(4, dtype=np.float64)]

with torch.no_grad():
    for voxels, imu_raw_state, gt_full_state in tqdm(loader):
        if voxels.size(1) != SEQ_LEN:
            continue

        voxels = voxels.to(device)
        imu_raw_state = imu_raw_state.to(device)
        gt_full_state = gt_full_state.to(device)

        # Forward pass to get DBA parameters (learned deltas/corrections)
        dba_params = model(voxels, imu_raw_state)  # (1, S, 32)

        # Use DBA params as learned pose deltas: map first 6 dims to SE(3) delta
        # (rest are learned noise scales or other params)
        for s in range(SEQ_LEN):
            params_s = dba_params[0, s, :6].cpu().numpy()  # (6,) - use first 6 as pose delta

            # Interpret as: [tx, ty, tz, rx, ry, rz] in small angle approx
            delta_t = params_s[:3]
            delta_r_euler = params_s[3:]

            # Build delta SE(3)
            delta_R = R.from_euler('xyz', delta_r_euler).as_matrix()
            delta_T = np.eye(4)
            delta_T[:3, :3] = delta_R
            delta_T[:3, 3] = delta_t

            # Integrate predicted trajectory
            pred_traj_mat[-1] @ delta_T  # dummy op to avoid warning
            pred_traj_mat.append(pred_traj_mat[-1] @ delta_T)

            # Use GT state positions as GT trajectory
            gt_p = gt_full_state[0, s, :3].cpu().numpy()
            gt_all.append(gt_p)

# Collect predicted positions from integrated trajectory
for mat in pred_traj_mat[1:]:  # skip identity
    pred_all.append(mat[:3, 3])

pred_traj = np.array(pred_all)
gt_traj = np.array(gt_all)

print(f"Total points: {pred_traj.shape[0]}")
print(f"Pred range: {pred_traj.min(axis=0)} to {pred_traj.max(axis=0)}")
print(f"GT range: {gt_traj.min(axis=0)} to {gt_traj.max(axis=0)}")

# Compute metrics
errors = np.linalg.norm(pred_traj - gt_traj, axis=1)
rmse = np.sqrt(np.mean(errors**2))
print(f"RMSE: {rmse:.6f} m")

# Plot
fig, ax = plt.subplots(figsize=(12, 10))
ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'b-', linewidth=1, label='Ground Truth', alpha=0.7)
ax.scatter(gt_traj[::20, 0], gt_traj[::20, 1], c='blue', s=10, alpha=0.4)
ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'r--', linewidth=1, label='Predicted', alpha=0.7)
ax.scatter(pred_traj[::20, 0], pred_traj[::20, 1], c='red', s=10, marker='x', alpha=0.4)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title(f'DEIO Trajectory (RMSE: {rmse:.6f} m)')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('deio_trajectory_simple.png', dpi=150)
print(f"Plot saved to deio_trajectory_simple.png")
