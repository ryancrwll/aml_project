import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys

# Import new DEIO modules
try:
    from vo_deio_net import DEIONet, DifferentiableBundleAdjustment
    from deio_data import DEIODataset
except ImportError:
    print("FATAL: Ensure vo_deio_net.py and deio_data.py are in the path.")
    sys.exit(1)

# --- Configuration (MUST MATCH deio_train.py) ---
DATA_FILE = './data/indoor_flying1_data.hdf5'
GT_FILE = './data/indoor_flying1_gt.hdf5'
CHECKPOINT_PATH = './checkpoints/Mono_noCalib_noIMU.pth' # Loading the best checkpoint
SEQ_LEN = 10
BATCH_SIZE = 1 # Eval must be 1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- HYPERPARAMETERS (Check against deio_train.py) ---
USE_STEREO = False       # Match your training setting
USE_IMU_DATA = False
USE_CALIB = False
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}

# DEIO Specific Parameters
IMU_RAW_STATE_DIM = 6   # Accel (3) + Gyro (3)
GT_FULL_STATE_DIM = 16  # P(3), Q(4), V(3), Ba(3), Bg(3) = 16 total

def extract_position_from_state(state_vector):
    """
    Extracts the (x, y, z) position from the 15D state vector.
    Assumes structure is P(3), Q(4), V(3), Ba(3), Bg(3).
    """
    # P is the first 3 elements (indices 0, 1, 2)
    return state_vector[:, :3]

def align_trajectories_umeyama(pred_xyz, gt_xyz):
    """
    Performs optimal alignment (rotation, translation, and scale) between two
    point sets using the Umeyama algorithm (Sim(3) alignment).
    """
    N = pred_xyz.shape[0]
    if N < 3: return pred_xyz, 1.0, np.eye(3), np.zeros(3)

    # Centroids
    centroid_pred = np.mean(pred_xyz, axis=0)
    centroid_gt = np.mean(gt_xyz, axis=0)

    # Centered points
    X = pred_xyz - centroid_pred
    Y = gt_xyz - centroid_gt

    X_var_sum = np.sum(X**2)
    EPSILON = 1e-6
    if X_var_sum < EPSILON: return pred_xyz, 1.0, np.eye(3), np.zeros(3)

    # H matrix (Cross-covariance) and SVD
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T

    # Rotation R
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])
    R_matrix = V @ D @ U.T

    # Scale c
    c = np.trace(H @ R_matrix) / X_var_sum

    # Translation t
    t = centroid_gt - c * R_matrix @ centroid_pred

    # Apply transformation
    pred_aligned = (c * R_matrix @ pred_xyz.T).T + t

    return pred_aligned, c, R_matrix, t

# --- Evaluation Function ---
def evaluate_model():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint not found at {CHECKPOINT_PATH}. Please train the DEIO model first.")
        return

    device = torch.device(DEVICE)
    print(f"--- Starting Evaluation ---")
    print(f"Mode: {'STEREO' if USE_STEREO else 'MONO'} | {'CALIBRATED' if USE_CALIB else 'UNCALIBRATED'}")
    print(f"IMU Input: {'ENABLED' if USE_IMU_DATA else 'DISABLED (Events-Only)'}")

    input_channels = VOXEL_PARAMS['B'] * 2 if USE_STEREO else VOXEL_PARAMS['B']

    # Initialize DEIONet with the IMU toggle
    model = DEIONet(
        input_channels=input_channels,
        imu_state_dim=IMU_RAW_STATE_DIM,
        use_imu=USE_IMU_DATA # <--- Updated to use config
    ).to(device)

    dba_layer = DifferentiableBundleAdjustment(state_dim=GT_FULL_STATE_DIM, seq_len=SEQ_LEN,
                                               dba_param_dim=model.dba_param_dim).to(device)

    # Load weights into the learnable DEIONet (CNN/GRU)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    except RuntimeError as e:
        print(f"\nFATAL ERROR loading weights: {e}")
        print("Tip: Check if 'input_channels' matches training (did you train with Stereo=True but eval with Stereo=False?)")
        return

    model.eval()
    dba_layer.eval()

    print(f"DEIO Model loaded from {CHECKPOINT_PATH}. Channels: {input_channels}")

    test_dataset = DEIODataset(
        data_path=DATA_FILE,
        gt_path=GT_FILE,
        seq_len=SEQ_LEN,
        crop_params=VOXEL_PARAMS,
        use_stereo=USE_STEREO,
        use_calib=USE_CALIB,
        calib_path=CALIB_PATH if USE_CALIB else None,
        imu_state_dim=IMU_RAW_STATE_DIM,
        gt_state_dim=GT_FULL_STATE_DIM
    )
    # Batch size must be 1 for sequential trajectory integration
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

    all_pred_positions = []
    all_gt_positions = []

    print("Running inference and state estimation...")
    with torch.no_grad():
        for voxels, imu_raw_state, gt_full_state in tqdm(test_loader):

            if voxels.size(1) != SEQ_LEN: continue

            voxels, imu_raw_state, gt_full_state = (
                voxels.to(device), imu_raw_state.to(device), gt_full_state.to(device)
            )

            # The model internally handles zeroing IMU if use_imu=False
            dba_params = model(voxels, imu_raw_state)

            optimized_state = dba_layer(dba_params, imu_raw_state, gt_full_state)

            # Extract positions from ALL timesteps in the sequence
            # Shape [B, S, 15] -> extract [S, 3] for positions
            pred_states_seq = optimized_state[0, :SEQ_LEN, :].cpu().numpy()  # (S, 15)
            gt_states_seq = gt_full_state[0, :SEQ_LEN, :].cpu().numpy()      # (S, 15)

            # Extract P(3) vectors for all timesteps
            pred_pos_seq = extract_position_from_state(pred_states_seq)  # (S, 3)
            gt_pos_seq = extract_position_from_state(gt_states_seq)      # (S, 3)

            # Store all positions in the sequence
            for pred_p, gt_p in zip(pred_pos_seq, gt_pos_seq):
                all_pred_positions.append(pred_p)
                all_gt_positions.append(gt_p)

    # Convert lists to NumPy arrays
    pred_xyz_unaligned = np.array(all_pred_positions, dtype=np.float32)
    gt_xyz = np.array(all_gt_positions, dtype=np.float32)

    print("Aligning trajectories...")
    # align_trajectories_umeyama expects (predicted, ground_truth) order
    pred_xyz_aligned, scale, R_matrix, t = align_trajectories_umeyama(pred_xyz_unaligned.astype(np.float64), gt_xyz.astype(np.float64))
    pred_xyz_aligned = pred_xyz_aligned.astype(np.float32)

    errors_aligned = np.linalg.norm(pred_xyz_aligned - gt_xyz, axis=1)
    rmse_trans_aligned = np.sqrt(np.mean(errors_aligned**2))

    print(f"\n--- DEIO Evaluation Metrics ---")
    print(f"Total Trajectory Steps: {gt_xyz.shape[0]} steps")
    print(gt_xyz.shape)
    print(f"Sim(3) Scale Factor: {scale:.4f}")
    print(f"Trajectory RMSE (ALIGNED): {rmse_trans_aligned:.4f} meters")

    print("\nGenerating plot...")
    print(f"GT shape: {gt_xyz.shape}, min: {gt_xyz.min(axis=0)}, max: {gt_xyz.max(axis=0)}")
    print(f"Pred aligned shape: {pred_xyz_aligned.shape}, min: {pred_xyz_aligned.min(axis=0)}, max: {pred_xyz_aligned.max(axis=0)}")

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot ground truth
    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], 'b-', linewidth=1.5, label='Ground Truth', alpha=0.7)
    ax.scatter(gt_xyz[::10, 0], gt_xyz[::10, 1], c='blue', s=20, marker='>', alpha=0.5)

    # Plot predicted
    ax.plot(pred_xyz_aligned[:, 0], pred_xyz_aligned[:, 1], 'r--', linewidth=1.5, label='Predicted (Aligned)', alpha=0.7)
    ax.scatter(pred_xyz_aligned[::10, 0], pred_xyz_aligned[::10, 1], c='red', s=20, marker='.', alpha=0.5)

    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f"DEIO Trajectory ({'Fusion' if USE_IMU_DATA else 'Events Only'}) - RMSE: {rmse_trans_aligned:.4f} m", fontsize=14)
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    out_path = 'deio_trajectory_comparison_aligned.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {out_path}")
    plt.close()

if __name__ == "__main__":
    evaluate_model()