import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import sys
import h5py
import cv2

# Import DEIO modules
try:
    from vo_deio_net import DEIONet, DifferentiableBundleAdjustment
    from deio_data import DEIODataset
except ImportError:
    print("FATAL: Ensure vo_deio_net.py and deio_data.py are in the path.")
    sys.exit(1)

# ==========================================
# CONFIGURATION
# ==========================================
DATA_FILE = './data/indoor_flying1_data.hdf5'
GT_FILE = './data/indoor_flying1_gt.hdf5'

# CHECKPOINT
CHECKPOINT_PATH = './checkpoints/stereo_calib_noIMU.pth'

# MODEL PARAMS
SEQ_LEN = 10
VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5} # Must match training (B=5)
USE_STEREO = True       # Using Stereo for both
USE_IMU_DATA = False    # Events only vs Frames only
IMU_RAW_STATE_DIM = 6
GT_FULL_STATE_DIM = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 1. FRAME-BASED STEREO VO CLASS
# ==========================================
class FrameStereoVO:
    def __init__(self):
        # MVSEC Intrinsics
        self.fx, self.fy = 224.502, 224.288
        self.cx, self.cy = 169.117, 126.965
        self.b = 0.1009
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])

        # Params
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.feature_params = dict(maxCorners=1500, qualityLevel=0.01, minDistance=10, blockSize=7)

        self.prev_img = None
        self.prev_pts = None
        self.map_3d = None
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.trajectory = []

    def process_frame(self, img_l, img_r):
        if img_l.ndim == 3: img_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)

        # 1. Initialization
        if self.prev_img is None:
            p0 = cv2.goodFeaturesToTrack(img_l, mask=None, **self.feature_params)
            if p0 is None: return False
            p1, st, _ = cv2.calcOpticalFlowPyrLK(img_l, img_r, p0, None, **self.lk_params)
            if p1 is None: return False

            good_l = p0[st==1]
            good_r = p1[st==1]
            disp = good_l[:, 0] - good_r[:, 0]
            valid = (disp > 0.5) & (disp < 100.0)

            if np.sum(valid) < 10: return False

            z = (self.fx * self.b) / disp[valid]
            x = (good_l[valid, 0] - self.cx) * z / self.fx
            y = (good_l[valid, 1] - self.cy) * z / self.fy

            self.map_3d = np.stack([x, y, z], axis=1)
            self.prev_pts = good_l[valid].reshape(-1, 1, 2)
            self.prev_img = img_l
            self.trajectory.append(self.cur_t.flatten())
            return True

        # 2. Tracking
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_img, img_l, self.prev_pts, None, **self.lk_params)
        if p1 is None or np.sum(st) < 10:
            self.prev_img = None
            return False

        good_p1 = p1[st==1]
        good_3d = self.map_3d[st.flatten()==1]

        # 3. PnP
        succ, rvec, tvec, _ = cv2.solvePnPRansac(good_3d, good_p1, self.K, None,
                                                    iterationsCount=100, reprojectionError=3.0)
        if succ:
            R_mat, _ = cv2.Rodrigues(rvec)
            t_rel = -R_mat.T @ tvec
            R_rel = R_mat.T
            self.cur_t = self.cur_t + self.cur_R @ t_rel
            self.cur_R = self.cur_R @ R_rel
            self.trajectory.append(self.cur_t.flatten())

            # Replenish
            if len(good_p1) < 800:
                p_new = cv2.goodFeaturesToTrack(img_l, mask=None, **self.feature_params)
                if p_new is not None:
                    p_new_r, st_new, _ = cv2.calcOpticalFlowPyrLK(img_l, img_r, p_new, None, **self.lk_params)
                    if p_new_r is not None:
                        d = p_new[st_new==1, 0] - p_new_r[st_new==1, 0]
                        v = (d > 0.5) & (d < 100.0)
                        if np.sum(v) > 10:
                            z = (self.fx * self.b) / d[v]
                            x = (p_new[st_new==1][v, 0] - self.cx) * z / self.fx
                            y = (p_new[st_new==1][v, 1] - self.cy) * z / self.fy
                            self.map_3d = np.stack([x, y, z], axis=1)
                            self.prev_pts = p_new[st_new==1][v].reshape(-1, 1, 2)
                            self.prev_img = img_l
            else:
                self.prev_img = img_l
                self.prev_pts = good_p1.reshape(-1, 1, 2)
            return True
        else:
            self.prev_img = None
            return False

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def align_trajectories_umeyama(pred_xyz, gt_xyz):
    """ Align Sim(3) """
    # Ensure shapes match
    min_len = min(pred_xyz.shape[0], gt_xyz.shape[0])
    pred_xyz = pred_xyz[:min_len]
    gt_xyz = gt_xyz[:min_len]

    centroid_pred = np.mean(pred_xyz, axis=0)
    centroid_gt = np.mean(gt_xyz, axis=0)

    X = pred_xyz - centroid_pred
    Y = gt_xyz - centroid_gt

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    d = np.sign(np.linalg.det(V @ U.T))
    D = np.diag([1.0, 1.0, d])
    R_matrix = V @ D @ U.T

    var_pred = np.sum(X**2)
    c = np.trace(H @ R_matrix) / var_pred
    t = centroid_gt - c * R_matrix @ centroid_pred

    pred_aligned = (c * R_matrix @ pred_xyz.T).T + t
    return pred_aligned, c, R_matrix, t

def run_frame_baseline(data_path, dataset_helper):
    print("\n--- Running Frame-Based Stereo VO ---")
    vo = FrameStereoVO()

    with h5py.File(data_path, 'r') as f:
        images_l = f['davis']['left']['image_raw']
        images_r = f['davis']['right']['image_raw']
        ts = f['davis']['left']['image_raw_ts'][:]

        gt_synced = []

        for i in tqdm(range(len(images_l))):
            img_l = images_l[i]
            img_r = images_r[i]

            if vo.process_frame(img_l, img_r):
                # Use helper to get 4x4 or 7D pose correctly
                p, _ = dataset_helper._get_pose_at_time(ts[i])
                gt_synced.append(p)

    return np.array(vo.trajectory), np.array(gt_synced)

def evaluate_comparison():
    # 1. SETUP DEIO
    input_channels = VOXEL_PARAMS['B'] * 2 if USE_STEREO else VOXEL_PARAMS['B']
    model = DEIONet(input_channels=input_channels, imu_state_dim=IMU_RAW_STATE_DIM, use_imu=USE_IMU_DATA).to(DEVICE)
    dba = DifferentiableBundleAdjustment(state_dim=GT_FULL_STATE_DIM, seq_len=SEQ_LEN, dba_param_dim=model.dba_param_dim).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading DEIO Checkpoint: {CHECKPOINT_PATH}")
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        except RuntimeError:
            print("WARNING: Channel mismatch in checkpoint. Using random weights for debug.")
    else:
        print("WARNING: No checkpoint found. Using random weights.")

    model.eval()

    # Dataset for DEIO
    ds_deio = DEIODataset(
        data_path=DATA_FILE, gt_path=GT_FILE,
        seq_len=SEQ_LEN, crop_params=VOXEL_PARAMS,
        use_stereo=USE_STEREO, use_calib=False
    )
    loader = DataLoader(ds_deio, batch_size=1, shuffle=False)

    # 2. RUN DEIO
    print("\n--- Running DEIO Inference ---")
    deio_traj = []
    deio_gt = []

    # Initial Point
    current_pose = np.eye(4)
    deio_traj.append(current_pose[:3, 3].copy())

    # Get first GT point for alignment start
    p0, _ = ds_deio._get_pose_at_time(ds_deio.event_timestamps_dict['left'][0])
    deio_gt.append(p0)

    with torch.no_grad():
        for voxels, imu, gt_seq in tqdm(loader):
            voxels = voxels.to(DEVICE)
            imu = imu.to(DEVICE)
            gt_seq = gt_seq.to(DEVICE)

            dba_params = model(voxels, imu)
            opt_state = dba(dba_params, imu, gt_state=gt_seq, use_gt_init=True)

            # Position from Network
            pos = opt_state[0, -1, :3].cpu().numpy()
            deio_traj.append(pos)

            # Position from GT (Target)
            gt_pos = gt_seq[0, -1, :3].cpu().numpy()
            deio_gt.append(gt_pos)

    deio_traj = np.array(deio_traj)
    deio_gt = np.array(deio_gt)

    # 3. RUN FRAME BASELINE
    frame_traj, frame_gt = run_frame_baseline(DATA_FILE, ds_deio)

    # 4. ALIGNMENT & METRICS
    print("\n--- Calculating Metrics ---")

    # DEIO Alignment
    deio_aligned, s_d, _, _ = align_trajectories_umeyama(deio_traj, deio_gt)
    rmse_deio = np.sqrt(np.mean(np.linalg.norm(deio_aligned - deio_gt[:len(deio_aligned)], axis=1)**2))

    # Frame Alignment
    frame_aligned, s_f, _, _ = align_trajectories_umeyama(frame_traj, frame_gt)
    rmse_frame = np.sqrt(np.mean(np.linalg.norm(frame_aligned - frame_gt[:len(frame_aligned)], axis=1)**2))

    print(f"DEIO RMSE:     {rmse_deio:.4f} m (Scale: {s_d:.2f})")
    print(f"Frame VO RMSE: {rmse_frame:.4f} m (Scale: {s_f:.2f})")

    # 5. PLOT
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot GT (High Res - using frame GT as it's dense)
    ax.plot(frame_gt[:, 0], frame_gt[:, 1], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)

    # Plot Frame
    ax.plot(frame_aligned[:, 0], frame_aligned[:, 1], 'b--', linewidth=1.5, label=f'Frame Stereo VO (RMSE: {rmse_frame:.2f})')

    # Plot DEIO
    ax.plot(deio_aligned[:, 0], deio_aligned[:, 1], 'r--', linewidth=2, label=f'DEIO Event Stereo (RMSE: {rmse_deio:.2f})')

    ax.set_title("Comparison: Frame-Based vs. Event-Based Stereo VO", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend()
    ax.axis('equal')
    ax.grid(True, alpha=0.3)

    plt.savefig("final_comparison_result.png")
    print("Saved plot to final_comparison_result.png")

if __name__ == "__main__":
    evaluate_comparison()