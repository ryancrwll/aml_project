import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
# import from your existing files
from dataloader import MVSECDataset
from convert_events import VoxelGrid

# --- DEIO requires a full GT state (position, orientation, velocity, bias)
# --- And raw IMU measurements (accel, gyro) per time step.

class DEIODataset(MVSECDataset):
    def __init__(self, data_path, gt_path, seq_len=10, crop_params=None, use_stereo=False,
                 use_calib=False, calib_path=None, imu_state_dim=6, gt_state_dim=15):
        # We call the parent constructor to load all events, indices, and IMU data
        super().__init__(data_path, gt_path, seq_len, crop_params, use_stereo, use_calib, calib_path)
        self.imu_state_dim = imu_state_dim
        self.gt_state_dim = gt_state_dim # 15: P(3), Q(4), V(3), Accel_Bias(3), Gyro_Bias(3)

    # --- NOTE: We override the GT extraction logic to return a full state, not a delta ---
    def _get_full_state_at_time(self, timestamp):
        """
        Extracts a full IMU state (P, Q, V, Ba, Bg) at a specific timestamp.
        Since the HDF5 contains 4x4 SE(3) poses (not a full IMU state),
        we extract P(3) and Q(4) from the pose matrix and pad V, Ba, Bg with zeros.
        Returns 16-D state for consistency with dataset initialization.
        """
        # Get pose at timestamp using the parent class method
        p, q = self._get_pose_at_time(timestamp)

        # Construct 16-D state: P(3) + Q(4) + V(3) + Ba(3) + Bg(3) = 16 total
        # Since full IMU state not available, pad with zeros
        state = np.concatenate([
            p,                                          # P(3)
            q,                                          # Q(4)
            np.zeros(3, dtype=np.float32),             # V(3) - padded
            np.zeros(3, dtype=np.float32),             # Ba(3) - padded
            np.zeros(3, dtype=np.float32)              # Bg(3) - padded
        ]).astype(np.float32)

        return state

    def __getitem__(self, idx):
        # This implementation assumes mono-camera for simplicity (left only)
        start_frame = self.valid_indices[idx]
        cam = 'left'
        voxel_seq = []
        imu_raw_seq = []
        gt_state_seq = []

        for i in range(self.seq_len):
            idx_start = int(self.event_indices_dict[cam][start_frame + i])
            idx_end = int(self.event_indices_dict[cam][start_frame + i + 1])
            event_slice = self.events_dict[cam][idx_start:idx_end].astype(np.float32)

            t_start = self.event_timestamps_dict[cam][start_frame + i]
            t_end = self.event_timestamps_dict[cam][start_frame + i + 1]

            # 1. Create Voxel Grid (Input)
            grid = self.voxel_grid(event_slice, t_start=t_start, t_end=t_end)
            voxel_seq.append(grid)

            # 2. Extract RAW IMU measurements (Input)
            # Find IMU data between t_start and t_end
            if 'left' in self.imu_sources:
                imu_ts_left, imu_meas_left = self.imu_sources['left']
                mask = (imu_ts_left >= t_start) & (imu_ts_left < t_end)
                slice_l = imu_meas_left[mask] # (N_imu_steps, 6)

                # DEIO requires IMU integration; we simplify by using the mean raw state
                # You would need to pre-integrate or provide the sequence of raw IMU data here.
                if slice_l.size == 0:
                    imu_raw_state = np.zeros(self.imu_state_dim, dtype=np.float32)
                else:
                    # Placeholder: Use the mean raw IMU accel/gyro (6D) for the time step
                    imu_raw_state = slice_l.mean(axis=0)
            else:
                imu_raw_state = np.zeros(self.imu_state_dim, dtype=np.float32)

            imu_raw_seq.append(torch.as_tensor(imu_raw_state, dtype=torch.float32))

            # 3. Get Full GT State (Target)
            # The target is the full state at the end of the time step
            gt_state = self._get_full_state_at_time(t_end)
            gt_state_seq.append(torch.as_tensor(gt_state, dtype=torch.float32))

        # Output: Voxel, IMU Raw State (mean), Full GT State (P, Q, V, B)
        return torch.stack(voxel_seq), torch.stack(imu_raw_seq), torch.stack(gt_state_seq)