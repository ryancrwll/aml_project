import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
# import from your existing files
from dataloader import MVSECDataset
from convert_events import VoxelGrid

class DEIODataset(MVSECDataset):
    def __init__(self, data_path, gt_path, seq_len=10, crop_params=None, use_stereo=False,
                 use_calib=False, calib_path=None, imu_state_dim=6, gt_state_dim=15):
        super().__init__(data_path, gt_path, seq_len, crop_params, use_stereo, use_calib, calib_path)
        self.imu_state_dim = imu_state_dim
        self.gt_state_dim = gt_state_dim
        self.use_stereo = use_stereo # Ensure this is stored

    def _get_full_state_at_time(self, timestamp):
        """
        Extracts a full IMU state (P, Q, V, Ba, Bg) at a specific timestamp.
        """
        p, q = self._get_pose_at_time(timestamp)

        # Construct 16-D state: P(3) + Q(4) + V(3) + Ba(3) + Bg(3) = 16 total
        state = np.concatenate([
            p,
            q,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32)
        ]).astype(np.float32)

        return state

    def __getitem__(self, idx):
        start_frame = self.valid_indices[idx]

        # --- FIX: Determine which cameras to use based on stereo flag ---
        cameras = ['left', 'right'] if self.use_stereo else ['left']

        voxel_seq = []
        imu_raw_seq = []
        gt_state_seq = []

        for i in range(self.seq_len):
            t_start = self.event_timestamps_dict['left'][start_frame + i]
            t_end = self.event_timestamps_dict['left'][start_frame + i + 1]

            # 1. Create Voxel Grid (Input)
            # We must loop through required cameras and stack them
            step_grids = []
            for cam in cameras:
                idx_start = int(self.event_indices_dict[cam][start_frame + i])
                idx_end = int(self.event_indices_dict[cam][start_frame + i + 1])
                event_slice = self.events_dict[cam][idx_start:idx_end].astype(np.float32)

                # Generate grid for this camera (C, H, W)
                grid = self.voxel_grid(event_slice, t_start=t_start, t_end=t_end)
                step_grids.append(grid)

            # Concatenate along channel dimension (dim=0)
            # Mono: (5, H, W) -> Stays (5, H, W)
            # Stereo: List[(5,H,W), (5,H,W)] -> Becomes (10, H, W)
            combined_grid = torch.cat(step_grids, dim=0)
            voxel_seq.append(combined_grid)

            # 2. Extract RAW IMU measurements (Input)
            # IMU is usually only associated with the left camera or a central unit
            if 'left' in self.imu_sources:
                imu_ts_left, imu_meas_left = self.imu_sources['left']
                mask = (imu_ts_left >= t_start) & (imu_ts_left < t_end)
                slice_l = imu_meas_left[mask]

                if slice_l.size == 0:
                    imu_raw_state = np.zeros(self.imu_state_dim, dtype=np.float32)
                else:
                    imu_raw_state = slice_l.mean(axis=0)
            else:
                imu_raw_state = np.zeros(self.imu_state_dim, dtype=np.float32)

            imu_raw_seq.append(torch.as_tensor(imu_raw_state, dtype=torch.float32))

            # 3. Get Full GT State (Target)
            gt_state = self._get_full_state_at_time(t_end)
            gt_state_seq.append(torch.as_tensor(gt_state, dtype=torch.float32))

        # Output:
        # Voxel: (Seq, Channels, H, W) -> Channels is 5 (Mono) or 10 (Stereo)
        return torch.stack(voxel_seq), torch.stack(imu_raw_seq), torch.stack(gt_state_seq)