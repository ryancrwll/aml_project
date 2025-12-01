import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from convert_events import VoxelGrid
import sys # Import sys for printing diagnostics

class MVSECDataset(Dataset):
    def __init__(self, data_path, gt_path, seq_len=10, crop_params=None):
        self.seq_len = seq_len
        self.crop_params = crop_params if crop_params else {'H': 260, 'W': 346, 'B': 15}

        # 1. Load Data & Force Materialization (Deep Copy)
        try:
            with h5py.File(data_path, 'r') as f:
                self.events = np.array(f['davis']['left']['events'])
                self.event_indices = np.array(f['davis']['left']['image_raw_event_inds'])
                self.event_timestamps = np.array(f['davis']['left']['image_raw_ts'])

            with h5py.File(gt_path, 'r') as f:
                if 'davis' in f:
                    self.pose = np.array(f['davis']['left']['pose'])
                else:
                    self.pose = np.array(f['pose'])

        except Exception as e:
            print(f"Error loading HDF5 files: {e}")
            raise

        # 2. Setup Processing
        self.voxel_grid = VoxelGrid(**self.crop_params)

        # Calculate how many full sequences we can make
        self.valid_indices = list(range(0, len(self.event_indices) - seq_len))

    def _get_pose_at_time(self, timestamp):
        """Finds and extracts the 7-DOF pose at a specific timestamp."""
        # Handle Structured vs Unstructured Pose Data
        if self.pose.dtype.names:
            # Structured Array (access by field name)
            times = self.pose['t']
            idx = np.searchsorted(times, timestamp)
            idx = np.clip(idx, 0, len(self.pose) - 1)
            target = self.pose[idx]

            p = np.array([target['x'], target['y'], target['z']])
            q = np.array([target['qx'], target['qy'], target['qz'], target['qw']])

        elif self.pose.ndim == 3 and self.pose.shape[1:] == (4, 4):
            # Since the array is (N, 4, 4), the first dimension is the index, not time.
            # We must assume the timestamps are the frame indices 0 to N-1 for search.
            times = np.arange(len(self.pose)) # Treat array indices as timestamps for search

            # NOTE: We are using event time (float) to search array indices (int), which is an approximation

            idx = int(timestamp) # Assuming timestamp is the frame index (e.g., 0, 1, 2)
            idx = np.clip(idx, 0, len(self.pose) - 1)
            target = self.pose[idx] # Target is now a (4, 4) matrix

            # Extract Translation (top-right 3x1 vector)
            p = target[:3, 3]

            # Extract Rotation Matrix (top-left 3x3 matrix)
            R_matrix = target[:3, :3]

            # Convert Rotation Matrix to Quaternion (qx, qy, qz, qw)
            r = R.from_matrix(R_matrix)
            q = r.as_quat()

        elif self.pose.ndim == 2 and self.pose.shape[1] == 8:
            # Existing Unstructured Array (N, 8) logic
            times = self.pose[:, 0]
            idx = np.searchsorted(times, timestamp)
            idx = np.clip(idx, 0, len(self.pose) - 1)
            target = self.pose[idx]

            p = target[1:4] # x, y, z
            q = target[4:8] # qx, qy, qz, qw

        else:
            raise ValueError(f"Unsupported pose array shape {self.pose.shape} for loading.")

        return p, q

    def _compute_relative_pose(self, t_start, t_end):
        """
        Computes 6-DOF delta between two timestamps in the BODY FRAME.
        This fixes the 'squiggly' trajectory issue.
        """
        p1, q1 = self._get_pose_at_time(t_start)
        p2, q2 = self._get_pose_at_time(t_end)

        # 1. Calculate Global Translation Difference
        global_diff = p2 - p1

        # 2. Get Rotation of the Start Frame
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)

        # 3. Rotate Translation into Body Frame (Critical Step)
        # We apply the INVERSE of the current rotation to the global difference.
        # This converts "North movement" into "Forward movement" relative to the drone.
        local_trans = r1.inv().apply(global_diff)

        # 4. Calculate Local Rotation Difference
        # R_2 = R_1 * R_rel  ->  R_rel = R_1_inv * R_2
        local_rot_matrix = r1.inv() * r2
        local_rot_euler = local_rot_matrix.as_euler('xyz', degrees=False)

        # Concatenate [dx, dy, dz, d_roll, d_pitch, d_yaw]
        return np.concatenate([local_trans, local_rot_euler]).astype(np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        start_frame = self.valid_indices[idx]

        voxel_seq = []
        pose_seq = []

        for i in range(self.seq_len):
            idx_start = int(self.event_indices[start_frame + i])
            idx_end = int(self.event_indices[start_frame + i + 1])

            # 2. Slice Events
            event_slice = self.events[idx_start:idx_end]

            # FIX: Force the slice to be a clean float array immediately
            if event_slice.dtype != np.float32:
                 event_slice = event_slice.astype(np.float32)

            # 3. Create Voxel Grid using frame time boundaries instead of event times
            # This fixes the issue where all events have the same timestamp
            t_start = self.event_timestamps[start_frame + i]
            t_end = self.event_timestamps[start_frame + i + 1]
            grid = self.voxel_grid(event_slice, t_start=t_start, t_end=t_end)
            voxel_seq.append(grid)

            # 4. Get Pose Ground Truth
            if event_slice.shape[0] > 0:
                # If the GT poses are stored as (N, 4, 4) matrices then
                # these correspond to frame indices rather than continuous
                # timestamps. In that case use the sequence frame indices
                # (derived from start_frame) to index into the pose array.
                if self.pose.ndim == 3 and self.pose.shape[1:] == (4, 4):
                    t0_idx = start_frame + i
                    t1_idx = start_frame + i + 1
                    delta = self._compute_relative_pose(t0_idx, t1_idx)
                else:
                    # Access timestamps
                    t0 = event_slice[0, 2]
                    t1 = event_slice[-1, 2]

                    # Compute pose delta using timestamps
                    delta = self._compute_relative_pose(t0, t1)
            else:
                delta = np.zeros(6, dtype=np.float32)

            pose_seq.append(torch.as_tensor(delta, dtype=torch.float32))

        return torch.stack(voxel_seq), torch.stack(pose_seq)