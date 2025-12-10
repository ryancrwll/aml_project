import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from convert_events import VoxelGrid
import sys
import yaml
import cv2

class MVSECDataset(Dataset):
    def __init__(self, data_path, gt_path, seq_len=10, crop_params=None, use_stereo=False, use_calib=False, calib_path=None):
        """
        Initialize MVSECDataset with optional calibration-aware processing.

        Args:
            data_path: Path to HDF5 data file
            gt_path: Path to HDF5 ground truth file
            seq_len: Sequence length
            crop_params: Dict with H, W, B for voxel grid
            use_stereo: If True, concatenates left+right camera voxels (10 channels). If False, uses left only (5 channels).
            use_calib: If True, loads calibration data and applies undistortion to events.
                      This improves VO accuracy by correcting for lens distortion.
            calib_path: Path to calibration YAML file (only used if use_calib=True)
        """
        self.seq_len = seq_len
        self.crop_params = crop_params if crop_params else {'H': 260, 'W': 346, 'B': 15}
        self.use_stereo = use_stereo
        self.calib_data = None

        # Load calibration if provided
        if use_calib and calib_path:
            with open(calib_path, 'r') as f:
                self.calib_data = yaml.safe_load(f)

        # Determine which cameras to load
        self.cameras = ['left', 'right'] if use_stereo else ['left']

        # 1. Load Data & Force Materialization (Deep Copy)
        try:
            with h5py.File(data_path, 'r') as f:
                # Load for each camera
                self.events_dict = {}
                self.event_indices_dict = {}
                self.event_timestamps_dict = {}

                for cam in self.cameras:
                    self.events_dict[cam] = np.array(f['davis'][cam]['events'])
                    self.event_indices_dict[cam] = np.array(f['davis'][cam]['image_raw_event_inds'])
                    self.event_timestamps_dict[cam] = np.array(f['davis'][cam]['image_raw_ts'])

                # Try to find IMU data inside the HDF5. Common locations: '/imu', '/davis/imu'
                # Support per-camera IMUs (left/right) and fallback single IMU stream.
                self.imu_sources = {}  # map camera name -> (timestamps, measurements)

                # Helper to load potential imu dataset
                def _load_imu_group(group):
                    # Accept dataset or group. If a dataset is passed, return its array.
                    if isinstance(group, h5py.Dataset):
                        return np.array(group)

                    # group may contain datasets; try 'data' first
                    if isinstance(group, h5py.Group) and 'data' in group:
                        arr = np.array(group['data'])
                        return arr

                    # otherwise search for a dataset with 6-7 columns
                    if isinstance(group, h5py.Group):
                        for k in group.keys():
                            try:
                                ds = group[k]
                                if isinstance(ds, h5py.Dataset):
                                    arr = np.array(ds)
                                    if arr.ndim == 2 and arr.shape[1] >= 6:
                                        return arr
                            except Exception:
                                continue

                    return None

                # First try per-camera IMU under davis/left/imu and davis/right/imu
                def _try_camera_imu(cam_group, cam_name):
                    arr = _load_imu_group(cam_group)
                    if arr is None:
                        return False
                    cols = arr.shape[1]
                    imu_ts = None
                    imu_meas = None
                    if cols >= 7:
                        if np.all(np.diff(arr[:, 0]) >= 0):
                            imu_ts = arr[:, 0]
                            imu_meas = arr[:, 1:7]
                        elif np.all(np.diff(arr[:, -1]) >= 0):
                            imu_ts = arr[:, -1]
                            imu_meas = arr[:, :6]
                        else:
                            imu_ts = arr[:, 0]
                            imu_meas = arr[:, 1:7]
                    elif cols == 6:
                        # No timestamps -> cannot align accurately
                        return False

                    if imu_ts is not None and imu_meas is not None:
                        self.imu_sources[cam_name] = (imu_ts.astype(np.float64), imu_meas.astype(np.float32))
                        return True
                    return False

                # Check canonical per-camera locations
                if 'davis' in f and 'left' in f['davis'] and 'imu' in f['davis']['left']:
                    _try_camera_imu(f['davis']['left']['imu'], 'left')
                if 'davis' in f and 'right' in f['davis'] and 'imu' in f['davis']['right']:
                    _try_camera_imu(f['davis']['right']['imu'], 'right')

                # If no per-camera IMU found, search top-level or /imu
                if len(self.imu_sources) == 0:
                    imu_arr = None
                    if 'imu' in f:
                        imu_arr = _load_imu_group(f['imu'])
                    else:
                        for name, item in f.items():
                            if isinstance(item, h5py.Group) and 'imu' in name.lower():
                                imu_arr = _load_imu_group(item)
                                if imu_arr is not None:
                                    break

                    if imu_arr is not None:
                        cols = imu_arr.shape[1]
                        if cols >= 7:
                            if np.all(np.diff(imu_arr[:, 0]) >= 0):
                                imu_ts = imu_arr[:, 0]
                                imu_meas = imu_arr[:, 1:7]
                            elif np.all(np.diff(imu_arr[:, -1]) >= 0):
                                imu_ts = imu_arr[:, -1]
                                imu_meas = imu_arr[:, :6]
                            else:
                                imu_ts = imu_arr[:, 0]
                                imu_meas = imu_arr[:, 1:7]
                        else:
                            imu_ts = None
                            imu_meas = None

                        if imu_ts is not None and imu_meas is not None:
                            # assign to 'left' as default single-stream source
                            self.imu_sources['left'] = (imu_ts.astype(np.float64), imu_meas.astype(np.float32))

                # Final availability flag
                self.imu_available = len(self.imu_sources) > 0

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

        # 3. Setup calibration-based undistortion if enabled
        self.undistort_maps = {}  # Cache for undistortion maps
        if self.calib_data is not None:
            self._build_undistortion_maps()

        # Calculate how many full sequences we can make. When using multiple cameras
        # (stereo) use the minimum number of event-index entries across cameras so
        # we don't generate start indices that are valid for one camera but exceed
        # the other's length (which caused IndexError in DataLoader workers).
        min_len = min(len(self.event_indices_dict[cam]) for cam in self.cameras)
        # valid start_frame values must satisfy start_frame + seq_len <= min_len - 1
        # which yields start_frame <= min_len - seq_len - 1, so range end is min_len - seq_len
        self.valid_indices = list(range(0, max(0, min_len - seq_len)))

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
        Computes 6-DOF delta between two timestamps in the GLOBAL FRAME.
        Output order: [dx, dy, dz, d_roll, d_pitch, d_yaw]
        Coordinates: x, y, z from 4x4 pose matrix; d_roll, d_pitch, d_yaw are Euler angles.
        WARNING: The network output MUST match this order!
        """
        p1, q1 = self._get_pose_at_time(t_start)
        p2, q2 = self._get_pose_at_time(t_end)

        # 1. Calculate Global Translation Difference
        global_diff = p2 - p1  # [dx, dy, dz]

        # 2. Get Rotation matrices
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)

        # 3. Calculate Rotation Difference in GLOBAL FRAME
        # R_diff = R_2 * R_1^T
        rot_diff = r2 * r1.inv()
        euler_diff = rot_diff.as_euler('xyz', degrees=False)  # [d_roll, d_pitch, d_yaw]

        # Concatenate [dx, dy, dz, d_roll, d_pitch, d_yaw]
        return np.concatenate([global_diff, euler_diff]).astype(np.float32)

    def __len__(self):
        return len(self.valid_indices)

    def _build_undistortion_maps(self):
        """
        Pre-compute undistortion maps for each camera using calibration.
        Maps are cached to avoid recomputation during data loading.
        """
        for cam_name in self.cameras:
            cam_idx = 0 if cam_name == 'left' else 1
            cam_key = f'cam{cam_idx}'

            if cam_key not in self.calib_data:
                continue

            cam_data = self.calib_data[cam_key]

            try:
                K = np.array(cam_data.get('intrinsics', []))
                D = np.array(cam_data.get('distortion_coeffs', []))
                resolution = tuple(cam_data.get('resolution', []))

                if K.size == 0 or D.size == 0 or resolution == (0, 0):
                    continue

                # Build intrinsic matrix
                K_matrix = np.array([
                    [K[0], 0, K[2]],
                    [0, K[1], K[3]],
                    [0, 0, 1]
                ], dtype=np.float32)

                # Compute undistortion maps using OpenCV
                # This handles equidistant and other distortion models
                map_x, map_y = cv2.initUndistortRectifyMap(
                    K_matrix,
                    D,
                    None,
                    K_matrix,
                    resolution,
                    cv2.CV_32F
                )

                self.undistort_maps[cam_name] = {
                    'map_x': map_x,
                    'map_y': map_y,
                    'K': K_matrix,
                    'D': D
                }

            except Exception as e:
                print(f"Warning: Could not build undistortion maps for {cam_name}: {e}")

    def _undistort_events(self, events, camera):
        """
        Apply distortion correction to event coordinates.

        Args:
            events: (N, 4) array with columns [x, y, t, p]
            camera: 'left' or 'right'

        Returns:
            events with undistorted x, y coordinates
        """
        if camera not in self.undistort_maps:
            return events

        maps = self.undistort_maps[camera]

        try:
            # Extract and round coordinates to pixel indices
            x = np.round(events[:, 0]).astype(np.int32)
            y = np.round(events[:, 1]).astype(np.int32)

            # Clamp to valid range
            height, width = maps['map_x'].shape
            x = np.clip(x, 0, width - 1)
            y = np.clip(y, 0, height - 1)

            # Remap using precomputed maps
            undistorted_x = maps['map_x'][y, x]
            undistorted_y = maps['map_y'][y, x]

            # Clamp undistorted coordinates to image bounds
            undistorted_x = np.clip(undistorted_x, 0, width - 1)
            undistorted_y = np.clip(undistorted_y, 0, height - 1)

            # Create corrected events array
            events_corrected = events.copy()
            events_corrected[:, 0] = undistorted_x
            events_corrected[:, 1] = undistorted_y

            return events_corrected

        except Exception as e:
            print(f"Warning: Undistortion failed for {camera}: {e}")
            return events

    def get_calibration(self, camera='left'):
        """
        Get calibration data for a camera.
        Returns intrinsics, distortion coefficients, and transformation matrices.
        """
        if self.calib_data is None:
            return None

        # Map camera names to calib keys
        cam_idx = 0 if camera == 'left' else 1
        cam_key = f'cam{cam_idx}'

        if cam_key not in self.calib_data:
            return None

        cam_data = self.calib_data[cam_key]

        return {
            'intrinsics': np.array(cam_data.get('intrinsics', [])),
            'resolution': tuple(cam_data.get('resolution', [])),
            'distortion_coeffs': np.array(cam_data.get('distortion_coeffs', [])),
            'distortion_model': cam_data.get('distortion_model', 'equidistant'),
            'camera_model': cam_data.get('camera_model', 'pinhole'),
            'T_cam_imu': np.array(cam_data.get('T_cam_imu', [])),
            'projection_matrix': np.array(cam_data.get('projection_matrix', [])),
            'rectification_matrix': np.array(cam_data.get('rectification_matrix', [])),
        }

    def get_stereo_baseline(self):
        """
        Get the stereo baseline (T_cn_cnm1) from calibration.
        Returns the transformation from left to right camera.
        """
        if self.calib_data is None or 'cam1' not in self.calib_data:
            return None

        cam1_data = self.calib_data['cam1']
        T_cn_cnm1 = np.array(cam1_data.get('T_cn_cnm1', []))

        if T_cn_cnm1.size == 0:
            return None

        return T_cn_cnm1

    def __getitem__(self, idx):
        start_frame = self.valid_indices[idx]

        if self.use_stereo:
            # For stereo, return both cameras' voxel grids concatenated
            voxel_seq_left = []
            voxel_seq_right = []
            pose_seq = []

            imu_feats_seq = []
            imu_ts_seq = []
            for i in range(self.seq_len):
                # Process LEFT camera
                idx_start_left = int(self.event_indices_dict['left'][start_frame + i])
                idx_end_left = int(self.event_indices_dict['left'][start_frame + i + 1])
                event_slice_left = self.events_dict['left'][idx_start_left:idx_end_left]
                if event_slice_left.dtype != np.float32:
                    event_slice_left = event_slice_left.astype(np.float32)

                # Apply undistortion if calibration available
                if self.calib_data is not None:
                    event_slice_left = self._undistort_events(event_slice_left, 'left')

                # Extract actual event timestamps for LEFT camera (CRITICAL for alignment with GT)
                if event_slice_left.shape[0] > 0:
                    t_start_left = event_slice_left[0, 2]
                    t_end_left = event_slice_left[-1, 2]
                else:
                    t_start_left = self.event_timestamps_dict['left'][start_frame + i]
                    t_end_left = self.event_timestamps_dict['left'][start_frame + i + 1]

                grid_left = self.voxel_grid(event_slice_left, t_start=t_start_left, t_end=t_end_left)
                voxel_seq_left.append(grid_left)

                # Process RIGHT camera
                idx_start_right = int(self.event_indices_dict['right'][start_frame + i])
                idx_end_right = int(self.event_indices_dict['right'][start_frame + i + 1])
                event_slice_right = self.events_dict['right'][idx_start_right:idx_end_right]
                if event_slice_right.dtype != np.float32:
                    event_slice_right = event_slice_right.astype(np.float32)

                # Apply undistortion if calibration available
                if self.calib_data is not None:
                    event_slice_right = self._undistort_events(event_slice_right, 'right')

                # Extract actual event timestamps for RIGHT camera (CRITICAL for alignment with GT)
                if event_slice_right.shape[0] > 0:
                    t_start_right = event_slice_right[0, 2]
                    t_end_right = event_slice_right[-1, 2]
                else:
                    t_start_right = self.event_timestamps_dict['right'][start_frame + i]
                    t_end_right = self.event_timestamps_dict['right'][start_frame + i + 1]

                grid_right = self.voxel_grid(event_slice_right, t_start=t_start_right, t_end=t_end_right)
                voxel_seq_right.append(grid_right)

                # Get pose (GT is from left camera frame)
                # Use LEFT camera timestamps for both GT computation (consistent with voxel grid)
                if event_slice_left.shape[0] > 0 or event_slice_right.shape[0] > 0:
                    if self.pose.ndim == 3 and self.pose.shape[1:] == (4, 4):
                        t0_idx = start_frame + i
                        t1_idx = start_frame + i + 1
                        delta = self._compute_relative_pose(t0_idx, t1_idx)
                    else:
                        # Use LEFT camera event timestamps for pose delta (matches voxel grid)
                        delta = self._compute_relative_pose(t_start_left, t_end_left)
                else:
                    delta = np.zeros(6, dtype=np.float32)

                pose_seq.append(torch.as_tensor(delta, dtype=torch.float32))

                # IMU features for left camera (if available)
                if 'left' in self.imu_sources:
                    imu_ts_left, imu_meas_left = self.imu_sources['left']
                    mask_l = (imu_ts_left >= t_start_left) & (imu_ts_left < t_end_left)
                    slice_l = imu_meas_left[mask_l]
                    if slice_l.size == 0:
                        feat_l = np.zeros(12, dtype=np.float32)
                        ts_l = np.array([0.0, 0.0], dtype=np.float64)
                    else:
                        mean_l = slice_l.mean(axis=0)
                        std_l = slice_l.std(axis=0)
                        feat_l = np.concatenate([mean_l, std_l]).astype(np.float32)
                        ts_l = np.array([imu_ts_left[mask_l][0], imu_ts_left[mask_l][-1]], dtype=np.float64)
                else:
                    feat_l = np.zeros(12, dtype=np.float32)
                    ts_l = np.array([0.0, 0.0], dtype=np.float64)

                # IMU features for right camera (if available)
                if 'right' in self.imu_sources:
                    imu_ts_right, imu_meas_right = self.imu_sources['right']
                    mask_r = (imu_ts_right >= t_start_right) & (imu_ts_right < t_end_right)
                    slice_r = imu_meas_right[mask_r]
                    if slice_r.size == 0:
                        feat_r = np.zeros(12, dtype=np.float32)
                        ts_r = np.array([0.0, 0.0], dtype=np.float64)
                    else:
                        mean_r = slice_r.mean(axis=0)
                        std_r = slice_r.std(axis=0)
                        feat_r = np.concatenate([mean_r, std_r]).astype(np.float32)
                        ts_r = np.array([imu_ts_right[mask_r][0], imu_ts_right[mask_r][-1]], dtype=np.float64)
                else:
                    feat_r = np.zeros(12, dtype=np.float32)
                    ts_r = np.array([0.0, 0.0], dtype=np.float64)

                # Concatenate left/right features (24) and timestamps (4)
                imu_feat = np.concatenate([feat_l, feat_r]).astype(np.float32)
                imu_ts_step = np.concatenate([ts_l, ts_r]).astype(np.float32)

                imu_feats_seq.append(torch.as_tensor(imu_feat, dtype=torch.float32))
                imu_ts_seq.append(torch.as_tensor(imu_ts_step, dtype=torch.float32))

            # Concatenate left and right voxel grids along channel dimension
            voxel_seq_left = torch.stack(voxel_seq_left)  # (S, B, H, W)
            voxel_seq_right = torch.stack(voxel_seq_right)  # (S, B, H, W)
            voxel_seq = torch.cat([voxel_seq_left, voxel_seq_right], dim=1)  # (S, 2*B, H, W)

            # Stack imu features and imu timestamp vectors
            return voxel_seq, torch.stack(imu_feats_seq), torch.stack(imu_ts_seq), torch.stack(pose_seq)
        else:
            # Single camera (left)
            cam = 'left'
            voxel_seq = []
            pose_seq = []
            imu_seq = []

            imu_ts_seq = []
            for i in range(self.seq_len):
                idx_start = int(self.event_indices_dict[cam][start_frame + i])
                idx_end = int(self.event_indices_dict[cam][start_frame + i + 1])

                # 2. Slice Events
                event_slice = self.events_dict[cam][idx_start:idx_end]

                # FIX: Force the slice to be a clean float array immediately
                if event_slice.dtype != np.float32:
                     event_slice = event_slice.astype(np.float32)

                # Apply undistortion if calibration available
                if self.calib_data is not None:
                    event_slice = self._undistort_events(event_slice, cam)

                # 3. Extract actual event timestamps (CRITICAL for alignment with GT)
                if event_slice.shape[0] > 0:
                    t_start = event_slice[0, 2]   # First event timestamp
                    t_end = event_slice[-1, 2]    # Last event timestamp
                else:
                    # Fallback to frame timestamps if no events
                    t_start = self.event_timestamps_dict[cam][start_frame + i]
                    t_end = self.event_timestamps_dict[cam][start_frame + i + 1]

                grid = self.voxel_grid(event_slice, t_start=t_start, t_end=t_end)
                voxel_seq.append(grid)

                # 4. Get IMU features for this visual step (use 'left' source by default)
                if 'left' in self.imu_sources:
                    imu_ts_left, imu_meas_left = self.imu_sources['left']
                    mask = (imu_ts_left >= t_start) & (imu_ts_left < t_end)
                    slice_l = imu_meas_left[mask]
                    if slice_l.size == 0:
                        imu_feat = np.zeros(12, dtype=np.float32)
                        ts_pair = np.array([0.0, 0.0], dtype=np.float64)
                    else:
                        mean = slice_l.mean(axis=0)
                        std = slice_l.std(axis=0)
                        imu_feat = np.concatenate([mean, std]).astype(np.float32)
                        ts_pair = np.array([imu_ts_left[mask][0], imu_ts_left[mask][-1]], dtype=np.float64)
                else:
                    imu_feat = np.zeros(12, dtype=np.float32)
                    ts_pair = np.array([0.0, 0.0], dtype=np.float64)

                imu_seq.append(torch.as_tensor(imu_feat, dtype=torch.float32))
                imu_ts_seq.append(torch.as_tensor(ts_pair.astype(np.float32), dtype=torch.float32))

                # 5. Get Pose Ground Truth
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
                        # Use event timestamps (already extracted above) for pose delta
                        # This MUST match the timestamps used for voxel grid
                        delta = self._compute_relative_pose(t_start, t_end)
                else:
                    delta = np.zeros(6, dtype=np.float32)

                pose_seq.append(torch.as_tensor(delta, dtype=torch.float32))

            return torch.stack(voxel_seq), torch.stack(imu_seq), torch.stack(imu_ts_seq), torch.stack(pose_seq)