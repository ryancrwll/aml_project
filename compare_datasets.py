#!/usr/bin/env python3
"""
Compare dataset statistics across indoor_flying 1-4.
"""

import h5py
import numpy as np
import os

datasets = [
    ('./data/indoor_flying1_data.hdf5', './data/indoor_flying1_gt.hdf5'),
    ('./data/indoor_flying2_data.hdf5', './data/indoor_flying2_gt.hdf5'),
    ('./data/indoor_flying3_data.hdf5', './data/indoor_flying3_gt.hdf5'),
    ('./data/indoor_flying4_data.hdf5', './data/indoor_flying4_gt.hdf5'),
]

print("=" * 100)
print("DATASET STATISTICS COMPARISON")
print("=" * 100)

for data_path, gt_path in datasets:
    if not os.path.exists(data_path) or not os.path.exists(gt_path):
        print(f"\n{os.path.basename(data_path)}: NOT FOUND")
        continue

    with h5py.File(data_path, 'r') as f:
        left = f['davis']['left']
        events = np.array(left['events'])
        indices = np.array(left['image_raw_event_inds'])
        ts = np.array(left['image_raw_ts'])

        num_frames = len(ts)
        num_events = len(events)

        # Sample frame statistics
        frame_event_counts = []
        frame_durations = []
        for i in range(min(100, num_frames-1)):
            idx_start = int(indices[i])
            idx_end = int(indices[i+1])
            count = idx_end - idx_start
            frame_event_counts.append(count)

            if count > 0:
                t_start = events[idx_start, 2]
                t_end = events[idx_end-1, 2]
                duration = t_end - t_start
                frame_durations.append(duration * 1000)  # ms

        with h5py.File(gt_path, 'r') as g:
            if 'davis' in g:
                pose = np.array(g['davis']['left']['pose'])
            else:
                pose = np.array(g['pose'])

            # Compute GT motion statistics
            if pose.ndim == 3 and pose.shape[1:] == (4, 4):
                # Frame-indexed poses
                translations = []
                rotations = []
                for i in range(min(100, pose.shape[0]-1)):
                    p1 = pose[i, :3, 3]
                    p2 = pose[i+1, :3, 3]
                    trans = np.linalg.norm(p2 - p1)
                    translations.append(trans)

                    # Simple rotation norm
                    rot = pose[i+1, :3, :3] @ np.linalg.inv(pose[i, :3, :3])
                    rot_error = np.linalg.norm(rot - np.eye(3))
                    rotations.append(rot_error)

    print(f"\n{os.path.basename(data_path)}:")
    print(f"  Frames: {num_frames}, Events: {num_events:,}")
    print(f"  Avg events/frame: {num_events / num_frames:.1f}")
    print(f"  Frame event counts: min={min(frame_event_counts)}, max={max(frame_event_counts)}, mean={np.mean(frame_event_counts):.1f}")
    if frame_durations:
        print(f"  Frame durations: min={min(frame_durations):.2f}ms, max={max(frame_durations):.2f}ms, mean={np.mean(frame_durations):.2f}ms")
    print(f"  GT translation/frame: min={min(translations):.6f}m, max={max(translations):.6f}m, mean={np.mean(translations):.6f}m")
    print(f"  GT rotation/frame: min={min(rotations):.6f}, max={max(rotations):.6f}, mean={np.mean(rotations):.6f}")

print("\n" + "=" * 100)
print("KEY OBSERVATIONS:")
print("-" * 100)
print("If flying1 has significantly different characteristics (more events, faster motion, etc.)")
print("then the model may need dataset-specific fine-tuning or domain adaptation.")
print("=" * 100)
