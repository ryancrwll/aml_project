#!/usr/bin/env python3
"""
Investigate what event_timestamps_dict represents and how to align it with GT.
"""

import h5py
import numpy as np

DATA_PATH = './data/indoor_flying4_data.hdf5'

with h5py.File(DATA_PATH, 'r') as f:
    # Explore structure
    print("HDF5 Structure:")
    print(f.keys())
    print()

    davis = f['davis']
    left = davis['left']

    print("Left camera datasets:")
    print(left.keys())
    print()

    # Load relevant data
    events = np.array(left['events'])
    event_indices = np.array(left['image_raw_event_inds'])
    image_raw_ts = np.array(left['image_raw_ts'])

    print(f"Events shape: {events.shape}, dtype: {events.dtype}")
    print(f"Event indices shape: {event_indices.shape}, dtype: {event_indices.dtype}")
    print(f"Image raw ts shape: {image_raw_ts.shape}, dtype: {image_raw_ts.dtype}")
    print()

    # Sample first few entries
    print("First 10 frames' timestamp boundaries:")
    for i in range(min(10, len(image_raw_ts))):
        ts = image_raw_ts[i]
        idx_start = event_indices[i]
        idx_end = event_indices[i+1]

        # Get event slice
        event_slice = events[int(idx_start):int(idx_end)]

        if event_slice.shape[0] > 0:
            t0_first_event = event_slice[0, 2]
            t1_last_event = event_slice[-1, 2]
            print(f"Frame {i}: image_ts={ts:.6f}, "
                  f"events span [{idx_start}, {idx_end}), "
                  f"event_ts=[{t0_first_event:.6f}, {t1_last_event:.6f}], "
                  f"span_ms={1000*(t1_last_event - t0_first_event):.3f}")
        else:
            print(f"Frame {i}: image_ts={ts:.6f}, NO EVENTS")

    print()
    print("CONCLUSION:")
    print("-" * 80)
    print("image_raw_ts represents the FRAME CAPTURE TIME (from the camera).")
    print("The GT pose is computed between first and last EVENT in the frame slice.")
    print()
    print("SOLUTION:")
    print("Use event_slice[0, 2] and event_slice[-1, 2] for voxel grid boundaries,")
    print("not image_raw_ts values.")
    print("-" * 80)
