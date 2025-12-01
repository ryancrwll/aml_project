import torch
import numpy as np
from convert_events import VoxelGrid

def test_voxel_grid():
    print("--- Testing VoxelGrid ---")

    # 1. Setup Parameters
    H, W = 260, 346
    B = 5
    N = 100 # Number of events

    # 2. Create Dummy Unstructured Event Data (N, 4)
    # Columns: [x, y, t, p]
    events = np.zeros((N, 4), dtype=np.float32)

    # Random X and Y coordinates
    events[:, 0] = np.random.randint(0, W, size=N) # x
    events[:, 1] = np.random.randint(0, H, size=N) # y

    # Timestamps (increasing)
    events[:, 2] = np.linspace(0, 1000, N) # t (e.g., 0 to 1000 microseconds)

    # Polarity (-1 or 1)
    # Randomly assign -1 or 1
    events[:, 3] = np.random.choice([-1.0, 1.0], size=N) # p

    print(f"Generated dummy events with shape: {events.shape}")
    print(f"Sample event (x, y, t, p): {events[0]}")

    # 3. Initialize VoxelGrid
    voxel_grid = VoxelGrid(H=H, W=W, B=B, normalize=True)

    # 4. Run Conversion
    try:
        tensor = voxel_grid(events)
        print(f"\nSUCCESS: Voxel Grid generated.")
        print(f"Output Tensor Shape: {tensor.shape}") # Expected: (5, 260, 346)
        print(f"Tensor Type: {tensor.dtype}")
        print(f"Tensor Device: {tensor.device}")

        # Check for non-zero values (since we added events)
        if tensor.abs().sum() > 0:
            print("Verification: Tensor contains data (sum > 0).")
        else:
            print("WARNING: Tensor is all zeros. Check accumulation logic.")

    except Exception as e:
        print(f"\nFAILURE: VoxelGrid crashed with error: {e}")

if __name__ == "__main__":
    test_voxel_grid()