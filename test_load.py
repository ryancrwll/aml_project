import torch
import sys
import os
from torch.utils.data import DataLoader

# Ensure local modules are findable
sys.path.append(os.getcwd())

from dataloader import MVSECDataset

# --- CUSTOM COLLATE FUNCTION ---
def collate_fn_sequence(batch):
    """
    Custom collate function to handle batching of sequences of Tensors.
    It manually stacks the voxel sequences and pose sequences into a single batch.
    """
    # 1. Separate the two components (voxels and poses)
    voxel_sequences = [item[0] for item in batch]
    pose_sequences = [item[1] for item in batch]

    # 2. Stack the sequences to form the final batch (Batch_Size, Seq_Len, ...)
    try:
        batched_voxels = torch.stack(voxel_sequences)
        batched_poses = torch.stack(pose_sequences)
    except Exception as e:
        # This catch is mainly for debugging if the shapes somehow became non-uniform.
        print(f"\nFATAL COLLATE ERROR: Failed to stack sequences. Error details: {e}")
        raise

    return batched_voxels, batched_poses
# -----------------------------

def test_dataloader():
    print("--- Testing MVSECDataset with DataLoader (Final Attempt) ---")

    # --- CONFIGURATION (UPDATE THESE PATHS) ---
    DATA_PATH = './data/indoor_flying4_data.hdf5'
    GT_PATH = './data/indoor_flying4_gt.hdf5'
    # ------------------------------------------

    if not os.path.exists(DATA_PATH) or not os.path.exists(GT_PATH):
        print(f"ERROR: Could not find HDF5 files at specified paths.")
        print(f"Data: {DATA_PATH}")
        print(f"GT: {GT_PATH}")
        return

    # 1. Initialize Dataset
    print("Initializing Dataset...")
    try:
        dataset = MVSECDataset(
            data_path=DATA_PATH,
            gt_path=GT_PATH,
            seq_len=5,
            crop_params={'H': 260, 'W': 346, 'B': 5}
        )
        print(f"Dataset initialized. Total sequences: {len(dataset)}")
    except Exception as e:
        print(f"FAILURE: Dataset initialization failed: {e}")
        return

    # 2. Initialize DataLoader
    # Using the stable configuration: num_workers=0 and custom collate_fn
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn_sequence
    )

    # 3. Fetch One Batch
    print("\nAttempting to load one batch...")
    try:
        # Get the first batch
        voxels, poses = next(iter(loader))

        print("\nSUCCESS: Batch loaded.")
        print(f"Voxel Batch Shape: {voxels.shape}")
        print(f"Voxel Data Type: {voxels.dtype}")

        print(f"Pose Batch Shape: {poses.shape}")
        print(f"Pose Data Type: {poses.dtype}")

        print(f"\nSample Pose Delta (first in sequence): \n{poses[0, 0]}")

    except Exception as e:
        print(f"\nFAILURE: DataLoader iteration failed: {e}")

if __name__ == "__main__":
    test_dataloader()