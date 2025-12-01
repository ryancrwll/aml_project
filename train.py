import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os
import sys

# Ensure local modules (dataloader.py, network.py) are accessible
sys.path.append(os.path.dirname(__file__))

# --- Configuration (UPDATE PATHS AND PARAMETERS HERE) ---
# NOTE: These paths must match where your HDF5 files are located.
# Ensure local modules (dataloader.py, network.py) are accessible
sys.path.append(os.path.dirname(__file__))

from dataloader import MVSECDataset
from VOnetwork import VONet

# --- Configuration (UPDATE PATHS AND PARAMETERS HERE) ---
# NOTE: Use LISTS for multiple data files
DATA_FILES = [
    './data/indoor_flying4_data.hdf5',
    './data/indoor_flying3_data.hdf5',
    './data/indoor_flying2_data.hdf5'
]
GT_FILES = [
    './data/indoor_flying4_gt.hdf5',
    './data/indoor_flying3_gt.hdf5',
    './data/indoor_flying2_gt.hdf5'
]

# Training Parameters
BATCH_SIZE = 4
SEQ_LEN = 5  # Sequence length (T) for the RNN input
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model/Data Parameters (Must match VoxelGrid in convert_events.py)
VOXEL_PARAMS = {
    'H': 260,
    'W': 346,
    'B': 5 # C: Number of channels/bins in the Voxel Grid
}
ROT_WEIGHT = 1000.0 # Weight for rotation loss (keep high)

TRANS_SCALE_FACTOR = 1000.0

def pose_loss(pred, target, rot_weight, trans_scale):
    """
    Combined loss function: Scales translation targets up for stability.
    L = L_trans_scaled + rot_weight * L_rot
    """

    # Scale the ground truth translation (first 3 elements)
    target_t_scaled = target[..., :3] * trans_scale

    # Scale the predicted translation (first 3 elements)
    # NOTE: Prediction is naturally scaled up by the network to match the target.
    pred_t_scaled = pred[..., :3]

    # Translation loss (MSE on scaled values)
    loss_t = nn.functional.mse_loss(pred_t_scaled, target_t_scaled)

    # Rotation loss (Last 3 elements)
    loss_r = nn.functional.mse_loss(pred[..., 3:], target[..., 3:])

    total_loss = loss_t + rot_weight * loss_r
    return total_loss

def train():
    print(f"Starting training on {DEVICE}")

    # 1. Dataset Initialization
    try:
        # Create a list to hold individual datasets
        datasets = []

        # Iterate over all file pairs and create a dataset for each
        for data_file, gt_file in zip(DATA_FILES, GT_FILES):
            print(f"Loading dataset from: {data_file}")
            datasets.append(
                MVSECDataset(
                    data_path=data_file,
                    gt_path=gt_file,
                    seq_len=SEQ_LEN,
                    crop_params=VOXEL_PARAMS
                )
            )

        # Concatenate all individual datasets into one
        dataset = ConcatDataset(datasets)

        # Create the DataLoader using the combined dataset
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
        print(f"Dataset loaded successfully. Total sequences: {len(dataset)}")

    except Exception as e:
        print(f"FATAL ERROR: Data loading failed. Check paths or underlying HDF5 structure: {e}")
        return

    # 2. Model, Optimizer, and Checkpoints
    model = VONet(
        input_channels=VOXEL_PARAMS['B']
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        # Use tqdm for progress bar
        print(f"/n")
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for voxels, targets in loop:

            if voxels.numel() == 0 or voxels.size(1) != SEQ_LEN:
                continue

            voxels, targets = voxels.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass
            preds = model(voxels)

            # Calculate loss (using the new scale factor)
            loss = pose_loss(preds, targets, rot_weight=ROT_WEIGHT, trans_scale=TRANS_SCALE_FACTOR)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        # Save Checkpoint
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f'checkpoints/vo_model_ep{epoch+1}.pth')

if __name__ == "__main__":
    train()