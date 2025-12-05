import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os
import sys

# Ensure local modules (dataloader.py, network.py) are accessible
sys.path.append(os.path.dirname(__file__))

from dataloader import MVSECDataset
from VOnetwork import VONet

# --- Configuration (Set back to proven speeds, relying on Clipping for stability) ---
TRAIN_DATASETS = [
    {'data': './data/indoor_flying2_data.hdf5', 'gt': './data/indoor_flying2_gt.hdf5'},
    {'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
    {'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'}
]

BATCH_SIZE = 4
SEQ_LEN = 5
EPOCHS = 50
# LR = 5e-5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
ROT_WEIGHT = 50.0
TRANS_SCALE_FACTOR = 1000.0

# NEW HYPERPARAMETER
GRAD_CLIP_VALUE = 1.0 # Standard value for gradient clipping

# --- Multi-camera and Calibration Configuration ---
USE_STEREO = True  # If True: uses both left+right cameras (10 channels). If False: left camera only (5 channels)
USE_CALIB = False   # If True: loads calibration data from YAML file. If False: skips calibration loading
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'

def pose_loss(pred, target, rot_weight, trans_scale):
    """
    Compute pose loss.
    Inputs:
      pred: (B, S, 6) network predictions in order [tx, ty, tz, d_roll, d_pitch, d_yaw]
      target: (B, S, 6) ground truth in order [dx, dy, dz, d_roll, d_pitch, d_yaw]
      trans_scale: scale factor to match prediction magnitude to GT
    Returns: total loss = loss_translation + rot_weight * loss_rotation
    """
    # Translation loss: scale predictions to match GT scale
    target_t_scaled = target[..., :3] * trans_scale  # [B, S, 3]
    pred_t_scaled = pred[..., :3]  # [B, S, 3]
    loss_t = nn.functional.mse_loss(pred_t_scaled, target_t_scaled)

    # Rotation loss: compare Euler angles directly
    loss_r = nn.functional.mse_loss(pred[..., 3:], target[..., 3:])  # [B, S, 3]

    return loss_t + rot_weight * loss_r

def train():
    print(f"Starting training on {DEVICE}")
    print(f"Stereo Mode: {USE_STEREO}, Calibration: {USE_CALIB}")

    # 1. Load Datasets with multi-camera support
    datasets = []
    for paths in TRAIN_DATASETS:
        d_path = paths['data']
        g_path = paths['gt']
        if not os.path.exists(d_path) or not os.path.exists(g_path):
            continue
        try:
            ds = MVSECDataset(
                data_path=d_path,
                gt_path=g_path,
                seq_len=SEQ_LEN,
                crop_params=VOXEL_PARAMS,
                use_stereo=USE_STEREO,
                use_calib=USE_CALIB,
                calib_path=CALIB_PATH if USE_CALIB else None
            )
            datasets.append(ds)
        except Exception as e:
            print(f"  Error loading {d_path}: {e}")

    if not datasets:
        print("FATAL: No datasets loaded. Check your file paths.")
        return

    combined_dataset = ConcatDataset(datasets)
    print(f"Total training sequences: {len(combined_dataset)}")

    loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    # 2. Model Setup
    # Determine input channels based on stereo setting
    input_channels = VOXEL_PARAMS['B'] * 2 if USE_STEREO else VOXEL_PARAMS['B']

    model = VONet(input_channels=input_channels).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for voxels, targets in loop:
            if voxels.numel() == 0 or voxels.size(1) != SEQ_LEN:
                continue

            voxels, targets = voxels.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(voxels)
            loss = pose_loss(preds, targets, rot_weight=ROT_WEIGHT, trans_scale=TRANS_SCALE_FACTOR)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)

            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}\n")

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f'checkpoints/vo_model_ep{epoch+1}.pth')

if __name__ == "__main__":
    train()