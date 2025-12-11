"""
Train on `flying1` (hardest) and validate on `flying2`.
Logs per-epoch train/val to `logs/train_1train_2val_epoch_logs.csv` and stdout.
Saves checkpoints every 5 epochs and when validation improves.
"""

import os
import time
import csv
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataloader import MVSECDataset
from VOnetwork import VONet

# -------- CONFIG --------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP_VALUE = 1.0
ROT_WEIGHT = 10.0
TRANS_SCALE_FACTOR = 1000.0
SEQ_LEN = 10
NUM_WORKERS = 4

USE_STEREO = True
USE_CALIB = True
VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, 'train_1train_2val_epoch_logs.csv')
STDOUT_LOG = './train_1train_2val.log'
CKPT_DIR = './checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

HUBER_LOSS_FN = nn.SmoothL1Loss(reduction='mean', beta=1.0)

# -------- LOSS --------

def pose_loss(pred, target, rot_weight, trans_scale):
    pred_t = pred[..., :3]
    pred_r = pred[..., 3:]
    target_t = target[..., :3]
    target_r = target[..., 3:]

    loss_t = HUBER_LOSS_FN(pred_t, target_t)
    loss_r = HUBER_LOSS_FN(pred_r, target_r)
    return loss_t + rot_weight * loss_r

# -------- DATA LOADING --------

def load_dataset_pair():
    """Load train: flying1 ; val: flying2"""
    try:
        train_ds = MVSECDataset(
            data_path='./data/indoor_flying1_data.hdf5',
            gt_path='./data/indoor_flying1_gt.hdf5',
            seq_len=SEQ_LEN,
            crop_params=VOXEL_PARAMS,
            use_stereo=USE_STEREO,
            use_calib=USE_CALIB,
            calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml' if USE_CALIB else None,
        )
        print(f'Loaded train flying1: {len(train_ds)} sequences')
    except Exception as e:
        raise RuntimeError(f'Error loading flying1: {e}')

    try:
        val_ds = MVSECDataset(
            data_path='./data/indoor_flying2_data.hdf5',
            gt_path='./data/indoor_flying2_gt.hdf5',
            seq_len=SEQ_LEN,
            crop_params=VOXEL_PARAMS,
            use_stereo=USE_STEREO,
            use_calib=USE_CALIB,
            calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml' if USE_CALIB else None,
        )
        print(f'Loaded val flying2: {len(val_ds)} sequences')
    except Exception as e:
        raise RuntimeError(f'Error loading flying2: {e}')

    return train_ds, val_ds

# -------- VALIDATION --------

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_loss_t = 0.0
    total_loss_r = 0.0
    batches = 0
    sq_err_sum = 0.0
    count_pts = 0
    with torch.no_grad():
        for voxels, imu_feats, imu_ts, targets in val_loader:
            voxels = voxels.to(device)
            imu_feats = imu_feats.to(device)
            targets = targets.to(device)

            targets_scaled = targets.clone()
            targets_scaled[..., :3] = targets_scaled[..., :3] * TRANS_SCALE_FACTOR

            preds = model(voxels, imu_feats)

            # component losses
            pred_t = preds[..., :3]
            pred_r = preds[..., 3:]
            target_t_scaled = targets_scaled[..., :3]
            target_r = targets_scaled[..., 3:]

            loss_t = HUBER_LOSS_FN(pred_t, target_t_scaled)
            loss_r = HUBER_LOSS_FN(pred_r, target_r)
            loss = loss_t + ROT_WEIGHT * loss_r

            total_loss += loss.item()
            total_loss_t += loss_t.item()
            total_loss_r += loss_r.item()
            batches += 1

            # RMSE in meters: pred_t is scaled, so descale before compare
            pred_t_m = pred_t / TRANS_SCALE_FACTOR
            target_t_m = targets[..., :3]
            diff = (pred_t_m - target_t_m).cpu().numpy()
            sq_err_sum += (diff ** 2).sum()
            count_pts += diff.size

    avg_loss = total_loss / max(1, batches)
    avg_loss_t = total_loss_t / max(1, batches)
    avg_loss_r = total_loss_r / max(1, batches)
    val_rmse = (sq_err_sum / max(1, count_pts)) ** 0.5
    return avg_loss, avg_loss_t, avg_loss_r, val_rmse

# -------- TRAIN LOOP --------

def train():
    print('='*80)
    print('TRAINING: flying1 -> VALIDATE: flying2')
    print('='*80)
    print(f'Device: {DEVICE}, batch: {BATCH_SIZE}, epochs: {EPOCHS}, lr: {LR}')

    train_ds, val_ds = load_dataset_pair()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

    input_channels = 10 if USE_STEREO else 5
    model = VONet(input_channels=input_channels).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val = float('inf')

    # CSV logging header (add translation/rotation components and RMSE)
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_loss_t', 'train_loss_r', 'train_trans_RMSE_m', 'val_loss', 'val_loss_t', 'val_loss_r', 'val_trans_RMSE_m', 'timestamp'])

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        # accumulators for components
        train_loss_t_sum = 0.0
        train_loss_r_sum = 0.0
        train_batches = 0
        for voxels, imu_feats, imu_ts, targets in loop:
            voxels = voxels.to(DEVICE)
            imu_feats = imu_feats.to(DEVICE)
            targets = targets.to(DEVICE)

            targets_scaled = targets.clone()
            targets_scaled[..., :3] = targets_scaled[..., :3] * TRANS_SCALE_FACTOR

            preds = model(voxels, imu_feats)
            # compute component losses
            pred_t = preds[..., :3]
            pred_r = preds[..., 3:]
            target_t = targets_scaled[..., :3]
            target_r = targets_scaled[..., 3:]
            loss_t = HUBER_LOSS_FN(pred_t, target_t)
            loss_r = HUBER_LOSS_FN(pred_r, target_r)
            loss = loss_t + ROT_WEIGHT * loss_r

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step()

            total_loss += loss.item()
            train_loss_t_sum += loss_t.item()
            train_loss_r_sum += loss_r.item()
            train_batches += 1
            loop.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_train = total_loss / len(train_loader)
        # component averages
        avg_train_t = train_loss_t_sum / max(1, train_batches)
        avg_train_r = train_loss_r_sum / max(1, train_batches)

        # Validate and get component metrics
        avg_val, avg_val_t, avg_val_r, val_rmse = validate(model, val_loader, DEVICE)

        # Log to CSV
        ts = time.time()
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f'{avg_train:.6f}', f'{avg_train_t:.6f}', f'{avg_train_r:.6f}', f'{0.0:.6f}', f'{avg_val:.6f}', f'{avg_val_t:.6f}', f'{avg_val_r:.6f}', f'{val_rmse:.6f}', ts])

        # Print summary
        print(f'-- Epoch {epoch+1}: Train Loss = {avg_train:.6f}, Val Loss = {avg_val:.6f}')

        # Scheduler
        scheduler.step()

        # Save every 5 epochs
        if (epoch + 1) % 5 == 0:
            path = os.path.join(CKPT_DIR, f'vo_model_1train_2val_ep{epoch+1}.pth')
            torch.save(model.state_dict(), path)
            print(f'  Saved checkpoint: {path}')

        # Save best
        if avg_val < best_val:
            best_val = avg_val
            best_path = os.path.join(CKPT_DIR, 'vo_model_1train_2val_best.pth')
            torch.save(model.state_dict(), best_path)
            print(f'  NEW BEST (val={avg_val:.6f}): {best_path}')

    print('\nTraining completed')

if __name__ == '__main__':
    train()
