#!/usr/bin/env python3
"""
Comprehensive training on all 4 flying datasets with proper scale factor alignment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import datetime

from dataloader import MVSECDataset
from VOnetwork import VONet

# === Configuration ===
TRAIN_DATASETS = [
    {'data': './data/indoor_flying1_data.hdf5', 'gt': './data/indoor_flying1_gt.hdf5'},
    {'data': './data/indoor_flying2_data.hdf5', 'gt': './data/indoor_flying2_gt.hdf5'},
    {'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
    {'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'}
]

BATCH_SIZE = 8
SEQ_LEN = 10
EPOCHS = 50
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
ROT_WEIGHT = 300.0
TRANS_SCALE_FACTOR = 1000.0  # CRITICAL: Must match eval in main.py

USE_STEREO = True
USE_CALIB = True
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'

HUBER_LOSS_FN = nn.SmoothL1Loss(reduction='mean', beta=1.0)
GRAD_CLIP_VALUE = 1.0

def pose_loss(pred, target, rot_weight, trans_scale):
    """Compute pose loss with proper scaling."""
    pred_t = pred[..., :3]
    pred_r = pred[..., 3:]
    target_t = target[..., :3]
    target_r = target[..., 3:]

    # Scale predictions to match GT scale
    pred_t_scaled = pred_t / trans_scale

    # Translation loss
    loss_t = HUBER_LOSS_FN(pred_t_scaled, target_t)

    # Rotation loss
    loss_r = HUBER_LOSS_FN(pred_r, target_r)

    return loss_t + rot_weight * loss_r

def load_datasets():
    """Load all training datasets."""
    datasets = []
    total_seqs = 0

    for i, cfg in enumerate(TRAIN_DATASETS):
        dataset_name = cfg['data'].split('_')[2].split('.')[0]  # e.g., 'flying1'
        try:
            ds = MVSECDataset(
                data_path=cfg['data'],
                gt_path=cfg['gt'],
                seq_len=SEQ_LEN,
                crop_params=VOXEL_PARAMS,
                use_stereo=USE_STEREO,
                use_calib=USE_CALIB,
                calib_path=CALIB_PATH
            )
            datasets.append(ds)
            total_seqs += len(ds)
            print(f"  {dataset_name}: {len(ds)} sequences")
        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")

    if not datasets:
        raise RuntimeError("No datasets loaded!")

    combined = ConcatDataset(datasets)
    return combined, total_seqs

def train():
    """Main training loop."""
    print("=" * 80)
    print("COMPREHENSIVE TRAINING - ALL 4 FLYING DATASETS")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Stereo: {USE_STEREO}, Calibration: {USE_CALIB}")
    print(f"TRANS_SCALE_FACTOR: {TRANS_SCALE_FACTOR}")
    print(f"ROT_WEIGHT: {ROT_WEIGHT}")
    print()

    # Load datasets
    print("Loading datasets...")
    combined_ds, total_seqs = load_datasets()
    print(f"Total training sequences: {total_seqs}\n")

    # Create loader
    loader = DataLoader(combined_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)

    # Initialize model
    input_channels = 10 if USE_STEREO else 5
    model = VONet(input_channels=input_channels).to(DEVICE)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Training loop
    best_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for voxels, imu_feats, imu_ts, targets in loop:
            voxels = voxels.to(DEVICE)
            imu_feats = imu_feats.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            preds = model(voxels, imu_feats)

            # Compute loss
            loss = pose_loss(preds, targets, rot_weight=ROT_WEIGHT, trans_scale=TRANS_SCALE_FACTOR)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix({'loss': f'{loss.item():.6f}'})

        avg_loss = total_loss / len(loader)
        print(f"  Average loss: {avg_loss:.6f}")

        # Learning rate schedule
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            ckpt_path = f'./checkpoints/vo_model_all4_ep{epoch+1}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_ckpt = './checkpoints/vo_model_all4_best.pth'
                torch.save(model.state_dict(), best_ckpt)
                print(f"  NEW BEST: {best_ckpt} (loss={avg_loss:.6f})")

        print()

    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    train()
