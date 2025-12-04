import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os
import sys
import numpy as np

# Ensure local modules (dataloader.py, network.py) are accessible
sys.path.append(os.path.dirname(__file__))

from dataloader import MVSECDataset
from VOnetwork import VONet

# --- Configuration ---
TRAIN_DATASETS = [
    {'data': './data/indoor_flying2_data.hdf5', 'gt': './data/indoor_flying2_gt.hdf5'},
    {'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
    {'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'}
]

BATCH_SIZE = 4
SEQ_LEN = 5
EPOCHS = 50
LR = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
ROT_WEIGHT = 200.0
TRANS_SCALE_FACTOR = 1000.0
GRAD_CLIP_VALUE = 1.0

def pose_loss(pred, target, rot_weight, trans_scale):
    target_t_scaled = target[..., :3] * trans_scale
    pred_t_scaled = pred[..., :3]
    loss_t = nn.functional.mse_loss(pred_t_scaled, target_t_scaled)
    loss_r = nn.functional.mse_loss(pred[..., 3:], target[..., 3:])
    return loss_t + rot_weight * loss_r

def get_gradient_stats(model):
    """Compute gradient statistics for diagnostics."""
    grad_norms = []
    for p in model.parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.data.norm().item())

    if not grad_norms:
        return 0.0, 0.0, 0.0

    grad_norms = np.array(grad_norms)
    return grad_norms.mean(), grad_norms.max(), grad_norms.min()

def train():
    print(f"Starting DIAGNOSTIC training on {DEVICE}")

    # 1. Load Datasets
    datasets = []
    for paths in TRAIN_DATASETS:
        d_path = paths['data']
        g_path = paths['gt']
        if not os.path.exists(d_path) or not os.path.exists(g_path):
            continue
        try:
            ds = MVSECDataset(data_path=d_path, gt_path=g_path, seq_len=SEQ_LEN, crop_params=VOXEL_PARAMS)
            datasets.append(ds)
            print(f"  Loaded: {d_path} ({len(ds)} sequences)")
        except Exception as e:
            print(f"  Failed: {d_path} - {e}")

    if not datasets:
        print("FATAL: No datasets loaded. Check your file paths.")
        return

    combined_dataset = ConcatDataset(datasets)
    print(f"Total training sequences: {len(combined_dataset)}\n")

    loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)

    # 2. Model Setup
    model = VONet(input_channels=VOXEL_PARAMS['B']).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 3. Training Loop with Diagnostics
    loss_history = []
    grad_mean_history = []
    grad_max_history = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        epoch_grad_means = []
        epoch_grad_maxs = []

        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch_idx, (voxels, targets) in enumerate(loop):
            if voxels.numel() == 0 or voxels.size(1) != SEQ_LEN:
                continue

            voxels, targets = voxels.to(DEVICE), targets.to(DEVICE)

            # Diagnostics: Check input data
            if batch_idx == 0:
                print(f"\n  [Batch 0] Voxel range: [{voxels.min():.4f}, {voxels.max():.4f}]")
                print(f"  [Batch 0] Target trans range: [{targets[..., :3].min():.6f}, {targets[..., :3].max():.6f}]")
                print(f"  [Batch 0] Target rot range: [{targets[..., 3:].min():.6f}, {targets[..., 3:].max():.6f}]")

            optimizer.zero_grad()
            preds = model(voxels)

            # Diagnostics: Check prediction range
            if batch_idx == 0:
                print(f"  [Batch 0] Pred trans range: [{preds[..., :3].min():.4f}, {preds[..., :3].max():.4f}]")
                print(f"  [Batch 0] Pred rot range: [{preds[..., 3:].min():.6f}, {preds[..., 3:].max():.6f}]")

            loss = pose_loss(preds, targets, rot_weight=ROT_WEIGHT, trans_scale=TRANS_SCALE_FACTOR)

            loss.backward()

            # Gradient statistics BEFORE clipping
            grad_mean, grad_max, grad_min = get_gradient_stats(model)
            epoch_grad_means.append(grad_mean)
            epoch_grad_maxs.append(grad_max)

            if batch_idx == 0:
                print(f"  [Batch 0] Gradients before clip - Mean: {grad_mean:.6f}, Max: {grad_max:.6f}, Min: {grad_min:.6f}")

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)

            # Gradient statistics AFTER clipping
            grad_mean_after, grad_max_after, _ = get_gradient_stats(model)
            if batch_idx == 0:
                print(f"  [Batch 0] Gradients after clip - Mean: {grad_mean_after:.6f}, Max: {grad_max_after:.6f}")

            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=total_loss / (batch_idx + 1))

        avg_loss = total_loss / max(1, len(loader))
        avg_grad_mean = np.mean(epoch_grad_means) if epoch_grad_means else 0.0
        avg_grad_max = np.mean(epoch_grad_maxs) if epoch_grad_maxs else 0.0

        loss_history.append(avg_loss)
        grad_mean_history.append(avg_grad_mean)
        grad_max_history.append(avg_grad_max)

        print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.6f} | Grad Mean: {avg_grad_mean:.6f} | Grad Max: {avg_grad_max:.6f}\n")

        # Check for stagnation: loss hasn't improved in last 5 epochs
        if epoch >= 5:
            recent_losses = loss_history[-5:]
            loss_improvement = max(recent_losses) - min(recent_losses)
            if loss_improvement < 1e-5:
                print(f"⚠️  WARNING: Loss appears stagnant! Last 5 losses: {[f'{l:.6f}' for l in recent_losses]}")
                print(f"    Improvement: {loss_improvement:.6e}\n")

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f'checkpoints/vo_model_ep{epoch+1}.pth')

    # Final Summary
    print("\n=== TRAINING COMPLETE ===")
    print(f"Final Loss: {loss_history[-1]:.6f}")
    print(f"Loss Range: [{min(loss_history):.6f}, {max(loss_history):.6f}]")
    print(f"Avg Gradient Mean: {np.mean(grad_mean_history):.6f}")
    print(f"Avg Gradient Max: {np.mean(grad_max_history):.6f}")

if __name__ == "__main__":
    train()
