import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os
import time
import csv
import sys

sys.path.append(os.path.dirname(__file__))

from dataloader import MVSECDataset
from VOnetwork import VONet
import argparse

TRAIN_DATASETS = [
    {'data': './data/indoor_flying1_data.hdf5', 'gt': './data/indoor_flying2_gt.hdf5'},
    {'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
    {'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'}
]

BATCH_SIZE = 8
SEQ_LEN = 10
EPOCHS = 50
# LR = 5e-5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
ROT_WEIGHT = 50.0
TRANS_SCALE_FACTOR = 7.5

GRAD_CLIP_VALUE = 1.0 # Standard value for gradient clipping

# --- Multi-camera and Calibration Configuration ---
USE_STEREO = True  # If True: uses both left+right cameras (10 channels). If False: left camera only (5 channels)
USE_CALIB = True   # If True: loads calibration data from YAML file. If False: skips calibration loading
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'
LOG_PATH = f'./logs/training_s{USE_STEREO}_c{USE_CALIB}'
HUBER_LOSS_FN = nn.SmoothL1Loss(reduction='mean', beta=1.0)

def pose_loss(pred, target, rot_weight, trans_scale):
    """
    Compute pose loss.
    Inputs:
      pred: (B, S, 6) network predictions in order [tx, ty, tz, d_roll, d_pitch, d_yaw]
      target: (B, S, 6) ground truth in order [dx, dy, dz, d_roll, d_pitch, d_yaw]
      trans_scale: scale factor to match prediction magnitude to GT
    Returns: total loss = loss_translation + rot_weight * loss_rotation
    """
    ############### STANDARD MSE LOSS ###########################
    # # Translation loss: scale predictions to match GT scale
    # target_t_scaled = target[..., :3] * trans_scale  # [B, S, 3]
    # pred_t_scaled = pred[..., :3]  # [B, S, 3]
    # loss_t = nn.functional.mse_loss(pred_t_scaled, target_t_scaled)

    # # Rotation loss: compare Euler angles directly
    # loss_r = nn.functional.mse_loss(pred[..., 3:], target[..., 3:])  # [B, S, 3]
    #############################################################

    ################### HUBER LOSS ##############################
    # 1. Translation loss (Smooth L1)
    target_t_scaled = target[..., :3] * trans_scale  # [B, S, 3]
    pred_t_scaled = pred[..., :3]                   # [B, S, 3]

    # Use Huber Loss for translation
    loss_t = HUBER_LOSS_FN(pred_t_scaled, target_t_scaled)

    # 2. Rotation loss (Smooth L1)
    # Use Huber Loss for rotation
    loss_r = HUBER_LOSS_FN(pred[..., 3:], target[..., 3:])  # [B, S, 3]
    #############################################################

    return loss_t + rot_weight * loss_r, loss_t/loss_r

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

    model = VONet(input_channels=input_channels, use_stereo=USE_STEREO).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        losses_ratio = 0
        time_start = time.time()
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for voxels, imu_feats, imu_ts, targets in loop:
            if voxels.numel() == 0 or voxels.size(1) != SEQ_LEN:
                continue

            voxels, imu_feats, imu_ts, targets = voxels.to(DEVICE), imu_feats.to(DEVICE), imu_ts.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(voxels, imu_feats)
            loss, ratio = pose_loss(preds, targets, rot_weight=ROT_WEIGHT, trans_scale=TRANS_SCALE_FACTOR)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)

            optimizer.step()

            total_loss += loss.item()
            losses_ratio += ratio.item()
            loop.set_postfix(loss=total_loss / (loop.n + 1))

        avg_loss = total_loss / len(loader)
        avg_ratio = losses_ratio / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}\n")

        # --- NEW LOGGING CODE ---
        with open(LOG_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f'{avg_loss:.6f}', f'{avg_ratio}', time_start-time.time()])

        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
            torch.save(model.state_dict(), f'checkpoints/vo_model_ep{epoch+1}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train / Fine-tune VO model')
    parser.add_argument('--datasets', default='2,3,4', help='Comma-separated dataset ids to use (1..4). Defaults to 2,3,4')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--resume', default=None, help='Path to checkpoint to resume / fine-tune from')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze feature extractor when fine-tuning')
    parser.add_argument('--use-stereo', action='store_true', dest='use_stereo')
    parser.add_argument('--no-stereo', action='store_false', dest='use_stereo')
    parser.set_defaults(use_stereo=USE_STEREO)
    parser.add_argument('--use-calib', action='store_true', dest='use_calib')
    parser.add_argument('--no-calib', action='store_false', dest='use_calib')
    parser.set_defaults(use_calib=USE_CALIB)
    parser.add_argument('--run', action='store_true', help='Start training immediately')

    args = parser.parse_args()

    # Map dataset ids to paths
    DS_MAP = {
        '1': {'data': './data/indoor_flying1_data.hdf5', 'gt': './data/indoor_flying1_gt.hdf5'},
        '2': {'data': './data/indoor_flying2_data.hdf5', 'gt': './data/indoor_flying2_gt.hdf5'},
        '3': {'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
        '4': {'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'}
    }

    # Override config from CLI
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    USE_STEREO = bool(args.use_stereo)
    USE_CALIB = bool(args.use_calib)

    # Build TRAIN_DATASETS from ids
    ids = [s.strip() for s in args.datasets.split(',') if s.strip()]
    TRAIN_DATASETS = [DS_MAP[i] for i in ids if i in DS_MAP]

    if args.run:
        # update globals used by train()
        globals().update({'TRAIN_DATASETS': TRAIN_DATASETS, 'BATCH_SIZE': BATCH_SIZE, 'EPOCHS': EPOCHS, 'LR': LR, 'USE_STEREO': USE_STEREO, 'USE_CALIB': USE_CALIB})

        def train_with_resume(resume_path=None, freeze_backbone=False):
            print(f"Starting training on {DEVICE}")
            print(f"Datasets: {ids}, Stereo: {USE_STEREO}, Calib: {USE_CALIB}")

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

            # Model init
            input_channels = VOXEL_PARAMS['B'] * 2 if USE_STEREO else VOXEL_PARAMS['B']
            model = VONet(input_channels=input_channels, use_stereo=USE_STEREO).to(DEVICE)

            # If resume checkpoint provided, load weights
            if resume_path and os.path.exists(resume_path):
                print(f"Loading pretrained weights from {resume_path} for fine-tuning...")
                model.load_state_dict(torch.load(resume_path, map_location=DEVICE))

            # Optionally freeze backbone (best-effort; avoids API assumptions)
            if freeze_backbone:
                for name, p in model.named_parameters():
                    if 'cnn' in name or 'conv' in name or 'encoder' in name:
                        p.requires_grad = False
                print("Backbone-encoder parameters frozen (best-effort)")

            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            # Training loop
            LOG_DIR = 'logs'
            LOG_PATH = os.path.join(LOG_DIR, f'training_log_{time.strftime("%Y%m%d-%H%M%S")}.csv')
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(LOG_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'avg_loss', 'loss_ratio', 'timestamp'])
            for epoch in range(EPOCHS):
                model.train()
                total_loss = 0
                avg_ratio = 0
                voxelsinloop = 0
                time_start = time.time()
                loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
                for voxels, imu_feats, imu_ts, targets in loop:
                    if voxels.numel() == 0 or voxels.size(1) != SEQ_LEN:
                        continue
                    voxelsinloop +=1
                    voxels, imu_feats, imu_ts, targets = voxels.to(DEVICE), imu_feats.to(DEVICE), imu_ts.to(DEVICE), targets.to(DEVICE)
                    optimizer.zero_grad()
                    preds = model(voxels, imu_feats)
                    loss, ratio = pose_loss(preds, targets, rot_weight=ROT_WEIGHT, trans_scale=TRANS_SCALE_FACTOR)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
                    optimizer.step()
                    avg_ratio += ratio.item()
                    total_loss += loss.item()
                    loop.set_postfix(loss=total_loss / (loop.n + 1))
                avg_ratio /= voxelsinloop
                avg_loss = total_loss / len(loader)
                with open(LOG_PATH, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, f'{avg_loss:.6f}', f'{avg_ratio}', time_start-time.time()])


                # Save every epoch for safety during long runs
                torch.save(model.state_dict(), f'checkpoints/vo_model_ep{epoch+1}.pth')

        # Call training with provided resume and freeze args
        train_with_resume(resume_path=args.resume, freeze_backbone=args.freeze_backbone)
    else:
        print("Train CLI parsed. Use --run to start training now. Example: python train.py --datasets 1,4 --resume ./checkpoints/vo_model_ep30.pth --epochs 5 --run")