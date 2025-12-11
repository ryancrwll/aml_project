"""
Fine-tune from best checkpoint `vo_model_1train_2val_best.pth` with new hyperparams.
LR=5e-5, ROT_WEIGHT=200, EPOCHS=30
Logs to `logs/finetune_1train_2val_epoch_logs.csv` and stdout `train_finetune_from_best.log`.
"""
import os, time, csv
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import MVSECDataset
from VOnetwork import VONet

# Config
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
EPOCHS = 30
LR = 5e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP_VALUE = 1.0
ROT_WEIGHT = 200.0
TRANS_SCALE_FACTOR = 1000.0
SEQ_LEN = 10
NUM_WORKERS = 4
USE_STEREO = True
USE_CALIB = True
VOXEL_PARAMS = {'H':260,'W':346,'B':5}

LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)
LOG_CSV = os.path.join(LOG_DIR, 'finetune_1train_2val_epoch_logs.csv')
CKPT_DIR = './checkpoints'

HUBER_LOSS_FN = nn.SmoothL1Loss(reduction='mean', beta=1.0)

# Load datasets (same pair)
def load_dataset_pair():
    train_ds = MVSECDataset('./data/indoor_flying1_data.hdf5','./data/indoor_flying1_gt.hdf5',seq_len=SEQ_LEN,crop_params=VOXEL_PARAMS,use_stereo=USE_STEREO,use_calib=USE_CALIB,calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
    val_ds = MVSECDataset('./data/indoor_flying2_data.hdf5','./data/indoor_flying2_gt.hdf5',seq_len=SEQ_LEN,crop_params=VOXEL_PARAMS,use_stereo=USE_STEREO,use_calib=USE_CALIB,calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
    return train_ds, val_ds

# validate returns components and RMSE (copied logic)
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
            pred_t = preds[..., :3]
            pred_r = preds[..., 3:]
            target_t_scaled = targets_scaled[..., :3]
            target_r = targets_scaled[..., 3:]
            loss_t = HUBER_LOSS_FN(pred_t, target_t_scaled)
            loss_r = HUBER_LOSS_FN(pred_r, target_r)
            loss = loss_t + ROT_WEIGHT * loss_r
            total_loss += loss.item(); total_loss_t += loss_t.item(); total_loss_r += loss_r.item(); batches += 1
            pred_t_m = pred_t / TRANS_SCALE_FACTOR
            target_t_m = targets[..., :3]
            diff = (pred_t_m - target_t_m).cpu().numpy()
            sq_err_sum += (diff ** 2).sum(); count_pts += diff.size
    avg_loss = total_loss / max(1,batches)
    avg_loss_t = total_loss_t / max(1,batches)
    avg_loss_r = total_loss_r / max(1,batches)
    val_rmse = (sq_err_sum / max(1,count_pts)) ** 0.5
    return avg_loss, avg_loss_t, avg_loss_r, val_rmse

# Training
if __name__ == '__main__':
    print('Fine-tune: LR=',LR,' ROT_WEIGHT=',ROT_WEIGHT)
    train_ds, val_ds = load_dataset_pair()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
    model = VONet(input_channels=10).to(DEVICE)

    best_ckpt = os.path.join(CKPT_DIR, 'vo_model_1train_2val_best.pth')
    if os.path.exists(best_ckpt):
        st = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(st)
        print('Loaded best checkpoint for finetune')

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # CSV header
    with open(LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_loss_t','train_loss_r','val_loss','val_loss_t','val_loss_r','val_trans_RMSE_m','timestamp'])

    best_val = 1e9
    for epoch in range(EPOCHS):
        model.train(); total_loss=0.0; t_loss_t=0.0; t_loss_r=0.0; batches=0
        loop = tqdm(train_loader, desc=f'Finetune Epoch {epoch+1}/{EPOCHS}')
        for voxels, imu_feats, imu_ts, targets in loop:
            voxels = voxels.to(DEVICE); imu_feats = imu_feats.to(DEVICE); targets = targets.to(DEVICE)
            targets_scaled = targets.clone(); targets_scaled[..., :3] = targets_scaled[..., :3] * TRANS_SCALE_FACTOR
            preds = model(voxels, imu_feats)
            pred_t = preds[..., :3]; pred_r = preds[..., 3:]
            loss_t = HUBER_LOSS_FN(pred_t, targets_scaled[..., :3])
            loss_r = HUBER_LOSS_FN(pred_r, targets_scaled[..., 3:])
            loss = loss_t + ROT_WEIGHT * loss_r
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE); optimizer.step()
            total_loss += loss.item(); t_loss_t += loss_t.item(); t_loss_r += loss_r.item(); batches += 1
        avg_train = total_loss / max(1,len(train_loader)); avg_train_t = t_loss_t/max(1,batches); avg_train_r = t_loss_r/max(1,batches)
        avg_val, avg_val_t, avg_val_r, val_rmse = validate(model, val_loader, DEVICE)
        ts=time.time()
        with open(LOG_CSV,'a',newline='') as f: csv.writer(f).writerow([epoch+1,f'{avg_train:.6f}',f'{avg_train_t:.6f}',f'{avg_train_r:.6f}',f'{avg_val:.6f}',f'{avg_val_t:.6f}',f'{avg_val_r:.6f}',f'{val_rmse:.6f}',ts])
        print(f'Finetune Epoch {epoch+1}: Train {avg_train:.6f} Val {avg_val:.6f} Val_RMSE {val_rmse:.6f}')
        scheduler.step()
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(CKPT_DIR,'vo_model_1train_2val_finetuned_best.pth'))
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'vo_model_1train_2val_finetuned_ep{epoch+1}.pth'))

    print('Finetune completed')
