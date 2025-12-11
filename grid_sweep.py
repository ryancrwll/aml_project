"""
Grid sweep: short runs (10 epochs) for LR x ROT_WEIGHT combinations.
Each run starts from `vo_model_1train_2val_best.pth` and writes its own CSV and checkpoint.
Runs sequentially to avoid concurrent GPU overload.
"""
import os, time, csv
from itertools import product
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from dataloader import MVSECDataset
from VOnetwork import VONet

# configs
LR_LIST = [5e-5, 1e-4, 2e-4]
ROT_LIST = [100.0, 300.0, 600.0]
EPOCHS = 10
BATCH=8
SEQ_LEN=10
USE_STEREO=True
USE_CALIB=True
NUM_WORKERS=4
TRANS_SCALE_FACTOR=1000.0

CKPT_DIR = './checkpoints'
LOG_DIR = './logs'
os.makedirs(LOG_DIR, exist_ok=True)

HUBER = nn.SmoothL1Loss(reduction='mean', beta=1.0)

# datasets
train_ds = MVSECDataset('./data/indoor_flying1_data.hdf5','./data/indoor_flying1_gt.hdf5',seq_len=SEQ_LEN,crop_params={'H':260,'W':346,'B':5},use_stereo=USE_STEREO,use_calib=USE_CALIB,calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
val_ds = MVSECDataset('./data/indoor_flying2_data.hdf5','./data/indoor_flying2_gt.hdf5',seq_len=SEQ_LEN,crop_params={'H':260,'W':346,'B':5},use_stereo=USE_STEREO,use_calib=USE_CALIB,calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_base = os.path.join(CKPT_DIR,'vo_model_1train_2val_best.pth')

for lr, rot in product(LR_LIST, ROT_LIST):
    tag = f'lr{lr:.0e}_rot{int(rot)}'
    print('Starting run', tag)
    model = VONet(input_channels=10).to(device)
    if os.path.exists(best_base):
        model.load_state_dict(torch.load(best_base,map_location=device))
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)
    log_csv = os.path.join(LOG_DIR, f'grid_{tag}.csv')
    with open(log_csv,'w',newline='') as f: csv.writer(f).writerow(['epoch','train_loss','train_loss_t','train_loss_r','val_loss','val_loss_t','val_loss_r','val_RMSE','ts'])
    best_val = 1e9
    for ep in range(EPOCHS):
        model.train(); t_loss=0.0; t_loss_t=0.0; t_loss_r=0.0; nb=0
        for vox, imu, imu_ts, targets in train_loader:
            vox=vox.to(device); imu=imu.to(device); targets=targets.to(device)
            targets_scaled = targets.clone(); targets_scaled[..., :3] *= TRANS_SCALE_FACTOR
            preds = model(vox, imu)
            loss_t = HUBER(preds[..., :3], targets_scaled[..., :3])
            loss_r = HUBER(preds[..., 3:], targets_scaled[..., 3:])
            loss = loss_t + rot * loss_r
            opt.zero_grad(); loss.backward(); opt.step()
            t_loss += loss.item(); t_loss_t += loss_t.item(); t_loss_r += loss_r.item(); nb+=1
        sched.step()
        avg_train = t_loss / max(1,nb)
        # validate
        model.eval(); v_loss=0.0; v_t=0.0; v_r=0.0; vb=0; sq=0.0; cnt=0
        with torch.no_grad():
            for vox, imu, imu_ts, targets in val_loader:
                vox=vox.to(device); imu=imu.to(device); targets=targets.to(device)
                targets_scaled = targets.clone(); targets_scaled[..., :3] *= TRANS_SCALE_FACTOR
                preds = model(vox, imu)
                loss_t = HUBER(preds[..., :3], targets_scaled[..., :3])
                loss_r = HUBER(preds[..., 3:], targets_scaled[..., 3:])
                loss = loss_t + rot * loss_r
                v_loss += loss.item(); v_t += loss_t.item(); v_r += loss_r.item(); vb+=1
                diff = (preds[..., :3]/TRANS_SCALE_FACTOR - targets[..., :3]).cpu().numpy()
                sq += (diff**2).sum(); cnt += diff.size
        avg_val = v_loss / max(1,vb); avg_vt = v_t/max(1,vb); avg_vr = v_r/max(1,vb); val_rmse = (sq/max(1,cnt))**0.5
        with open(log_csv,'a',newline='') as f: csv.writer(f).writerow([ep+1,f'{avg_train:.6f}',f'{t_loss_t/max(1,nb):.6f}',f'{t_loss_r/max(1,nb):.6f}',f'{avg_val:.6f}',f'{avg_vt:.6f}',f'{avg_vr:.6f}',f'{val_rmse:.6f}',time.time()])
        print(f'Grid {tag} Ep{ep+1}: Train {avg_train:.4f} Val {avg_val:.4f} ValRMSE {val_rmse:.6f}')
        if avg_val < best_val:
            best_val = avg_val
            torch.save(model.state_dict(), os.path.join(CKPT_DIR, f'grid_{tag}_best.pth'))
    print('Finished run', tag)

print('All grid runs completed')
