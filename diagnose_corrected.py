import torch
from torch.utils.data import DataLoader
from dataloader import MVSECDataset
from VOnetwork import VONet
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANS_SCALE_FACTOR = 1000.0
ckpt = './checkpoints/vo_model_corrected_best.pth'
if not torch.cuda.is_available():
    map_loc='cpu'
else:
    map_loc=DEVICE

print('Loading model:', ckpt)
model = VONet(input_channels=10).to(DEVICE)
st = torch.load(ckpt, map_location=map_loc)
model.load_state_dict(st)
print('Loaded')

# dataset
ds = MVSECDataset('./data/indoor_flying1_data.hdf5','./data/indoor_flying1_gt.hdf5',seq_len=10,crop_params={'H':260,'W':346,'B':5},use_stereo=True,use_calib=True,calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
loader = DataLoader(ds, batch_size=2, shuffle=False)
vox, imu, imu_ts, targets = next(iter(loader))
vox = vox.to(DEVICE); imu=imu.to(DEVICE); targets=targets.to(DEVICE)

model.eval()
with torch.no_grad():
    preds = model(vox, imu)

# stats
print('pred shape', preds.shape, 'targets shape', targets.shape)
pred_t = preds[..., :3]
pred_r = preds[..., 3:]

target_t = targets[..., :3]
target_r = targets[..., 3:]

pred_t_m = pred_t / TRANS_SCALE_FACTOR

print('\nTranslation (meters)')
print('pred mean/std/min/max:', pred_t_m.mean().item(), pred_t_m.std().item(), pred_t_m.min().item(), pred_t_m.max().item())
print('tgt mean/std/min/max:', target_t.mean().item(), target_t.std().item(), target_t.min().item(), target_t.max().item())

pred_mag = torch.norm(pred_t_m, dim=-1).cpu().numpy()
tgt_mag = torch.norm(target_t, dim=-1).cpu().numpy()
print('pred mag mean/std:', pred_mag.mean(), pred_mag.std())
print('tgt mag mean/std:', tgt_mag.mean(), tgt_mag.std())

print('\nRotation (rad)')
print('pred mean/std/min/max:', pred_r.mean().item(), pred_r.std().item(), pred_r.min().item(), pred_r.max().item())
print('tgt mean/std/min/max:', target_r.mean().item(), target_r.std().item(), target_r.min().item(), target_r.max().item())

mse_t = ((pred_t_m - target_t)**2).mean().item()
mse_r = ((pred_r - target_r)**2).mean().item()
print('\nMSE trans (m^2):', mse_t, 'RMSE(m):', np.sqrt(mse_t))
print('MSE rot (rad^2):', mse_r, 'RMSE(rad):', np.sqrt(mse_r))

# show sample
print('\nSample first step:')
print('pred_t_m[0,0]:', pred_t_m[0,0].cpu().numpy())
print('tgt_t[0,0]:', target_t[0,0].cpu().numpy())
print('pred_r[0,0]:', pred_r[0,0].cpu().numpy())
print('tgt_r[0,0]:', target_r[0,0].cpu().numpy())
