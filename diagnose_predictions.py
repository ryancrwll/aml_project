import torch
from torch.utils.data import DataLoader
from dataloader import MVSECDataset
from VOnetwork import VONet
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANS_SCALE_FACTOR = 1000.0

# Load finetuned best checkpoint
ckpt_path = './checkpoints/vo_model_1train_2val_finetuned_best.pth'
model = VONet(input_channels=10).to(DEVICE)
try:
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print('Loaded finetuned checkpoint')
except:
    print('Could not load finetuned checkpoint, trying baseline best')
    ckpt_path = './checkpoints/vo_model_1train_2val_best.pth'
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    print('Loaded baseline best checkpoint')

# Load a sample from dataset
ds = MVSECDataset('./data/indoor_flying1_data.hdf5', './data/indoor_flying1_gt.hdf5', seq_len=10, crop_params={'H':260,'W':346,'B':5}, use_stereo=True, use_calib=True, calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
loader = DataLoader(ds, batch_size=2, shuffle=False)
voxels, imu_feats, imu_ts, targets = next(iter(loader))
voxels = voxels.to(DEVICE)
imu_feats = imu_feats.to(DEVICE)
targets = targets.to(DEVICE)

print('\n=== DIAGNOSTIC ===')
print('Voxel shape:', voxels.shape)
print('Target shape (unscaled):', targets.shape)
print('Target min/max/mean:', targets.min().item(), targets.max().item(), targets.mean().item())

model.eval()
with torch.no_grad():
    preds = model(voxels, imu_feats)

print('Pred shape:', preds.shape)
print('Pred min/max/mean:', preds.min().item(), preds.max().item(), preds.mean().item())

# Descale predictions to meters
pred_t_m = preds[..., :3] / TRANS_SCALE_FACTOR  # B, S, 3 in meters
target_t_m = targets[..., :3]

pred_r = preds[..., 3:]
target_r = targets[..., 3:]

print('\nTranslation (meters):')
print('  Pred stats: min/max/mean/std:', pred_t_m.min().item(), pred_t_m.max().item(), pred_t_m.mean().item(), pred_t_m.std().item())
print('  Target stats: min/max/mean/std:', target_t_m.min().item(), target_t_m.max().item(), target_t_m.mean().item(), target_t_m.std().item())
print('  Pred sample (first batch, first step):', pred_t_m[0, 0].cpu().numpy())
print('  Target sample (first batch, first step):', target_t_m[0, 0].cpu().numpy())

# Per-step magnitude
pred_mag = torch.norm(pred_t_m, dim=-1).cpu().numpy()  # (B, S)
target_mag = torch.norm(target_t_m, dim=-1).cpu().numpy()
print('  Pred magnitude - mean/std:', pred_mag.mean(), pred_mag.std())
print('  Target magnitude - mean/std:', target_mag.mean(), target_mag.std())

print('\nRotation (radians):')
print('  Pred stats: min/max/mean/std:', pred_r.min().item(), pred_r.max().item(), pred_r.mean().item(), pred_r.std().item())
print('  Target stats: min/max/mean/std:', target_r.min().item(), target_r.max().item(), target_r.mean().item(), target_r.std().item())

# MSE in physical units
mse_trans = ((pred_t_m - target_t_m) ** 2).mean().item()
mse_rot = ((pred_r - target_r) ** 2).mean().item()
print('\nMSE (in physical units):')
print('  Translation MSE (m^2):', mse_trans, ' → RMSE (m):', np.sqrt(mse_trans))
print('  Rotation MSE (rad^2):', mse_rot, ' → RMSE (rad):', np.sqrt(mse_rot))

print('\nConclusion:')
if pred_mag.mean() < 0.001 and target_mag.mean() > 0.0001:
    print('  MODEL IS STILL PREDICTING NEAR-ZERO TRANSLATION! Not learning motion.')
elif abs(pred_mag.mean() - target_mag.mean()) > 5 * target_mag.std():
    print('  MODEL PREDICTIONS HAVE DIFFERENT SCALE/DISTRIBUTION THAN TARGET. Scale mismatch.')
else:
    print('  Model predictions show reasonable magnitudes. Issue is likely loss landscape or hyperparams.')
