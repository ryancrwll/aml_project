"""
Compute optimal ROT_WEIGHT by inspecting the data distribution.
ROT_WEIGHT should balance translation and rotation loss magnitudes.
"""
import torch
from torch.utils.data import DataLoader
from dataloader import MVSECDataset
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRANS_SCALE = 1000.0

# Load datasets
train_ds = MVSECDataset('./data/indoor_flying1_data.hdf5', './data/indoor_flying1_gt.hdf5', seq_len=10, crop_params={'H':260,'W':346,'B':5}, use_stereo=True, use_calib=True, calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
val_ds = MVSECDataset('./data/indoor_flying2_data.hdf5', './data/indoor_flying2_gt.hdf5', seq_len=10, crop_params={'H':260,'W':346,'B':5}, use_stereo=True, use_calib=True, calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')

print('Computing statistics from training data...')
all_t = []
all_r = []
for batch in DataLoader(train_ds, batch_size=8, num_workers=4):
    targets = batch[3]  # (B, S, 6)
    all_t.append(targets[..., :3])
    all_r.append(targets[..., 3:])

all_t = torch.cat(all_t, dim=0)  # (N, S, 3)
all_r = torch.cat(all_r, dim=0)  # (N, S, 3)

# Scale targets like in training
all_t_scaled = all_t * TRANS_SCALE

# Compute magnitudes
t_mag = torch.norm(all_t_scaled, dim=-1).mean().item()  # mean magnitude of scaled translation
r_mag = torch.norm(all_r, dim=-1).mean().item()  # mean magnitude of rotation

# std for loss scaling
t_std = torch.norm(all_t_scaled, dim=-1).std().item()
r_std = torch.norm(all_r, dim=-1).std().item()

print(f'Scaled translation: mean={t_mag:.6f}, std={t_std:.6f}')
print(f'Rotation: mean={r_mag:.6f}, std={r_std:.6f}')

# Compute optimal ROT_WEIGHT to balance losses
# If translation has mean magnitude t_mag and rotation has mean magnitude r_mag,
# then to make MSE roughly equal: ROT_WEIGHT * (r_error)^2 ~ (t_error)^2
# Assuming both have similar error magnitudes (% of their scale):
# ROT_WEIGHT ~ (t_mag / r_mag)^2
rot_weight_balanced = (t_mag / max(r_mag, 1e-6)) ** 2
print(f'\nSuggested ROT_WEIGHT to balance losses: {rot_weight_balanced:.2f}')
print(f'This assumes translation and rotation errors are proportional to their magnitudes.')

# More conservative: use sqrt instead of ^2
rot_weight_conservative = t_mag / max(r_mag, 1e-6)
print(f'Conservative ROT_WEIGHT (sqrt scaling): {rot_weight_conservative:.2f}')

# Even more conservative: just use a small fixed weight
print(f'\nAlternative: use fixed small ROT_WEIGHT (e.g., 10-50) to prioritize translation learning.')
