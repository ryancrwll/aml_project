import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import os

from dataloader import MVSECDataset
from VOnetwork import VONet

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH=4
SEQ_LEN=10
TRANS_SCALE_FACTOR=1000.0
ROT_WEIGHT=300.0

# find latest 1train_2val checkpoint (best preferred)
ckpt_dir='./checkpoints'
best_path=os.path.join(ckpt_dir,'vo_model_1train_2val_best.pth')
if not os.path.exists(best_path):
    # fallback to highest ep file
    eps=[p for p in os.listdir(ckpt_dir) if p.startswith('vo_model_1train_2val_ep')]
    if eps:
        eps_sorted=sorted(eps, key=lambda x: int(x.split('ep')[-1].split('.pth')[0]))
        best_path=os.path.join(ckpt_dir, eps_sorted[-1])
    else:
        best_path=None

print('Using checkpoint:', best_path)

# load dataset (train flying1 to inspect)
try:
    ds=MVSECDataset(data_path='./data/indoor_flying1_data.hdf5', gt_path='./data/indoor_flying1_gt.hdf5', seq_len=SEQ_LEN, crop_params={'H':260,'W':346,'B':5}, use_stereo=True, use_calib=True, calib_path='./data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml')
    print('Dataset size:', len(ds))
except Exception as e:
    print('Dataset load error', e)
    raise

loader=DataLoader(ds, batch_size=BATCH, shuffle=False)
voxels, imu_feats, imu_ts, targets = next(iter(loader))
print('Batch shapes:', voxels.shape, imu_feats.shape, imu_ts.shape, targets.shape)

model=VONet(input_channels=10).to(DEVICE)
if best_path:
    st=torch.load(best_path, map_location=DEVICE)
    try:
        model.load_state_dict(st)
        print('Loaded checkpoint')
    except Exception as e:
        print('Checkpoint load error', e)

voxels=voxels.to(DEVICE)
imu_feats=imu_feats.to(DEVICE)
targets=targets.to(DEVICE)

# forward
model.train()
preds=model(voxels, imu_feats)
print('Preds shape:', preds.shape)
print('Preds stats: mean/std/min/max:', preds.mean().item(), preds.std().item(), preds.min().item(), preds.max().item())
print('Targets stats (unscaled): mean/std/min/max:', targets.mean().item(), targets.std().item(), targets.min().item(), targets.max().item())

# compute translation & rotation parts before scaling
pred_t=preds[..., :3]
pred_r=preds[..., 3:]
target_t=targets[..., :3]
target_r=targets[..., 3:]

# scale targets
target_t_scaled=target_t * TRANS_SCALE_FACTOR

loss_fn=nn.SmoothL1Loss(reduction='mean', beta=1.0)
loss_t=loss_fn(pred_t, target_t_scaled)
loss_r=loss_fn(pred_r, target_r)
loss_total=loss_t + ROT_WEIGHT * loss_r
print('Loss components: loss_t, loss_r, total:', loss_t.item(), loss_r.item(), loss_total.item())

# Backward check: compute gradients and optimizer step
opt=optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
opt.zero_grad()
loss_total.backward()

# gradient norms
max_grad=0.0
mean_grad=0.0
count=0
for n,p in model.named_parameters():
    if p.grad is None:
        continue
    gnorm=p.grad.data.norm().item()
    mean_grad += gnorm
    max_grad = max(max_grad, gnorm)
    count += 1
    print(f'grad {n}: norm={gnorm:.6f}')
if count>0:
    mean_grad /= count
print('Mean grad norm:', mean_grad, 'Max grad norm:', max_grad)

# param change test
before=[p.data.clone() for p in model.parameters()]
opt.step()
after=[p.data for p in model.parameters()]
# compute avg param delta
deltas=[(a-b).norm().item() for a,b in zip(after,before)]
print('Param delta stats: mean/min/max:', sum(deltas)/len(deltas), min(deltas), max(deltas))

print('\nDone diagnostic')
