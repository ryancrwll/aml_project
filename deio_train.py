import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import os

# Import new modules
from vo_deio_net import DEIONet, DifferentiableBundleAdjustment
from deio_loss import DEIOCost
from deio_data import DEIODataset

# --- Configuration (Must be aligned with DEIO requirements) ---
TRAIN_DATASETS = [
    {'data': './data/indoor_flying2_data.hdf5', 'gt': './data/indoor_flying2_gt.hdf5'},
    {'data': './data/indoor_flying3_data.hdf5', 'gt': './data/indoor_flying3_gt.hdf5'},
    {'data': './data/indoor_flying4_data.hdf5', 'gt': './data/indoor_flying4_gt.hdf5'}
]
USE_IMU_DATA = False
BATCH_SIZE = 4
SEQ_LEN = 10
EPOCHS = 30
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
USE_STEREO = False
USE_CALIB = False
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'

# DEIO Specific Parameters
IMU_RAW_STATE_DIM = 6
GT_FULL_STATE_DIM = 16

# DEIO Hyperparameters
EVENT_WEIGHT = 100.0
GT_WEIGHT = 1.0
PRIOR_WEIGHT = 0.1
HUBER_DELTA = 0.1
GRAD_CLIP_VALUE = 1.0
CHECKPOINT_DIR = './checkpoints'

def train():
    print(f"Starting DEIO training on {DEVICE}")

    datasets = []
    for paths in TRAIN_DATASETS:
        d_path = paths['data']
        g_path = paths['gt']
        if not os.path.exists(d_path) or not os.path.exists(g_path):
            print(f"  Warning: Skipping missing dataset {d_path}")
            continue
        try:
            ds = DEIODataset(
                data_path=d_path,
                gt_path=g_path,
                seq_len=SEQ_LEN,
                crop_params=VOXEL_PARAMS,
                use_stereo=USE_STEREO,
                use_calib=USE_CALIB,
                calib_path=CALIB_PATH if USE_CALIB else None,
                imu_state_dim=IMU_RAW_STATE_DIM,
                gt_state_dim=GT_FULL_STATE_DIM
            )
            datasets.append(ds)
        except Exception as e:
            print(f"  Error loading {d_path}: {e}")

    if not datasets:
        print("FATAL: No datasets loaded. Check your file paths.")
        return

    combined_dataset = ConcatDataset(datasets)
    loader = DataLoader(combined_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
    print(f"Total training sequences: {len(combined_dataset)}")

    input_channels = VOXEL_PARAMS['B'] * 2 if USE_STEREO else VOXEL_PARAMS['B']
    model = DEIONet(
        input_channels=input_channels,
        imu_state_dim=IMU_RAW_STATE_DIM,
        use_imu=USE_IMU_DATA
    ).to(DEVICE)
    dba_layer = DifferentiableBundleAdjustment(state_dim=GT_FULL_STATE_DIM, seq_len=SEQ_LEN,
                                               dba_param_dim=model.dba_param_dim).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = DEIOCost(imu_residual_weight=GT_WEIGHT, event_residual_weight=EVENT_WEIGHT,
                         prior_residual_weight=PRIOR_WEIGHT, huber_delta=HUBER_DELTA)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"Checkpoints will be saved to: {CHECKPOINT_DIR}")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for voxels, imu_raw_state, gt_full_state in loop:

            voxels, imu_raw_state, gt_full_state = (
                voxels.to(DEVICE), imu_raw_state.to(DEVICE), gt_full_state.to(DEVICE)
            )

            optimizer.zero_grad()

            dba_params = model(voxels, imu_raw_state)
            optimized_state = dba_layer(dba_params, imu_raw_state, gt_full_state)

            loss, imu_cost, event_cost, prior_cost = criterion(
                optimized_state, imu_raw_state, dba_params, gt_full_state, use_imu=USE_IMU_DATA
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_VALUE)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(
                loss=total_loss / (loop.n + 1),
                imu=imu_cost.item(),
                event=event_cost.item()
            )

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.8f}")

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f'deio_model_ep{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"SAVED checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    train()