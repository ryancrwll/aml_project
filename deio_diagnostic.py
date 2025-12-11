import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys

# Ensure local modules are accessible
# Assuming the new files are in the same directory
try:
    from vo_deio_net import DEIONet, DifferentiableBundleAdjustment
    from deio_loss import DEIOCost
    from deio_data import DEIODataset
except ImportError as e:
    print(f"Error importing new DEIO files: {e}")
    print("Please ensure vo_deio_net.py, deio_loss.py, and deio_data.py are in your path.")
    sys.exit(1)


# --- Configuration (Must match deio_train.py) ---
DATA_FILE = './data/indoor_flying4_data.hdf5' # Use a small dataset for fast testing
GT_FILE = './data/indoor_flying4_gt.hdf5'
CHECKPOINT_PATH = './checkpoints/vo_model_ep20.pth' # Dummy checkpoint path
SEQ_LEN = 10
BATCH_SIZE = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

VOXEL_PARAMS = {'H': 260, 'W': 346, 'B': 5}
USE_STEREO = False # IMPORTANT: Match your intended training configuration
USE_CALIB = True
CALIB_PATH = './data/indoor_flying_calib/camchain-imucam-indoor_flying.yaml'

# DEIO Specific Parameters
IMU_RAW_STATE_DIM = 6   # Accel (3) + Gyro (3)
GT_FULL_STATE_DIM = 15  # P(3), Q(4), V(3), Ba(3), Bg(3)

# DEIO Hyperparameters (used by the Cost function)
EVENT_WEIGHT = 100.0
IMU_WEIGHT = 1.0
PRIOR_WEIGHT = 0.1
HUBER_DELTA = 0.1

def run_diagnostic():
    print(f"--- Starting DEIO Diagnostic Check on {DEVICE} ---")
    print(f"Configuration: Stereo={USE_STEREO}, Bins={VOXEL_PARAMS['B']}, SeqLen={SEQ_LEN}, Batch={BATCH_SIZE}")

    # 1. Setup Data Loader
    if not os.path.exists(DATA_FILE) or not os.path.exists(GT_FILE):
        print(f"FATAL: Data files not found. Please check paths: {DATA_FILE}, {GT_FILE}")
        return

    try:
        test_dataset = DEIODataset(
            data_path=DATA_FILE,
            gt_path=GT_FILE,
            seq_len=SEQ_LEN,
            crop_params=VOXEL_PARAMS,
            use_stereo=USE_STEREO,
            use_calib=USE_CALIB,
            calib_path=CALIB_PATH if USE_CALIB else None,
            imu_state_dim=IMU_RAW_STATE_DIM,
            gt_state_dim=GT_FULL_STATE_DIM
        )
        loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        print(f"SUCCESS: Data loader initialized. Total sequences: {len(test_dataset)}")
    except Exception as e:
        print(f"FATAL: Failed to initialize DEIODataset. Error: {e}")
        return

    # 2. Setup Model and Loss
    input_channels = VOXEL_PARAMS['B'] * 2 if USE_STEREO else VOXEL_PARAMS['B']

    # Initialize components
    model = DEIONet(input_channels=input_channels, imu_state_dim=IMU_RAW_STATE_DIM).to(DEVICE)
    dba_layer = DifferentiableBundleAdjustment(state_dim=GT_FULL_STATE_DIM, seq_len=SEQ_LEN,
                                               dba_param_dim=model.dba_param_dim).to(DEVICE)
    criterion = DEIOCost(imu_residual_weight=IMU_WEIGHT, event_residual_weight=EVENT_WEIGHT,
                         prior_residual_weight=PRIOR_WEIGHT, huber_delta=HUBER_DELTA)

    print(f"SUCCESS: Models initialized. Input Channels: {input_channels}")

    # 3. Single Forward Pass and Shape Check
    model.eval()
    with torch.no_grad():
        # Get one batch
        try:
            voxels, imu_raw_state, gt_full_state = next(iter(loader))
        except StopIteration:
            print("FATAL: DataLoader is empty. Check dataset paths/valid indices.")
            return

        voxels, imu_raw_state, gt_full_state = (
            voxels.to(DEVICE), imu_raw_state.to(DEVICE), gt_full_state.to(DEVICE)
        )

        print("\n--- SHAPE VERIFICATION ---")

        # Check Input Shapes
        print(f"Input Voxels Shape:     {voxels.shape}")
        expected_voxels = (BATCH_SIZE, SEQ_LEN, input_channels, VOXEL_PARAMS['H'], VOXEL_PARAMS['W'])
        assert voxels.shape == expected_voxels, f"Mismatch: {voxels.shape} != {expected_voxels}"

        print(f"Input IMU State Shape:  {imu_raw_state.shape}")
        expected_imu = (BATCH_SIZE, SEQ_LEN, IMU_RAW_STATE_DIM)
        assert imu_raw_state.shape == expected_imu, f"Mismatch: {imu_raw_state.shape} != {expected_imu}"

        print(f"GT State Shape:         {gt_full_state.shape}")
        expected_gt = (BATCH_SIZE, SEQ_LEN, GT_FULL_STATE_DIM)
        assert gt_full_state.shape == expected_gt, f"Mismatch: {gt_full_state.shape} != {expected_gt}"

        # 4. Model Forward Pass
        dba_params = model(voxels, imu_raw_state)
        print(f"DBA Params Output Shape: {dba_params.shape}")
        expected_dba = (BATCH_SIZE, SEQ_LEN, model.dba_param_dim)
        assert dba_params.shape == expected_dba, f"Mismatch: {dba_params.shape} != {expected_dba}"

        # 5. DBA Layer (Placeholder) Pass
        optimized_state = dba_layer(dba_params, imu_raw_state, gt_full_state)
        print(f"Optimized State Shape: {optimized_state.shape}")
        expected_opt_state = (BATCH_SIZE, SEQ_LEN, GT_FULL_STATE_DIM)
        assert optimized_state.shape == expected_opt_state, f"Mismatch: {optimized_state.shape} != {expected_opt_state}"

        # 6. Loss Calculation
        loss, imu_cost, event_cost, prior_cost = criterion(
            optimized_state, imu_raw_state, dba_params, gt_full_state
        )

        print("\n--- COST VERIFICATION (Check Loss Components) ---")
        print(f"Total Initial Cost:   {loss.item():.6f}")
        print(f"IMU Residual Cost:    {imu_cost.item():.6f}")
        print(f"Event Residual Cost:  {event_cost.item():.6f} (Heavily weighted)")
        print(f"Prior Residual Cost:  {prior_cost.item():.6f}")

        # Check Total Loss vs Components
        calculated_total = (IMU_WEIGHT * imu_cost.item() +
                            EVENT_WEIGHT * event_cost.item() +
                            PRIOR_WEIGHT * prior_cost.item())
        print(f"Calculated Total Loss: {calculated_total:.6f}")
        assert np.isclose(loss.item(), calculated_total, atol=1e-5), "FATAL: Total Loss calculation mismatch!"

        # Final Success Message
        print("\n--- DIAGNOSTIC SUCCESSFUL ---")
        print("All tensor shapes are correct, and initial costs are calculated.")
        print("You are now ready to start the DEIO training run.")

if __name__ == "__main__":
    run_diagnostic()