import torch
from vo_deio_net import DEIONet
from deio_loss import DEIOCost

def run_diagnostic():
    print("--- Starting DEIO Multi-Mode Diagnostic ---")
    B, S, C, H, W = 1, 5, 5, 260, 346
    voxels = torch.randn(B, S, C, H, W)
    imu_1 = torch.randn(B, S, 6)
    imu_2 = torch.zeros(B, S, 6) # Drastically different IMU input
    gt = torch.randn(B, S, 16)

    # Test Case 1: Events-Only Sensitivity
    model_eo = DEIONet(input_channels=C, use_imu=False)

    # --- CRITICAL FIX: Set to eval() to disable Dropout randomness ---
    model_eo.eval()

    with torch.no_grad():
        out_1 = model_eo(voxels, imu_1)
        out_2 = model_eo(voxels, imu_2)

    diff = torch.abs(out_1 - out_2).max().item()
    print(f"\n[Test 1] Events-Only Sensitivity Check: {diff:.8f}")

    # Test Case 2: Loss Weighting
    criterion = DEIOCost()
    total_l, imu_l, _, _ = criterion(gt, imu_1, out_1, gt, use_imu=False)
    print(f"[Test 2] Events-Only IMU Weight Check: Total={total_l.item():.4f}, IMU_Part={imu_l.item():.4f}")

    if diff < 1e-7:
        print("\nSUCCESS: Model correctly ignores IMU in Events-Only mode.")
    else:
        print("\nFAILURE: Model is still sensitive to IMU data.")

if __name__ == "__main__":
    run_diagnostic()