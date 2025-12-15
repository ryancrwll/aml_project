import torch
import torch.nn as nn

class DEIOCost(nn.Module):
    def __init__(self, imu_residual_weight=1.0, event_residual_weight=100.0,
                 prior_residual_weight=0.1, huber_delta=0.1):
        super(DEIOCost, self).__init__()
        self.robust_loss = nn.SmoothL1Loss(reduction='none', beta=huber_delta)

        # We rename this internally to "pose_weight" to be clear it's for GT supervision
        self.pose_weight = imu_residual_weight
        self.event_weight = event_residual_weight
        self.prior_weight = prior_residual_weight

    def forward(self, optimized_state, imu_measurements, visual_features, gt_state, use_imu=True):
        """
        optimized_state: The output from the network (Prediction)
        gt_state: The Ground Truth from your .hdf5 files (Target)
        """

        active_pose_weight = self.pose_weight

        # 1. Trajectory Loss (Compare Prediction vs GT)
        # Adding small noise to GT to simulate the uncertainty a real IMU filter would have,
        # but the target essentially remains the GT.
        target_state = gt_state + (torch.randn_like(gt_state) * 1e-4) + 1e-5

        pose_residual = optimized_state - target_state
        pose_cost = self.robust_loss(pose_residual, torch.zeros_like(pose_residual)).mean()

        # 2. Event Residual (Visual consistency proxy)
        visual_residual = visual_features.mean(dim=-1).unsqueeze(-1)
        event_cost = self.robust_loss(visual_residual, torch.zeros_like(visual_residual)).mean()

        # 3. Smoothness Prior (Prevent jitter)
        prior_cost = self.robust_loss(optimized_state[:, 1:, :] - optimized_state[:, :-1, :],
                                     torch.zeros_like(optimized_state[:, 1:, :])).mean()

        # Calculate Total Cost
        total_cost = (
            active_pose_weight * pose_cost +
            self.event_weight * event_cost +
            self.prior_weight * prior_cost
        )

        return total_cost, pose_cost, event_cost, prior_cost