import torch
import torch.nn as nn

class DEIONet(nn.Module):
    """
    CNN-GRU architecture for Deep Event Inertial Odometry (DEIO).

    Input shape: (Batch, Seq_Len, C, H, W) for Voxels
    Output shape: (Batch, Seq_Len, N_PARAMS) where N_PARAMS are the parameters
                  required by the DBA layer (e.g., visual feature weights, biases).
    """
    def __init__(self, input_channels=10, feature_dim=512, hidden_dim=256,
                 imu_state_dim=6, dba_param_dim=32, use_imu=True):
        super(DEIONet, self).__init__()
        self.use_imu = use_imu
        self.feature_dim = feature_dim
        self.imu_state_dim = imu_state_dim
        self.dba_param_dim = dba_param_dim

        # 1. CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, feature_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.dropout = nn.Dropout(p=0.5)

        # 2. IMU Encoder
        self.imu_encoder = nn.Sequential(
            nn.Linear(imu_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
        )

        # 3. RNN
        self.rnn = nn.GRU(
            input_size=feature_dim + 128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )

        # 4. DBA Head
        self.dba_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, dba_param_dim)
        )

    def forward(self, x, imu_raw_state):
        B, S, C, H, W = x.shape

        # --- MANDATORY FIX: IGNORE IMU INPUT IF TOGGLE IS OFF ---
        if not self.use_imu:
            imu_raw_state = torch.zeros_like(imu_raw_state)

        # 1. Spatial Features
        vis_feat = self.cnn(x.view(B * S, C, H, W)).squeeze(-1).squeeze(-1)
        vis_feat = self.dropout(vis_feat)

        # 2. IMU Features
        imu_feat = self.imu_encoder(imu_raw_state.view(B * S, -1))

        # 3. Combine and RNN
        rnn_input = torch.cat([vis_feat, imu_feat], dim=-1).view(B, S, -1)
        rnn_out, _ = self.rnn(rnn_input)

        # 4. DBA Parameters
        return self.dba_head(rnn_out)

# Keep the DifferentiableBundleAdjustment class as it was

class DifferentiableBundleAdjustment(nn.Module):
    def __init__(self, state_dim=15, seq_len=10, dba_param_dim=32):
        super(DifferentiableBundleAdjustment, self).__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.dba_param_dim = dba_param_dim

    def forward(self, dba_params, imu_measurements, gt_state=None, use_gt_init=True):
        B, S, _ = dba_params.shape
        device = dba_params.device
        optimized_state = torch.zeros(B, S, self.state_dim, device=device)

        if use_gt_init and (gt_state is not None):
            optimized_state[:, 0, :7] = gt_state[:, 0, :7]
        else:
            optimized_state[:, 0, 3:7] = torch.tensor([0., 0., 0., 1.], device=device)

        learned_state_delta = dba_params[..., :7] * 0.1
        for i in range(S - 1):
            optimized_state[:, i + 1, 0:3] = optimized_state[:, i, 0:3] + learned_state_delta[:, i, 0:3]
            new_q = optimized_state[:, i, 3:7] + learned_state_delta[:, i, 3:7]
            optimized_state[:, i + 1, 3:7] = new_q / new_q.norm(dim=-1, keepdim=True)

        return optimized_state