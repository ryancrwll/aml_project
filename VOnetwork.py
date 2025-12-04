import torch
import torch.nn as nn

class VONet(nn.Module):
    """
    CNN-RNN architecture for Event-Based Visual Odometry (VO).

    The CNN block extracts compact features from each Voxel Grid (C x H x W).
    The RNN (GRU) block models the sequential motion dynamics across time T.

    Input shape: (Batch, Seq_Len, Channels, H, W)
    Output shape: (Batch, Seq_Len, 6)
    """
    def __init__(self, input_channels=15, feature_dim=512, hidden_dim=256):
        """
        :param input_channels: Number of Voxel Grid bins (C).
        :param feature_dim: Dimension of the CNN output feature vector (Input to RNN).
        :param hidden_dim: Dimension of the RNN hidden state.
        """
        super(VONet, self).__init__()
        self.feature_dim = feature_dim

        # 1. CNN Encoder (Spatial Features)
        # Architecture is deep to compress the high-resolution Voxel Grid (260x346)
        self.cnn = nn.Sequential(
            # Input: (B*S, C, 260, 346)
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

            # Final Average Pooling: Reduces spatial size to 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2), # Output spatial size approx 130x173
        #     nn.BatchNorm2d(32), nn.ReLU(inplace=True),

        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Output spatial size approx 65x87
        #     nn.BatchNorm2d(64), nn.ReLU(inplace=True),

        #     nn.Conv2d(64, feature_dim, kernel_size=3, stride=2, padding=1), # Output spatial size approx 33x44
        #     nn.BatchNorm2d(feature_dim), nn.ReLU(inplace=True),

        #     # Final Average Pooling: Reduces spatial size to 1x1
        #     nn.AdaptiveAvgPool2d((1, 1))
        # )

        # 2. RNN (Temporal Features)
        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=2, # Two layers for deeper temporal modeling
            batch_first=True
        )

        # 3. Regressor (6-DOF Pose)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 6) # Output: [dx, dy, dz, d_roll, d_pitch, d_yaw]
        )

    def forward(self, x):
        # x shape: (Batch, Seq, Channels, H, W)
        B, S, C, H, W = x.shape

        # 1. CNN: Flatten Batch and Seq for Convolution
        c_in = x.reshape(B * S, C, H, W)
        feat = self.cnn(c_in)

        # 2. RNN: Reshape for GRU: (Batch, Seq, Features)
        feat = feat.squeeze().reshape(B, S, self.feature_dim)

        out, _ = self.rnn(feat)

        # 3. Regressor: Flatten for FC layers
        out = out.reshape(B * S, -1)
        pose = self.fc(out)

        # Reshape back to Sequence
        return pose.view(B, S, 6)