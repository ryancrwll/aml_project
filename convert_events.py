import numpy as np
import torch

class VoxelGrid(object):
    """
    Converts a stream of events (N, 4) into a Voxel Grid tensor (B, H, W).
    Handles unstructured float arrays where columns are [x, y, t, p].
    """
    def __init__(self, H, W, B=15, normalize=True):
        self.H = H
        self.W = W
        self.B = B
        self.normalize = normalize

    def __call__(self, events, t_start=None, t_end=None):
        """
        :param events: NumPy array of shape (N, 4).
        :param t_start: Optional frame start time. If provided, used for temporal normalization instead of event times.
        :param t_end: Optional frame end time. If provided, used for temporal normalization instead of event times.
        """
        if events.shape[0] == 0:
            return torch.zeros(self.B, self.H, self.W, dtype=torch.float32)

        # 1. Extract columns by index (Handling unstructured data)
        # Column 0: x, Column 1: y, Column 2: timestamp, Column 3: polarity
        x = events[:, 0].astype(np.int64)
        y = events[:, 1].astype(np.int64)
        t = events[:, 2].astype(np.float32)
        p = events[:, 3].astype(np.float32)

        # 2. Normalize timestamps to [0, B-1]
        if t_start is None or t_end is None:
            # Fallback: use event timestamps (legacy behavior)
            t_start = t[0]
            t_end = t[-1]

        t_total = t_end - t_start

        if t_total == 0:
            return torch.zeros(self.B, self.H, self.W, dtype=torch.float32)

        # Normalize to range [0, B]
        t_norm = (t - t_start) / t_total * (self.B - 1)

        # 3. Create Voxel Grid
        # Floor and Ceil indices for interpolation
        b_low = np.floor(t_norm).astype(np.int64)
        b_high = b_low + 1

        # Interpolation weights
        w_low = b_high - t_norm
        w_high = t_norm - b_low

        # Clip to valid range
        b_low = np.clip(b_low, 0, self.B - 1)
        b_high = np.clip(b_high, 0, self.B - 1)

        # Accumulate
        V = np.zeros((self.B, self.H, self.W), dtype=np.float32)
        np.add.at(V, (b_low, y, x), p * w_low)
        np.add.at(V, (b_high, y, x), p * w_high)

        V_tensor = torch.from_numpy(V)

        # 4. Normalize (Mean/Std)
        if self.normalize:
            mean = V_tensor.mean()
            std = V_tensor.std()
            if std > 0:
                V_tensor = (V_tensor - mean) / std
            else:
                V_tensor[:] = 0

        return V_tensor