import torch
from torch.utils.data import Dataset
import numpy as np

class SimulatedEEGDataset(Dataset):
    def __init__(self, num_samples=300, segment_len=256, channels=16, num_classes=2):
        self.samples = []
        for _ in range(num_samples):
            x = np.random.randn(segment_len, channels)  # [T, C]
            y = np.random.randint(0, num_classes)
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
