import os
import numpy as np
import torch
from torch.utils.data import Dataset
from mne.io import read_raw_edf

class CHBMITDataset(Dataset):
    def __init__(self, data_dir, segment_len=256, transform=None):
        self.data_dir = data_dir
        self.segment_len = segment_len
        self.transform = transform
        self.samples = []

        for file in os.listdir(data_dir):
            if file.endswith(".edf"):
                raw = read_raw_edf(os.path.join(data_dir, file), preload=True)
                data = raw.get_data()  # [channels, time]
                label = 1 if "seizure" in file else 0  # 简化示例
                
                for i in range(0, data.shape[1] - segment_len, segment_len):
                    segment = data[:, i:i + segment_len]
                    self.samples.append((segment.T, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
