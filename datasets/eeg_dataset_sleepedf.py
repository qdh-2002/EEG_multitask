import os
import numpy as np
import torch
from torch.utils.data import Dataset
from mne.io import read_raw_edf

class SleepEDFDataset(Dataset):
    def __init__(self, folder, segment_len=3000):
        self.samples = []
        for file in os.listdir(folder):
            if file.endswith(".edf"):
                raw = read_raw_edf(os.path.join(folder, file), preload=True)
                data = raw.get_data()
                label = 0  # 默认简单模拟，后续建议接入真正的sleep stage标签（W/N1/N2/N3/REM）
                for i in range(0, data.shape[1] - segment_len, segment_len):
                    segment = data[:, i:i + segment_len]
                    self.samples.append((segment.T, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
