import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

class DEAPDataset(Dataset):
    def __init__(self, mat_file, segment_len=256, transform=None):
        data = loadmat(mat_file)
        eeg_data = data['data']  # [trials, channels, time]
        labels = data['labels']  # [trials, 4] -> valence, arousal, dominance, liking

        self.samples = []
        for i in range(eeg_data.shape[0]):
            label = 1 if labels[i, 0] > 5 else 0  # 二分类情绪：高 vs 低愉悦度
            signal = eeg_data[i, :, :]
            for j in range(0, signal.shape[1] - segment_len, segment_len):
                self.samples.append((signal[:, j:j + segment_len].T, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
