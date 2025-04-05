import torch
from torch.utils.data import DataLoader
from models.multitask_transformer import MultiTaskEEGNet
from datasets.eeg_dataset_chbmit import CHBMITDataset
from datasets.eeg_dataset_deap import DEAPDataset
from datasets.eeg_dataset_sleepedf import SleepEDFDataset
from utils.metrics import multitask_loss

# 加载数据
train_loader = DataLoader(...)
val_loader = DataLoader(...)

model = MultiTaskEEGNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    model.train()
    for batch in train_loader:
        x, y_epi, y_emot, y_sleep = batch
        out_epi, out_emot, out_sleep = model(x)

        loss = multitask_loss(out_epi, y_epi, out_emot, y_emot, out_sleep, y_sleep)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

