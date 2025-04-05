import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.eeg_dataset_chbmit import CHBMITDataset
from datasets.eeg_dataset_deap import DEAPDataset
from datasets.eeg_dataset_sleepedf import SleepEDFDataset
from models.multitask_transformer import MultiTaskEEGNet
from utils.metrics import multitask_loss

# 配置
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
SEG_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
dataset_epi = CHBMITDataset("data/chbmit", segment_len=SEG_LEN)
dataset_emot = DEAPDataset("data/deap/data_preprocessed.mat", segment_len=SEG_LEN)
dataset_sleep = SleepEDFDataset("data/sleep_edf", segment_len=SEG_LEN)

# 为简单起见，这里用相同的DataLoader（你可分别训练或同步长度采样）
train_loader = DataLoader(list(zip(dataset_epi, dataset_emot, dataset_sleep)), batch_size=BATCH_SIZE, shuffle=True)

# 模型 & 优化器
model = MultiTaskEEGNet(input_dim=SEG_LEN).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
writer = SummaryWriter("runs/eeg_multitask")

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        (x1, y1), (x2, y2), (x3, y3) = batch
        x = torch.cat([x1, x2, x3], dim=0).to(DEVICE)  # 拼接模拟多任务输入
        y_epi = y1.to(DEVICE)
        y_emot = y2.to(DEVICE)
        y_sleep = y3.to(DEVICE)

        out_epi, out_emot, out_sleep = model(x)

        loss = multitask_loss(out_epi[:len(y1)], y_epi,
                              out_emot[len(y1):len(y1)+len(y2)], y_emot,
                              out_sleep[-len(y3):], y_sleep)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "multitask_eeg_model.pt")
writer.close()
