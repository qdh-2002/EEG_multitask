import torch
from torch.utils.data import DataLoader
from datasets.eeg_dataset_chbmit import CHBMITDataset
from datasets.eeg_dataset_deap import DEAPDataset
from datasets.eeg_dataset_sleepedf import SleepEDFDataset
from models.multitask_transformer import MultiTaskEEGNet
from sklearn.metrics import accuracy_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = MultiTaskEEGNet()
model.load_state_dict(torch.load("multitask_eeg_model.pt", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 加载数据
dataset_epi = CHBMITDataset("data/chbmit")
dataset_emot = DEAPDataset("data/deap/data_preprocessed.mat")
dataset_sleep = SleepEDFDataset("data/sleep_edf")

dataloader_epi = DataLoader(dataset_epi, batch_size=16)
dataloader_emot = DataLoader(dataset_emot, batch_size=16)
dataloader_sleep = DataLoader(dataset_sleep, batch_size=16)

# 验证函数
def evaluate(dataloader, task="epilepsy"):
    all_preds, all_labels = [], []
    for x, y in dataloader:
        x = x.to(DEVICE)
        with torch.no_grad():
            out_epi, out_emot, out_sleep = model(x)

        if task == "epilepsy":
            pred = torch.argmax(out_epi, dim=1)
        elif task == "emotion":
            pred = torch.argmax(out_emot, dim=1)
        else:
            pred = torch.argmax(out_sleep, dim=1)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"{task} Accuracy: {acc:.4f}")

evaluate(dataloader_epi, "epilepsy")
evaluate(dataloader_emot, "emotion")
evaluate(dataloader_sleep, "sleep")

