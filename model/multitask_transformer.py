import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MultiTaskEEGNet(nn.Module):
    def __init__(self, input_dim=64, emb_dim=128, n_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, emb_dim)

        encoder_layer = TransformerEncoderLayer(d_model=emb_dim, nhead=n_heads)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier_epilepsy = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 2))
        self.classifier_emotion  = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 3))
        self.classifier_sleep    = nn.Sequential(nn.Linear(emb_dim, 64), nn.ReLU(), nn.Linear(64, 5))

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1) 

        out_epi = self.classifier_epilepsy(x)
        out_emot = self.classifier_emotion(x)
        out_sleep = self.classifier_sleep(x)

        return out_epi, out_emot, out_sleep
