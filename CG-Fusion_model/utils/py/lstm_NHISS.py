import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
# from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score

class RNN(nn.Module):
    def __init__(self, class_n, Input_Size):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(
            input_size=Input_Size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(128, class_n)
    def forward(self, x):
        r_out, (h_c, h_h) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()