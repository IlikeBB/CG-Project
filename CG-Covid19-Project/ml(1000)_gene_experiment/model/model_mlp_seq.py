import torch
import torch.nn.functional as F
from torch import nn
loss_func = nn.CrossEntropyLoss(reduction="sum")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MLP,self).__init__()
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc_layer(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()