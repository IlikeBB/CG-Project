import torch
import torch.nn.functional as F
from torch import nn
loss_func = nn.CrossEntropyLoss(reduction="sum")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.dt1 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(in_features=1024, out_features=256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dt2 = nn.Dropout(0.25)
        self.linear3 = nn.Linear(in_features=256, out_features= num_classes)
        
    def forward(self, x):
        x = self.bn1(self.linear1(x))
        x = F.relu(x)
        # x = self.dt1(x)
        x = self.bn2(self.linear2(x))
        x = F.relu(x)
        # x = self.dt2(x)
        x = self.linear3(x)
        # x = F.softmax(x, dim=1)
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