
import torch, yaml
import torch.nn as nn
import torch.nn.functional as F

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

config = load_config("./connect.yml")
class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MLP,self).__init__()
        self.linear1 = nn.Linear(in_features=input_size, out_features=10)
        self.bn1 = nn.BatchNorm1d(10)
        self.dt1 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(in_features=10, out_features=5)
        self.bn2 = nn.BatchNorm1d(5)
        self.dt2 = nn.Dropout(0.25)
        self.linear3 = nn.Linear(in_features=5, out_features=num_classes)
        
    def forward(self, x):
        x = self.bn1(self.linear1(x))
        x = F.relu(x)
        x = self.bn2(self.linear2(x))
        x = F.relu(x)
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()