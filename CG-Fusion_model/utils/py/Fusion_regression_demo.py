from scipy.sparse.construct import rand
import torch, random, os, multiprocessing
import numpy as np, pandas as pd, nibabel as nib 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchio as tio
# multiprocess cpu 
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from utils.S1_utils import clip_gradient
num_workers = multiprocessing.cpu_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
random.seed(1234)
torch.manual_seed(1234)
csv_path = './csv/NIHSS_Continuous_Variable_222patient.csv'
table_ =  pd.read_csv(csv_path, index_col=False)
# print(table_.columns)
table_temp = table_['out_sum']
table_label = table_.drop(['ID','rnn_sum(out-in)'] ,axis=1)
# print(table_.columns)                           
print("table_label.columns.values", len(table_label.columns.values))
# table_label.to_csv('./NIHSS_fiter_222patient.csv',index=False)
# print(table_label)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(table_label)

y_nor = scaler.transform(table_label)
# print(y_nor)
new_table_nor = pd.DataFrame(y_nor, columns=table_label.columns)
new_table_nor['out_sum'] = table_temp
print(new_table_nor)
# new_table_nor = table_label

X_table = new_table_nor.drop(['out_sum'] ,axis=1)
y_table = new_table_nor['out_sum']

# X_table = new_table_nor.drop(['out_sum'] ,axis=1)
# y_table = new_table_nor['out_sum']
from torch.utils.data import TensorDataset, DataLoader
X_train, X_test, y_train, y_test = train_test_split(np.array(X_table), np.array(y_table), test_size=0.25, random_state=123) #seed = 42, 123
print("train", y_train.shape, "test", y_test.shape)
training_set = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
validation_set = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = torch.nn.Linear(22, 1)

    def forward(self, x):
        x = self.layer(x)      
        return x
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
        
class log_reply:
    def __init__(self):
        self.loss_sum =[]
    def get_item(self, loss):
        self.loss_sum.append(loss.cpu().detach().numpy())
        return np.mean(self.loss_sum)

def train(train_loader, model, criterion, optimizer, epoch):
    metric_ = log_reply()
    model.train()
    stream = tqdm(train_loader)
    for i, (value, count) in enumerate(stream, start=1):
        output = model(value.to(device)).squeeze(-1)
        loss = criterion(output, count.to(device))
        reply = metric_.get_item(loss)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, params['clip'])
        optimizer.step()
        stream.set_description(f"Epoch: {epoch}. Train. loss: {reply}, LR: {optimizer.param_groups[0]['lr']}")
    writer.add_scalar(f'Loss/Train Loss', reply, epoch)
    try:
        if (epoch>params['scheduler_epoch']):
            scheduler.step()
    except:
        pass
def validate(valid_loader, model, criterion, epoch):
    global best_vloss, best_vacc
    metric_ = log_reply()
    model.eval()
    stream_v = tqdm(valid_loader)
    with torch.no_grad():
        for i, (value, count) in enumerate(stream_v, start=1):
            # print(nihss)
            output = model(value.to(device)).squeeze(-1)
            # print(output)
            loss = criterion(output, count.to(device))
            reply = metric_.get_item(loss)
            stream_v.set_description(f"Epoch: {epoch}. Valid. loss: {reply}")
        writer.add_scalar(f'Loss/Valida Loss', reply, epoch)
        if reply<best_vloss:
            best_vloss = reply
            best_ck_name = f'{ck_pth}/best - vloss - {project_name}.pt'
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  best_vloss,}, best_ck_name)
            print('save...', best_ck_name)
# X
def  train_valid_process_main(model, training_set, validation_set, batch_size):
   global best_vloss, best_vacc
   best_vloss = np.inf
   best_vacc = 0.00
   # Subject Dataloader Building
   train_loader = DataLoader(training_set, batch_size=8, shuffle=True, num_workers=10)
   valid_loader = DataLoader(validation_set, batch_size=1, shuffle=False, num_workers=4)

   for epoch in range(1, params['epochs'] + 1):
      train(train_loader, model, loss_func, optimizer, epoch)
      validate(valid_loader, model, loss_func, epoch)
   return model

# checkpoint setting
if True: #model record
    params = {
        "type": "out_sum",
        "model": 'linear regression', #baseline = 'resnet18'
        "model_depth": 1,
        "device": "cuda",
        "opt": "Adam",
        "lr": 0.003, #baseline = 0.003
        "scheduler_epoch": 0, #nl: 5, ap: None
        "batch_size": 8, #baseline resnet18 : 8
        "epochs": 300,
        "clip": 0.5,
        "Adjust01": "lr 0.001, clip->0.5, SGD, epoch-300",
        "fixing": "None"
        }
    project_name = f"{params['type']} - {params['model']} - {params['opt']} - lr_{params['lr']} - epoch_{params['epochs']}- CEL"
    # project_folder = f"TEST01.10-03-222_patient_sum_ONLY_NIHSS_score_22(Continuous Variable)_Norm"
    project_folder = f"TEST_normalized"
    ck_pth = f'./checkpoint/{project_folder}'
    if os.path.exists(ck_pth)==False:
        os.mkdir(ck_pth)
    ck_name = project_name
    # write training setting txt
    #txt record model config and adjust 
    path = f'./checkpoint/{project_folder}/{project_name}.txt'
    f = open(path, 'w')
    lines = params
    f.writelines([f'{i} : {params[i]} \n' for i in params])
    f.close()
    # tensorboard setting
    tensorboard_logdir = f'./logsdir/regression/ {project_folder} - {project_name}'
writer=SummaryWriter(tensorboard_logdir)


net = Net()
net.initialize_weights()
net.to(device)
if params['opt']=='Adam':
    optimizer = Adam(net.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
else:
    optimizer = torch.optim.SGD(net.parameters(), lr=params['lr'], weight_decay = 1e-4, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max = 100)
# optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss()
print(net)
logs  = train_valid_process_main(net, training_set, validation_set, 8)
writer.close()