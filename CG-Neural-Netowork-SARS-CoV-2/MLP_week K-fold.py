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

# 讀取來源檔案 遮蔽時會讀取以儲存npy檔案
new_data=False
filter_ths = 10
week_=''
if new_data==True:
    raw_data = pd.read_csv('./4602_SARS-CoV-2_pima_0708.csv')
elif week_=='week_':
    # npy path
    train_npy = [f'./dataset2/numpy_train_test_split_added time/X_train_{filter_ths}up_weeks.npy', f'./dataset2/numpy_train_test_split_added time/y_train_{filter_ths}up_weeks.npy']
    test_npy = [f'./dataset2/numpy_train_test_split_added time/X_test_{filter_ths}up_weeks.npy', f'./dataset2/numpy_train_test_split_added time/y_test_{filter_ths}up_weeks.npy']
elif week_=='':
    # npy path
    train_npy = [f'./dataset/X_train_below{filter_ths}.npy', f'./dataset/y_train_below{filter_ths}.npy']
    test_npy = [f'./dataset/X_test_below{filter_ths}.npy', f'./dataset/y_test_below{filter_ths}.npy']


class data_loader:
    def __init__(self, x_train, y_train, x_test, y_test):

        self.x_train = np.array(x_train)
        self.x_test = np.array(x_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        print(self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

    def get_index(self, data_list, GT_list, b):
        labels = []
        datas =[]
        for patch, i in enumerate(GT_list):
            for index, label in enumerate(b):
                if i==label:
                    labels.append(index)
                    datas.append(data_list[patch])
        return np.array(datas), np.array(labels)

    def npy_loading(self):
        # train processing
        b1, _, _, w1= np.unique(self.y_train,return_counts=True,return_index=True,return_inverse=True)
        # test(valid) processing
        b2, _, _, w2= np.unique(self.y_test,return_counts=True,return_index=True,return_inverse=True)
        X_train, Y_train = self.get_index(self.x_train, self.y_train, b2)
        X_test, Y_test = self.get_index(self.x_test, self.y_test, b2)
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((Y_train, Y_test))
        return X, y, b2

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
        x = F.softmax(x, dim=1)
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
def Tensor_generate(X_train, y_train, X_test, y_test):# pytorch data zip processing
    train_zip = TensorDataset(torch.tensor(X_train), torch.tensor(y_train)) #zip X, y
    test_zip = TensorDataset(torch.tensor(X_test), torch.tensor(y_test)) #zip X, y
    train_loader = DataLoader(dataset=train_zip, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=test_zip, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
    
def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

def predict(model, _loader, device):
    y_pred = []
    y_true = []
    with torch.no_grad():
        model.eval()
        for step, (b_x, b_y) in enumerate(_loader):
            if torch.cuda.is_available():
                b_x_, b_y_= (torch.unsqueeze(b_x, 1)).data.type(torch.FloatTensor).to(device), b_y.to(device)
            r_out = model(b_x_)
            _, preds = torch.max(r_out, 1) 
            y_pred.extend(preds.view(-1).detach().cpu().numpy())    
            y_true.extend(b_y_.cpu().view(-1).detach().cpu().numpy())
            del b_x_, b_y_
            torch.cuda.empty_cache()
        acc = accuracy_score(y_true, y_pred)
        print(acc)
    return acc


# K-fold
skf = StratifiedKFold(shuffle=True, n_splits=5, random_state=123) #random seed = 123
loader = data_loader(np.load(train_npy[0], allow_pickle=True), np.load(train_npy[1], allow_pickle=True), 
                    np.load(test_npy[0], allow_pickle=True), np.load(test_npy[1], allow_pickle=True))

# dataset npy loading
X, y, test_class = loader.npy_loading()
batch_size = 50
class_n = len(test_class)
# cuda setting
device = ('cuda' if torch.cuda.is_available() else 'cpu')
if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
# parameter setting
Input_Size = X.shape[1]
LR = 0.0001
t_acc_fold_stack, v_acc_fold_stack = [], []
print(X.shape, y.shape)
fold_n = 1

for train_index, test_index in skf.split(X, y):
    # model setting
    min_valid_loss = np.inf
    rnn = MLP(max(y)+1, Input_Size).to(device)
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    loss_fun = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
    print('Starting Training and Valid K-fold: ', fold_n)
    tloss_stack, vloss_stack = [], []
    X_train, y_train, X_test, y_test = X[train_index], y[train_index], X[test_index], y[test_index]
    train_loader, valid_loader = Tensor_generate(X_train, y_train, X_test, y_test)

    for epoch in tqdm(range(400)):
        for step, (b_x, b_y) in enumerate(train_loader):
            # Forward pass
            rnn.train()
            train_loss = 0.0
            if torch.cuda.is_available():
                b_x_, b_y_= (torch.unsqueeze(b_x, 1)).data.type(torch.FloatTensor).to(device), b_y.to(device)
            
            optimizer.zero_grad()
            r_out = rnn(b_x_)
            loss = loss_fun(r_out, b_y_)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()*b_x_.size(0)
            del b_x_, b_y_
            torch.cuda.empty_cache()
        tloss_stack.append(loss)

        # Validation
        with torch.no_grad():
            valid_loss = 0.0
            rnn.eval()
            for step, (b_x, b_y) in enumerate(valid_loader):
                if torch.cuda.is_available():
                    b_x_, b_y_= (torch.unsqueeze(b_x, 1)).data.type(torch.FloatTensor).to(device), b_y.to(device)
                r_out = rnn(b_x_)
                loss = loss_fun(r_out, b_y_)
                valid_loss = loss.item()*b_x_.size(0)
                del b_x_, b_y_
                torch.cuda.empty_cache()
            vloss_stack.append(loss) 

            if min_valid_loss > valid_loss:
                # print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
                min_valid_loss = valid_loss
                
                # Saving State Dict
                torch.save({'state_dict': rnn.state_dict()}, f'./pth/MLP_{week_}saved_model_{filter_ths}_{fold_n}.pth.tar')
    plt.figure(figsize=(10,10))
    plt.plot(tloss_stack, label='train loss')
    plt.plot(vloss_stack, label='valid loss')
    plt.xlabel('Epochs', fontsize=18)
    plt.ylabel('CE loss',fontsize=18)
    plt.title(f'MLP {week_[::-1]} model Loss plot [nums > {filter_ths}]', fontsize=25)
    plt.legend()
    plt.savefig(f'./results/{filter_ths}_loss_MLP_{week_}{fold_n}.jpg')
    plt.close(2)
    
    print('Train Accuracy K-fold', fold_n)
    t_acc = predict(rnn, train_loader, device)
    t_acc_fold_stack.append(t_acc)
    print('Valid(Test) Accuracy K-fold', fold_n)
    v_acc = predict(rnn, valid_loader, device)
    v_acc_fold_stack.append(v_acc)
    fold_n+=1
    del train_loader, valid_loader

# %%
print(t_acc_fold_stack)
print(v_acc_fold_stack)


# %%
# import matplotlib.pyplot as plt
# cf_matrix = confusion_matrix(y_true, y_pred, normalize='pred') 
# per_cls_acc = cf_matrix.diagonal()/cf_matrix.sum(axis=0)
# print(per_cls_acc)

# df_cm = pd.DataFrame(cf_m atrix, test_class, test_class)
# plt.figure(figsize = (10,10))
# sns.heatmap(df_cm, annot=True, cmap='gist_heat_r')
# plt.xlabel("prediction", fontsize =15)
# plt.ylabel("label (ground truth)", fontsize =15)
# plt.title(f'LSTM week model Prediction [num > {filter_ths}]', fontsize=22)
# plt.savefig(f'./results/{filter_ths}_CM_LSTM_week.jpg')


