import socket, pickle, time, numpy as np, yaml, random,os
import pandas as pd, math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from freeze import *
from _thread import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from utils.metric_reply import logs_realtime_reply
from utils.models import MLP
global clients, train_models, reply_stack, send_stack
global initial_model_status
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
initial_model_status = 'echo'
cudnn.benchmark = True
random.seed(1234)
torch.manual_seed(1234)

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config
config = load_config("./connect.yml")

client = socket.socket()
client.connect((config['host'], config['port']))
device = ('cuda' if torch.cuda.is_available() else 'cpu')

def trainer(model, trainloader):
    model.to(device)
    LR = 0.0001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in tqdm(range(5)):
        for step, (text, label) in enumerate(trainloader, start=1):
            images = text.to(device)
            target = label.to(device)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def recv():
    received_data = b""
    while True:
        try:
            # client.settimeout(15)
            received_data += client.recv(614400000)
            reply_len = len(received_data)
            if 'exit' in str(received_data):
                return None, 0, reply_len
            try:
                pickle.loads(received_data)
                break
            except BaseException:
                pass
        except socket.timeout:
            print("{recv_timeout} Seconds of Inactivity. socket.timeout Exception Occurred".format(recv_timeout=10))
            return None, 0, reply_len
    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Data: {msg}.\n".format(msg=e))
        return None, 0, reply_len
    return received_data, 1, reply_len
# dataloader
data_df = pd.read_csv('./dataset/12-17-rhees death_within28days+feature.csv')
def dataloader(table):
    for i in table:
        if (i in ['ID','LOC','outcome'])==False:
            # print(i)
            cols_filter = [x for x in table[i] if math.isnan(float(x))==False ]
            med = np.median(cols_filter)
            table[i] = [med if math.isnan(float(x))==True else x for x in table[i]]
            min_cols, max_cols =np.min(cols_filter), np.max(cols_filter)

            normal = lambda x: (x - min_cols)/(max_cols - min_cols)
            table[i] = [normal(x) for x in table[i]]
            table[i] = [0 if math.isnan(float(x))==True else x for x in table[i]]
    return table
data_df = dataloader(data_df)
loc = 1
data_df_LOC = data_df[data_df["LOC"]==loc]
print("Total Patient Sample", len(data_df))
print("Loc Patient Sample", len(data_df_LOC))
loc_weight = len(data_df_LOC)/len(data_df)
print("Loc Weight", loc_weight)
X_train, X_test, y_train, y_test = train_test_split(data_df_LOC.drop(['outcome'],axis=1), data_df_LOC['outcome'], 
                                                                                                    test_size=0.25, stratify=list(data_df_LOC['outcome']), random_state=123) #seed = 42, 123

print("LOC:", loc)
print('train', ' 0: ', len(y_train)-sum(y_train),'1:',sum(y_train))
print('valid', '0: ', len(y_test)-sum(y_test), '1:',sum(y_test))

try:
    X_train_ = np.array(X_train.drop(['ID','LOC'],axis=1))
    X_test_ = np.array(X_test.drop(['ID','LOC'],axis=1))
    y_train_ = np.array(y_train)
    y_test_ = np.array(y_test)
except:
    X_train_ = np.array(X_train.drop(['ID'],axis=1))
    X_test_ = np.array(X_test.drop(['ID'],axis=1))
    y_train_ = np.array(y_train)
    y_test_ = np.array(y_test)
print(X_train_.shape, X_test_.shape, y_train_.shape, y_test_.shape)


training_set = TensorDataset(torch.FloatTensor(X_train_), torch.FloatTensor(y_train_))
validation_set = TensorDataset(torch.FloatTensor(X_test_), torch.FloatTensor(y_test_))
trainloader = DataLoader(training_set, batch_size=64, shuffle=True)
testloader = DataLoader(validation_set, batch_size=len(validation_set), shuffle=False)

train_models = MLP(num_classes=1, input_size=25).to(device)

data = 'Client 1: matrix data data'
matrix_data = {'weight': None}
data_dict = {'status':'echo', 'client 1':matrix_data}
print('Send Initial echo to Server.')


best_auc =0.00
while True:
    response = pickle.dumps(data_dict)
    client.sendall(response)
    received_data, status, reply_len = recv()
    if status == 0:
        print("Nothing Received from the Server")
        client.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1) 
        break
    else:
        print("New Message from the Server with subject {sub}".format(sub=received_data["status"]))
        print(f"Received {reply_len} bytes server data")
        localtime = time.localtime()
        print( time.strftime("%Y-%m-%d %I:%M:%S %p", localtime))

    received_data['status']='model'
    model_dict = received_data['client 1']['weight']
    # print(model_dict)
    train_models.load_state_dict(model_dict)
    if config['freeze_state']==True:
        freeze_by_names(train_models, ('linear1','bn1'))
    train_models = trainer(train_models, trainloader)
    with torch.no_grad():
                valid_loss = 0.0
                train_models.eval()
                train_models.to('cpu')
                for i, (text, label) in enumerate(trainloader, start=1):
                    output = train_models(text.to('cpu'))
                    label = label
                acc = metrics.accuracy_score(output>0.5, label)
                fpr, tpr, thresholds = metrics.roc_curve(label, output, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                print(f"///-------Client1 model-------///")
                # print("Accuracy:", round(acc,5), "\nAUC:", round(auc,5))
                print('Client Accuracy: ', acc, 'Client AUC: ', auc)
    if auc > best_auc:
        best_auc = auc
        ck_name = "{0}-{1}.pt".format(str(config['client_ck_name']), loc)
        torch.save({'model_state_dict': train_models.state_dict()}, ck_name)
    # change new client weight to reply dict
    weight_dict = train_models.state_dict()
    # print("----------------------------weight dict--------------------------------------")
    # print(weight_dict)
    for i in weight_dict:
        weight_dict[i] = weight_dict[i]*loc_weight
    # print(weight_dict)
    received_data['client 1']['weight']= weight_dict
    data_dict = received_data

    
    # print(weight_dict)
