import socket, pickle, time, numpy as np, yaml, random,os
import pandas as pd, math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
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
device = ('cuda' if torch.cuda.is_available() else 'cpu')
clients, reply_stack, send_stack, received_data_stacks={}, {} ,{}, {}
server = socket.socket()
server.bind((config['host'], config['port']))
server.listen(5)

def trainer(model, trainloader):
    model.to(device)
    LR = 0.0001
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for epoch in tqdm(range(30)):
        for step, (text, label) in enumerate(trainloader, start=1):
            images = text.to(device)
            target = label.to(device)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def recv(client_, c_ip=None):
    received_data = b""
    while True:
        try:
            client_.settimeout(20)
            received_data += client_.recv(614400000)
            try:
                pickle.loads(received_data)
                break
            except BaseException:
                pass
        except socket.timeout:
            print()
            return None, 0
    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print("Error Decoding the Data: {msg}.\n".format(msg=e))
        return None, 0
    # print(received_data)
    return received_data, 1

def average_weight():
    global received_data_stacks
    # print(received_data_stacks)
    print("average len", len(received_data_stacks))
    server_weight = train_models.state_dict()
    # print(train_models.state_dict())
    # print('------------------------------original dict---------------------------------')
    try:
        for i in received_data_stacks:
            temp_cleint = received_data_stacks[i][list(received_data_stacks[i].keys())[1]]['weight']
            for idx in (server_weight):
                server_weight[idx] = server_weight[idx] + temp_cleint[idx]
        for idx in (server_weight):
            server_weight[idx] = server_weight[idx] / (len(received_data_stacks)+1) #server: +1
    except Exception as e:
       print(e)
# #     # print(received_data_stacks)
    print("------model_averaging pass--------------------------")
    return server_weight

def sample_weight():
    global received_data_stacks
    # print(received_data_stacks)
    print("average len", len(received_data_stacks))
    server_weight = train_models.state_dict()
    # print(train_models.state_dict())
    # print('------------------------------original dict---------------------------------')
    try:
        # for i in received_data_stacks:
        #     temp_cleint = received_data_stacks[i][list(received_data_stacks[i].keys())[1]]['weight']
        #     for idx in (server_weight):
        #         server_weight[idx] = server_weight[idx] + temp_cleint[idx]
        # for idx in (server_weight):
        #     server_weight[idx] = server_weight[idx] / 2 #server: +1
        for i in received_data_stacks:
            if i==0:
                server_weight = received_data_stacks[i][list(received_data_stacks[i].keys())[1]]['weight']
            else:
                for idx in (server_weight):
                    temp_cleint = received_data_stacks[i][list(received_data_stacks[i].keys())[1]]['weight']
                    server_weight[idx] =  temp_cleint[idx]+server_weight[idx]
            print("=====================server weight=======================")
            print(server_weight)
    except Exception as e:
       print(e)
# #     # print(received_data_stacks)
    print("------model_sample averaging pass--------------------------")
    return server_weight

def thread_send(socket_config, reply_data):
    global  received_data_stacks
    
    # print(reply_data)
    response = pickle.dumps(reply_data)
    socket_config.sendall(response)
    received_data_stacks = {}
    # received_data, status = recv(socket_config)
    pass

def thread_reply(socket_config, c_ip, index, connect_status='open'):
    global initial_model_status, received_data_stacks
    # while True:
    if connect_status == 'close':
        return  socket_config.send(str.encode('exit'))
    elif connect_status == 'open':
        received_data, stat = recv(socket_config, c_ip)
        if stat ==0:
            print("Error: Missing Data")
            pass
            # if missing data, reply close connections signal
        elif stat == 1:
            if received_data['status']=='echo':
                # First train processing...
                client_name = list(received_data.keys())[1]
                weight_dict = train_models.state_dict()
                received_data[client_name]['weight']= weight_dict
                # del thread send   
                received_data_stacks[index] =received_data

            elif received_data['status']=='model':
                localtime = time.localtime()
                print(f' reply data from {c_ip}.', time.strftime("%Y-%m-%d %I:%M:%S %p", localtime))
                received_data_stacks[index] =received_data
                # Next train processing...
                pass
        else:
                pass

def evaluate(train_models):
    with torch.no_grad():
                train_models.to('cpu')
                train_models.eval()
                for i, (text, label) in enumerate(testloader, start=1):
                    output = train_models(text.to('cpu'))
                    label = label
                acc = metrics.accuracy_score(output>0.5, label)
                fpr, tpr, thresholds = metrics.roc_curve(label, output, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                print("Server Accuracy:", round(acc,5), "\t Server AUC:", round(auc,5))
    return acc, auc

def run():
    # processing thread reply and thread send
    global received_data_stacks, initial_model_status, train_models, clients
    try:
        while True:
            received_data_stacks = {}
            print('run initial client nums',len(received_data_stacks))
            print(initial_model_status)
            for idx, (ip, socket_) in enumerate(clients.items()):
                start_new_thread(thread_reply, (socket_, ip, idx, 'open'))

            print('run reply client nums',len(received_data_stacks))
            while True:
                if len(received_data_stacks) ==len(clients):
                    break
            print('run reply client nums',len(received_data_stacks))
            
            while True:
                if len(received_data_stacks) ==len(clients):
                    break
            time.sleep(2)
            # print(received_data_stacks)
            if initial_model_status =='echo':
                print('echo run send client nums',len(received_data_stacks))
                for idx, (ip, socket_) in enumerate(clients.items()):
                    start_new_thread(thread_send, (socket_, received_data_stacks[idx]))
                time.sleep(2)

            elif initial_model_status=='model':
                print('model run send client nums',len(received_data_stacks))
                avg_weight = sample_weight() #return average weight
                # print(avg_weight)
                train_models.load_state_dict(avg_weight)
                train_models = trainer(train_models, trainloader)
                acc, auc = evaluate(train_models)
                matrix_data = {'weight': avg_weight}
                
                for idx, (ip, socket_) in enumerate(clients.items()):
                    data_dict = {'status':'model', ('client '+str(idx+1)):matrix_data}
                    # print(data_dict)
                    start_new_thread(thread_send, (socket_, data_dict))
                # print(avg_weight)
                time.sleep(2)
                # return 0
                if auc >0.98:
                    torch.save({'model_state_dict': train_models.state_dict()}, f"{config['best_ck_name']}.pt")
                # if True:
                #     print('Training Done........')
                #     return 0
            initial_model_status = 'model'
    except socket.timeout:
        return 0
        
# ------train----------
# dataloader
def dataloader(table):
    for i in table:
        if (i in ['ID','LOC','outcome'])==False:
            cols_filter = [x for x in table[i] if math.isnan(float(x))==False ]
            med = np.median(cols_filter)
            table[i] = [med if math.isnan(float(x))==True else x for x in table[i]]
            min_cols, max_cols =np.min(cols_filter), np.max(cols_filter)
            normal = lambda x: (x - min_cols)/(max_cols - min_cols)
            table[i] = [normal(x) for x in table[i]]
            table[i] = [0 if math.isnan(float(x))==True else x for x in table[i]]
    return table
data_df = pd.read_csv('./dataset/original_555 sepsis dataset.csv')
data_df = dataloader(data_df)
X_train, X_test, y_train, y_test = train_test_split(data_df.drop(['outcome'],axis=1), data_df['outcome'], 
                                                                                                    test_size=0.25, stratify=list(data_df['outcome']), random_state=123) #seed = 42, 123
X_train = dataloader(X_train)
X_test = dataloader(X_test)
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
print(y_test_)
print(X_train_.shape, X_test_.shape, y_train_.shape, y_test_.shape)


# ------dataloader------
# train_zip = TensorDataset(torch.tensor(train_img), train_lab.flatten())
training_set = TensorDataset(torch.FloatTensor(X_train_), torch.FloatTensor(y_train_))
validation_set = TensorDataset(torch.FloatTensor(X_test_), torch.FloatTensor(y_test_))
trainloader = DataLoader(training_set, batch_size=64, shuffle=True)
testloader = DataLoader(validation_set, batch_size=64, shuffle=False)

train_models = MLP(num_classes=1, input_size=25)
train_models.initialize_weights()
train_models.to(device)
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
            print("///-------original model-------///")
            print("Accuracy:", round(acc,5), "\nAUC:", round(auc,5))
            print('Initial Accuracy: ', acc, 'Initial AUC: ', auc)      
# print(train_models)
max_connect = 3
while True:
    try:
        print("Waiting for Connection....")
        server.settimeout(20)
        c, addr = server.accept()
        clients[addr] = c
        # maximian connections client number. ex:2.
        if len(clients)==max_connect:
            print([i for i in clients])
            run()
        # if connections client number > maximian connections client number. ex
        # del dict socket status and repley disconnected signal
    except socket.timeout:
        print('Server Timeout')
        for ip, socket_ in clients.items():
            
            thread_reply(socket_, ip, 0, 'close')
        print('Connecting End')
        # Avoid occur sys error Address already in use. delay 5 sec.
        time.sleep(2)
        break
        