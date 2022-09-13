from scipy.sparse.construct import rand
import torch, random, os, multiprocessing
import numpy as np, pandas as pd, nibabel as nib 
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchio as tio
# multiprocess cpu 
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from scipy import ndimage
from utils.loss import FocalLoss
from utils.S1_utils import clip_gradient
# from utils.model_res_fu import generate_model
from utils.model_res import generate_model
num_workers = multiprocessing.cpu_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
random.seed(1234)
torch.manual_seed(1234)

class MLP(nn.Module):
    def __init__(self, num_classes, input_size):
        super(MLP,self).__init__()
        
        self.linear1 = nn.Linear(in_features=input_size, out_features = 10)
        self.linear3 = nn.Linear(in_features=10, out_features=num_classes)
        
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear3(x)
        return x

# load csv label
if True:
    csv_path = './NIHSS_score223+NL+AP_LSTM.csv'
    table_ =  pd.read_csv(csv_path)
    # print(table_.columns)
    table_label = table_.drop(['ID', 'predict (0-2"good", 3-6"bad")', '1/0: 3T/1.5T MRI', 'A/P', 'N/L', 'age', 
                                                           '發病日期', '有acute MRA/日期 (2wk內)', '病房日期', 'onset-to-image(有acute-發病日期)', 'onset-to-ward(病房日期-發病日期)', 'entry_mRS', 'NIHSS  total', 'Out_mRS',
                                                            'Out_mRS_class', 'Out_NIHSS: 1a', 'Out_NIHSS: 1b', 'Out_NIHSS: 1c',
                                                            'Out_NIHSS: 2', 'Out_NiHSS: 3', 'Out_NIHSS: 4', 'Out_NIHSS: 5a',
                                                            'Out_NIHSS:5b', 'NIHSS: 6a.1', 'Out_NIHSS:6b', 'Out_NIHSS:7',
                                                            'Out_NIHSS:8', 'Out_NIHSS:9', 'Out_NIHSS:10', 'Out_NIHSS:11',
                                                            'rnn_NIHSS: 1a', 'rnn_NIHSS: 1b', 'rnn_NIHSS: 1c', 'rnn_NIHSS: 2',
                                                            'rnn_NiHSS: 3', 'rnn_NIHSS: 4', 'rnn_NIHSS: 5a', 'rnn_NIHSS: 5b',
                                                            'rnn_NIHSS: 6a', 'rnn_NIHSS:6b', 'rnn_NIHSS:7', 'rnn_NIHSS:8',
                                                            'rnn_NIHSS:9', 'rnn_NIHSS:10', 'rnn_NIHSS:11']
                                                           ,axis=1)
    # print(table_.columns)                           
    def norm_one_zero(table):
        import math
        # print(table)
        for i in table:
            if (i in ['age_norm','gender','onset-to-image(datys)','onset-to-ward(days)','nihss_sum_norm','Out_mRS', 'out_sum', 'rnn_sum(out-in)'])==False:
                # print(i)
                cols_filter = [x for x in table[i] if math.isnan(float(x))==False ]
                med = np.median(cols_filter)
                table[i] = [med if math.isnan(float(x))==True else x for x in table[i]]
                min_cols, max_cols =np.min(cols_filter), np.max(cols_filter)

                normal = lambda x: (x - min_cols)/(max_cols - min_cols)
                table[i] = [normal(x) for x in table[i]]
                table[i] = [0 if math.isnan(float(x))==True else x for x in table[i]]

            if i == 'rnn_sum(out-in)':
                table[i] = [1 if x>=1  else 0 for x in table[i]]
        return table

    print("table_label.columns.values", len(table_label.columns.values))
    # print(table_label)
    nii_3t_train = sorted([i for i in os.listdir(os.path.join('./dataset/S2_data1.5&3.0/'))])

    table_label_norm = norm_one_zero(table_label)
    print(table_label_norm.head())
    X_train, X_test, y_train, y_test = train_test_split(nii_3t_train, table_label_norm,  stratify=list(table_label_norm['rnn_sum(out-in)']), test_size=0.25, random_state=123) #seed = 42, 123
    # print(y_train)
    print('train', ' 0: ', len(y_train['rnn_sum(out-in)'])-sum(y_train['rnn_sum(out-in)']),'1:',sum(y_train['rnn_sum(out-in)']))
    print('valid', '0: ', len(y_test['rnn_sum(out-in)'])-sum(y_test['rnn_sum(out-in)']), '1:',sum(y_test['rnn_sum(out-in)']))
    
    # balance dataset
def tio_process(nii_3t_, table_3t_, basepath_='./dataset/S2_data1.5&3.0/'):
    subjects_ = []
    for  (nii_path, nii_table) in zip(nii_3t_ , table_3t_):
        tb_len = nii_table.shape[-1]
        subject = tio.Subject(
            dwi = tio.ScalarImage(os.path.join(basepath_, nii_path)), 
            runn_sum = int(nii_table[-1]),
            score= nii_table[0:22])
        subjects_.append(subject)
    return subjects_

class logs_realtime_reply:
    def __init__(self):
        self.avg_dice = 0.0
        self.avg_loss=np.inf
        self.avg_tn = 0
        self.avg_fp = 0
        self.avg_fn = 0
        # self.running_metic = {"Loss":0, "TP":0, "FP":0, "FN": 0, "Spec": 0, "Sens": 0}
        self.running_metic = {"Loss":0, "Accuracy":0, "Spec": 0, "Sens": 0}
        self.end_epoch_metric = None
    def metric_stack(self, inputs, targets, loss):
        self.running_metic['Loss'] +=loss
        # metric setting
        _, SR = torch.max(inputs, 1)
        GT = targets
        TP = int((SR * GT).sum()) #TP
        FN = int((GT * (1-SR)).sum()) #FN
        TN = int(((1-GT) * (1-SR)).sum()) #TN
        FP = int(((1-GT) * SR).sum()) #FP
        self.running_metic['Accuracy'] += round((TP + TN)/(TP + TN + FP + FN), 5)*100
        self.running_metic['Sens'] += round(float(TP)/(float(TP+FN) + 1e-6), 5)
        self.running_metic['Spec'] += round(float(TN)/(float(TN+FP) + 1e-6), 5)

    def mini_batch_reply(self, current_step, epoch, iter_len):
        # avg_reply_metric = {"Loss":None, "TP":None, "FP":None, "FN": None, "Spec": None, "Sens": None}
        avg_reply_metric = {"Loss":None, "Accuracy": None,"Spec": None, "Sens": None}
        for j in avg_reply_metric:
            avg_reply_metric[j] = round(self.running_metic[j]/int(current_step),5)
        
        if current_step ==iter_len:
            self.end_epoch_metric = avg_reply_metric
        return avg_reply_metric

    def epoch_reply(self):
        return self.end_epoch_metric

def model_create():
    model = MLP(num_classes=2, input_size=22)
    model.to(device)
    return model

# model train
def train(train_loader, model, criterion, optimizer, epoch):
    get_logs_reply = logs_realtime_reply()
    model.train()
    stream = tqdm(train_loader)
   
    for i, data in enumerate(stream, start=1):
    # for i, data in enumerate(train_loader, start=1):
        # images = data['dwi'][tio.DATA].to(device)
        nihss = data['score'].to(device)
        # print(nihss)
        
        target = torch.LongTensor(data['runn_sum']).to(device)
        output = model(nihss.to(torch.float32)).squeeze(1)
        # print(output)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, params['clip'])
        optimizer.step()
        
        get_logs_reply.metric_stack(output, target, loss = round(loss.item(), 5))
        avg_reply_metric = get_logs_reply.mini_batch_reply(i, epoch, len(stream))
        avg_reply_metric['lr'] = optimizer.param_groups[0]['lr']
        stream.set_description(f"Epoch: {epoch}. Train. {str(avg_reply_metric)}")
    try:
        if (epoch>params['scheduler_epoch']):
            scheduler.step()
    except:
        pass
    for x in avg_reply_metric:
        # print(avg_reply_metric)
        writer.add_scalar(f'{x}/Train {x}', avg_reply_metric[x], epoch)

def validate(valid_loader, model, criterion, epoch):
    global best_vloss, best_vacc
    get_logs_reply2 = logs_realtime_reply()
    model.eval()
    stream_v = tqdm(valid_loader)
    with torch.no_grad():
        for i, data in enumerate(stream_v, start=1):
            images = data['dwi'][tio.DATA].to(device)
            nihss = data['score'].to(device)
            target = torch.LongTensor(data['runn_sum']).to(device)
            # output = model(images.to(torch.float32)).squeeze(1)
            output = model(nihss.to(torch.float32)).squeeze(1)
            loss = criterion(output, target)
            get_logs_reply2.metric_stack(output, target, loss = round(loss.item(), 5))
            avg_reply_metric = get_logs_reply2.mini_batch_reply(i, epoch, len(stream_v))
            stream_v.set_description(f"Epoch: {epoch}. Valid. {str(avg_reply_metric)}")
        avg_reply_metric = get_logs_reply2.epoch_reply()

    for x in avg_reply_metric:
        if x =='Accuracy' and avg_reply_metric[x] > best_vacc:
            best_vacc = avg_reply_metric[x]
            current_loss = avg_reply_metric['Loss']
            save_ck_name = f"{ck_pth}/{project_name} --  epoch:{epoch} | vLoss:{round(current_loss,5)} | vAcc:{round(avg_reply_metric['Accuracy'], 5)}.pt"
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  current_loss,}, save_ck_name)
            print('save...', save_ck_name)
        if x=='Loss' and avg_reply_metric[x]<best_vloss:
            best_vloss = avg_reply_metric[x]
            current_loss = avg_reply_metric['Loss']
            best_ck_name = f'{ck_pth}/best - vloss - {project_name}.pt'
            torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 
                    'loss':  current_loss,}, best_ck_name)
            print('save...', best_ck_name)
        # print(avg_reply_metric)
        writer.add_scalar(f'{x}/Valida {x}', avg_reply_metric[x], epoch)
# X
def  train_valid_process_main(model, training_set, validation_set, batch_size):
    global best_vloss, best_vacc
    best_vloss = np.inf
    best_vacc = 0.00
    # Subject Dataloader Building
    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, 
        shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,  
        shuffle=False, num_workers=2)

    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, loss, optimizer, epoch)
        validate(valid_loader, model, loss, epoch)
    return model

if True: #model record
    params = {
        "type": "runn_sum(out-in)",
        "model": '3dresnet', #baseline = 'resnet18'
        "model_depth": 18,
        "device": "cuda",
        "opt": "Adam",
        "lr": 0.01, #baseline = 0.003
        "scheduler_epoch": None, #nl: 5, ap: None
        "batch_size": 6, #baseline resnet18 : 8
        "epochs": 300,
        "clip":0.5,
        "img size": 384, 
        "img depth": 28,
        "Adjust01": "lr 0.01, weight ce losss, two linear layer, out-in >1",
        "fixing": "None"
        }

if True: #data augmentation, dataloader, 
    training_subjects = tio_process(X_train, np.array(y_train), basepath_ = './dataset/S2_data1.5&3.0/')
    validation_subjects = tio_process(X_test, np.array(y_test), basepath_ = './dataset/S2_data1.5&3.0/')
    print('Training set:', len(training_subjects), 'subjects   ', '||   Validation set:', len(validation_subjects), 'subjects')
    # Transform edit
    training_transform = tio.Compose([])
    validation_transform = tio.Compose([])
    training_set = tio.SubjectsDataset(training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(validation_subjects, transform=validation_transform)

    # checkpoint setting
    project_name = f"{params['type']} - {params['model']}{params['model_depth']} - lr_{params['lr']} - CEL"
    project_folder = f"TEST01.06-03-223_patient_sum_ONLY_NIHSS_score_22(AP+NL & runn_sum(out-in)>=1)"
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
    tensorboard_logdir = f'./logsdir/S2/ {project_folder} - {project_name}'
    writer=SummaryWriter(tensorboard_logdir)
    
if True: #model edit area
    # model create
    # model = model_create(depth=params['model_depth'])
    model = model_create()
    # loss
    class_weights=compute_class_weight(class_weight = 'balanced', classes=np.array([0,1]), y=np.array(table_label['rnn_sum(out-in)']))
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    print("class_weights", class_weights)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss = FocalLoss()
    # optimizer
    if params['opt']=='Adam':
        optimizer = Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay = 1e-4, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max = 100)
    logs  = train_valid_process_main(model, training_set, validation_set, params['batch_size'])

writer.close()