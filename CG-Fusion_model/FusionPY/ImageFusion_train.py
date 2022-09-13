import torch, random, os, multiprocessing
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
import torchio as tio
# multiprocess cpu 
from sklearn.metrics import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
from utils.S1_utils import clip_gradient
from utils.model_res_double import generate_model
num_workers = multiprocessing.cpu_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
random.seed(1234)
torch.manual_seed(1234)
# load csv label
if True:
    csv_path = './csv/NIHSS_fiter_222patient.csv'
    table_ =  pd.read_csv(csv_path, index_col=False)
    table_label = table_['rnn_sum(out-in)']
    table_label = [1 if i >=1 else 0 for i in table_label ]
    nii_3t_train = table_['ID']
    X_train, X_test, y_train, y_test = train_test_split(np.array(nii_3t_train), np.array(table_label), stratify=list(table_label), test_size=0.25, random_state=123) #seed = 42, 123
    print(X_train)
# Subject Function Building
def tio_process_train(nii_3t_, table_3t_, basepath_= ['./dataset/DWI_1-350/', './dataset/T1_1-350/']):
    subjects = [tio.Subject(
            dwi_ = tio.ScalarImage(os.path.join(basepath_[0], nii_path + '_dwi.nii.gz')), 
            T1_ = tio.ScalarImage(os.path.join(basepath_[1], nii_path + '_T1.nii.gz'),), 
            score = nii_table,
            ) for  (nii_path, nii_table) in tqdm(zip(nii_3t_ , table_3t_))]

    return subjects

def tio_process_valid(nii_3t_, table_3t_, basepath_= ['./dataset/DWI_1-350/', './dataset/T1_1-350/']):
    subjects = [tio.Subject(
            dwi_ = tio.ScalarImage(os.path.join(basepath_[0], nii_path + '_dwi.nii.gz')), 
            T1_ = tio.ScalarImage(os.path.join(basepath_[1], nii_path + '_T1.nii.gz')), 
            NIHSS  = nii_table, 
            score = nii_table
            ) for  (nii_path, nii_table) in tqdm(zip(nii_3t_ , table_3t_))]
    return subjects

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

def model_create(depth=10):
    model = generate_model(model_depth=depth, n_input_channels=1, n_classes=2)
    model.to(device)
    return model

# model train
def train(train_loader, model, criterion, optimizer, epoch):
    get_logs_reply = logs_realtime_reply()
    model.train()
    stream = tqdm(train_loader)
   
    for i, data in enumerate(stream, start=1):
        # print(data['dwi_']['data'].shape, data['dwi_']['path'])
        # print(data['T1_']['data'].shape,  data['T1_']['path'])
        images_dwi = data['dwi_'][tio.DATA].to(device)
        images_T1 = data['T1_'][tio.DATA].to(device)
        # print(images_dwi.shape, images_T1.shape, i)
        target = torch.LongTensor(data['score']).to(device)
        output = model(images_dwi.permute(0,1,4,2,3), images_T1.permute(0,1,4,2,3)).squeeze(1)
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
# model validate
def validate(valid_loader, model, criterion, epoch):
    global best_vloss, best_vacc
    get_logs_reply2 = logs_realtime_reply()
    model.eval()
    stream_v = tqdm(valid_loader)
    with torch.no_grad():
        for i, data in enumerate(stream_v, start=1):
            images_dwi = data['dwi_'][tio.DATA].to(device)
            images_T1 = data['T1_'][tio.DATA].to(device)
            target = torch.LongTensor(data['score']).to(device)
            output = model(images_dwi.permute(0,1,4,2,3), images_T1.permute(0,1,4,2,3)).squeeze(1)
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
        shuffle=False, num_workers=10)
    valid_loader = torch.utils.data.DataLoader(validation_set, batch_size=batch_size,  
        shuffle=False, num_workers=4)

    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, loss, optimizer, epoch)
        validate(valid_loader, model, loss, epoch)
    return model

if True: #model record
    params = {
        "type": "out-in",
        "model": '3dresnet', #baseline = 'resnet18'
        "model_depth": 10,
        "device": "cuda",
        "opt": "Adam",
        "lr": 0.01, #baseline = 0.003
        "scheduler_epoch": None , #nl: 5, ap: None
        "batch_size": 3, #baseline resnet18 : 8
        "epochs": 150,
        "clip":0.5,
        "img size": 384, 
        "img depth": 28,
        # "adjust01": "CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, last_epoch=-1)",
        # "augment": "RandomFlip['AP'], RandomElasticDeformation",
        "augment": "Resample(512,512,28), RandomElasticDeformation, RandomFlip, RandomAffine, class weight",
        "fixing": "None"
        }

if True: #data augmentation, dataloader, 
    # Transform edit
    training_transform = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        # tio.ToCanonical(), 
        tio.Resample('T1_'),
        # tio.RandomFlip(axes=('AP',)), #for AP class
        # tio.RandomAffine(scales=(0.9,1.2), degrees=15),

    ])
    validation_transform = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.Resample('T1_'),
    ])

    training_subjects = tio_process_train(X_train, y_train, basepath_ = ['./dataset/DWI_1-350/', './dataset/T1_1-350/'])
    validation_subjects = tio_process_valid(X_test, y_test, basepath_ = ['./dataset/DWI_1-350/', './dataset/T1_1-350/'])
    print('Training set:', len(training_subjects), 'subjects   ', '||   Validation set:', len(validation_subjects), 'subjects')
    training_set = tio.SubjectsDataset(training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(validation_subjects, transform=validation_transform)

    # checkpoint setting
    project_name = f"{params['type']} - {params['model']} - {params['opt']} - lr_{params['lr']} - epoch_{params['epochs']}"
    project_folder = f"2022.02.07 - Test02"
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
    model = model_create(depth=params['model_depth'])
    # loss
    from sklearn.utils.class_weight import compute_class_weight
    class_weights=compute_class_weight(class_weight = 'balanced', classes=np.array([0,1]), y=np.array(table_label))
    class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
    print("class_weights", class_weights)
    loss = torch.nn.CrossEntropyLoss(weight=class_weights)
    # loss = FocalLoss()
    # optimizer
    if params['opt']=='Adam':
        optimizer = Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay = 1e-4, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.01, patience=3)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, last_epoch=-1)
    logs  = train_valid_process_main(model, training_set, validation_set, params['batch_size'])

    writer.close()