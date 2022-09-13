from scipy.sparse.construct import rand
import torch, random, os, multiprocessing
import numpy as np, pandas as pd, nibabel as nib 
from sklearn.model_selection import train_test_split
import torch.backends.cudnn as cudnn
import torchio as tio
# multiprocess cpu 
from sklearn.metrics import *
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from scipy import ndimage
from utils.S1_utils import clip_gradient
from utils.model_res_fu_rnn import generate_model
num_workers = multiprocessing.cpu_count()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True
random.seed(1234)
torch.manual_seed(1234)
# load csv label
temp = '1220'
# print(temp[3::])
if True:
    csv_path = './csv/NIHSS_Continuous_Variable_222patient.csv'
    table_ =  pd.read_csv(csv_path, index_col=False)
    table_temp = table_['out_sum']
    table_label = table_.drop(['ID','rnn_sum(out-in)'] ,axis=1)
    print("table_label.columns.values", len(table_label.columns.values))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(table_label)
    y_nor = scaler.transform(table_label)
    # print(table_label)
    nii_3t_train = sorted([i for i in os.listdir(os.path.join('./dataset/S2_data1.5&3.0/'))])

    new_table_nor = pd.DataFrame(y_nor, columns=table_label.columns)
    new_table_nor['out_sum'] = table_temp
    print(new_table_nor)
    X_train, X_test, y_train, y_test = train_test_split(nii_3t_train, new_table_nor, test_size=0.25, random_state=123) #seed = 42, 123

    # balance dataset
def tio_process(nii_3t_, table_3t_, basepath_='./dataset/S2_data1.5&3.0/'):
    subjects_ = []
    for  (nii_path, nii_table) in zip(nii_3t_ , table_3t_):
        tb_len = nii_table.shape[-1]
        subject = tio.Subject(
            dwi = tio.ScalarImage(os.path.join(basepath_, nii_path)), 
            out_sum = float(nii_table[-1]),
            score= nii_table[0:22])
        subjects_.append(subject)
    return subjects_

class log_reply:
    def __init__(self):
        self.loss_sum =[]
    def get_item(self, loss):
        self.loss_sum.append(loss.cpu().detach().numpy())
        return np.mean(self.loss_sum)


def model_create(depth=18):
    model = generate_model(model_depth=depth, n_input_channels=1, n_classes=2)
    model.to(device)
    return model

# model train
def train(train_loader, model, criterion, optimizer, epoch):
    metric_ = log_reply()
    model.train()
    stream = tqdm(train_loader)
   
    for i, data in enumerate(stream, start=1):
    # for i, data in enumerate(train_loader, start=1):
        images = data['dwi'][tio.DATA].to(device)
        nihss = data['score'].to(device)
        target = data['out_sum'].to(device)
        output = model(images.to(torch.float32), nihss.to(torch.float32)).squeeze(1)
        loss = criterion(output, target.to(torch.float32))
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
        for i, data in enumerate(stream_v, start=1):
            images = data['dwi'][tio.DATA].to(device)
            nihss = data['score'].to(device)
            target = data['out_sum'].to(device)
            output = model(images.to(torch.float32), nihss.to(torch.float32)).squeeze(1)
            loss = criterion(output, target.to(torch.float32))
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
        "type": "out_sum",
        "model": '3dresnet+regression', #baseline = 'resnet18'
        "model_depth": 18,
        "device": "cuda",
        "opt": "SGD",
        "lr": 0.003, #baseline = 0.003
        "scheduler_epoch": 0 , #nl: 5, ap: None
        "batch_size": 6, #baseline resnet18 : 8
        "epochs": 100,
        "clip":0.5,
        "img size": 384, 
        "img depth": 28,
        "Adjust01": "add relu layer before cat layer",
        "fixing": "None"
        }

if True: #data augmentation, dataloader, 
    training_subjects = tio_process(X_train, np.array(y_train), basepath_ = './dataset/S2_data1.5&3.0/')
    validation_subjects = tio_process(X_test, np.array(y_test), basepath_ = './dataset/S2_data1.5&3.0/')
    print('Training set:', len(training_subjects), 'subjects   ', '||   Validation set:', len(validation_subjects), 'subjects')
    # Transform edit
    training_transform = tio.Compose([
        # tio.CropOrPad((params['img size'], params['img size'], params['img depth'])),
        tio.OneOf({
            tio.RandomElasticDeformation(): 0.2,
            tio.RandomFlip(axes=('AP',), flip_probability=1.0): 0.5, #for AP class
            # tio.RandomFlip( flip_probability=1.0): 0.5, #for NL class
            tio.RandomAffine(degrees=15, scales=(1.0, 1.0)): 0.3, 
        }),
    ])
    validation_transform = tio.Compose([])
    training_set = tio.SubjectsDataset(training_subjects, transform=training_transform)

    validation_set = tio.SubjectsDataset(validation_subjects, transform=validation_transform)

    # checkpoint setting
    project_name = f"{params['type']} - {params['model']} - {params['opt']} - lr_{params['lr']} - epoch_{params['epochs']}"
    project_folder = f"TEST01.11-02-222_patient_Out_Nihss-Score_AP+NL_NonSegmentation_DWI+Regression"
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
    tensorboard_logdir = f'./logsdir/Fusion_DWI+Regression/ {project_folder} - {project_name}'
    writer=SummaryWriter(tensorboard_logdir)

if True: #model edit area
    # model create
    model = model_create(depth=params['model_depth'])
    # loss
    loss = torch.nn.MSELoss()
    # optimizer
    if params['opt']=='Adam':
        optimizer = Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay = 1e-4, momentum=0.9)
        scheduler = CosineAnnealingLR(optimizer, T_max = 100)
    logs  = train_valid_process_main(model, training_set, validation_set, params['batch_size'])

writer.close()