from sklearn.model_selection import train_test_split
import torch
import numpy as np
from Bio import SeqIO
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns, os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


import copy
import datetime
from tqdm import tqdm_notebook
def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    running_metric=0.0
    len_data = len(dataset_dl.dataset)
    for xb, yb in tqdm_notebook(dataset_dl):
    # for xb, yb in (dataset_dl):
        xb=xb.to(device)
        yb=yb.to(device)
        # print(type(xb), type(yb.shape))
        output=model(xb)
        loss_b,metric_b=loss_batch(loss_func, output, yb, opt)
        running_loss+=loss_b
        
        if metric_b is not None:
            running_metric+=metric_b
        if sanity_check is True:
            break
    loss=running_loss/float(len_data)
    metric=running_metric/float(len_data)
    return loss, metric

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

def metrics_batch(output, target):
    pred = output.argmax(dim=1, keepdim=True)
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)
    with torch.no_grad():
        metric_b = metrics_batch(output,target)
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        clip_gradient(opt, 0.5)
        opt.step()
    return loss.item(), metric_b


def train_val(model, params):
    if True:
        num_epochs=params["num_epochs"]
        loss_func=params["loss_func"]
        opt=params["optimizer"]
        train_dl=params["train_dl"]
        val_dl=params["val_dl"]
        sanity_check=params["sanity_check"]
        lr_scheduler=params["lr_scheduler"]
        path2weights=params["path2weights"]
        
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')
    
    for epoch in range(num_epochs):
        current_lr=get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,opt)
        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            now_time = (datetime.datetime.now().strftime("%Y%m%d%H%M"))
            torch.save({'state_dict': model.state_dict()}, (path2weights+now_time+'.pt'))
            print("Copied best model weights!")
        
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)
        
        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)
        

        print("train loss: %.6f, dev loss: %.6f,  train accuracy: %.2f,valid accuracy: %.2f" %(train_loss,val_loss, 100*train_metric,100*val_metric))
        print("-"*10) 
    model.load_state_dict(best_model_wts)
        
    return model, loss_history, metric_history