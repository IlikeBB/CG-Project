import torch, copy, numpy as np
import matplotlib.pyplot as plt
from model.model_mlp_seq import MLP
from sklearn import metrics
from torchvision.models import resnet18
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch import optim
from torch import nn

class train_val_function:
    def __init__(self,):
        self.model = None
        self.cuda_status = None
        self.model_type = None
        self.opt = None
        self.loss_func = None
        self.lr_scheduler = None
        self.multi_model_check = False

    def model_config(self, model_type = 'cnn', data_shape = 1000, class_num = 1, use_cuda = True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if class_num == 1:
            self.multi_model_check = False
        else:
            self.multi_model_check = True

        self.model_type = model_type
        if use_cuda == True:
            self.cuda_status = device
        else:
            self.cuda_status = 'cpu'

        if self.model_type =='cnn':
            model = resnet18(pretrained=False, num_classes=class_num)
            self.model = model.to(self.cuda_status)
        else:
            self.model = MLP(class_num, data_shape[1]).to(self.cuda_status)
        # model.initialize_weights()
    def get_model(self,):
        return self.model

    def clip_gradient(self, optimizer, grad_clip):
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    param.grad.data.clamp_(-grad_clip, grad_clip)

    def get_lr(self, opt):
        for param_group in opt.param_groups:
            return param_group['lr']

    def metrics_batch(self, output, target):
        acc = metrics.accuracy_score(target.cpu().detach().numpy(), output.cpu().detach().numpy()>0.5)
        return acc

    def auc_batch(self, output, target):
        fpr, tpr, _ = metrics.roc_curve(target.cpu().detach().numpy(), output.cpu().detach().numpy(), pos_label=1)
        return np.round(metrics.auc(fpr, tpr), 5)
    
    def loss_batch(self, loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            metric_b = self.metrics_batch(output,target)
            auc_b = self.auc_batch(output,target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), metric_b, auc_b

    def loss_epoch(self, model, loss_func,dataset_dl, sanity_check=False, opt=None):
        running_loss=0.0
        running_metric=0.0
        running_auc=0.0
        len_data = len(dataset_dl)
        for xb, yb in (dataset_dl):
            xb=xb.to(self.cuda_status)
            if self.multi_model_check == False:
                yb=yb.to(self.cuda_status).float()
            else:
                yb=yb.to(self.cuda_status)
            # print(type(xb), type(yb.shape))
            # output=torch.squeeze(torch.sigmoid(model(xb)))
            output=torch.squeeze(model(xb))
            if self.multi_model_check == False:
                output = torch. sigmoid(output)
            loss_b,metric_b, auc_b = self.loss_batch(loss_func, output, yb, opt)
            running_loss+=loss_b
            
            if metric_b is not None:
                running_metric+=metric_b
            if auc_b is not None:
                running_auc+=auc_b
            if sanity_check is True:
                break
        loss=running_loss/float(len_data)
        metric=running_metric/float(len_data)
        aucs=running_auc/float(len_data)
        return loss, metric, aucs
    
    def model_params(self, loss_name = 'BCE', opt_name = 'Adam' , lr_scheduler_name = 'ReduceLROnPlateau'):
        if loss_name =='BCE':
            self.loss_func = nn.BCELoss(reduction='mean')
        elif loss_name =='CE':
            self.loss_func = nn.CrossEntropyLoss(reduction="sum")
        if opt_name == 'Adam':
            self.opt = optim.Adam(self.model.parameters(), lr=0.003)
        if lr_scheduler_name != None:
            self.lr_scheduler = ReduceLROnPlateau(self.opt, mode='min',factor=0.5, patience=5,verbose=1)

    def train_val_main(self, params):
        model = self.model
        num_epochs=params["num_epochs"]
        train_dl=params["train_dl"]
        val_dl=params["val_dl"]
        sanity_check=params["sanity_check"]
        path2weights=params["path2weights"]
        self.model_params(params["loss_func"], params["optimizer"], params["lr_scheduler"])
    
        loss_history={"train": [], "val": [],}
        metric_history={"train": [], "val": [],}
        auc_history={"train": [], "val": [],}

        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss=float('inf')
        
        for epoch in range(num_epochs):
            current_lr = self.get_lr(self.opt)
            print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))
            model.train()
            train_loss, train_metric, train_auc = self.loss_epoch(model, self.loss_func,train_dl,sanity_check, self.opt)
            loss_history["train"].append(train_loss)
            metric_history["train"].append(train_metric)
            auc_history["train"].append(train_auc)
            model.eval()
            with torch.no_grad():
                val_loss, val_metric, val_auc = self.loss_epoch(model, self.loss_func,val_dl,sanity_check)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), path2weights)
                print("Copied best model weights!")
            
            loss_history["val"].append(val_loss)
            metric_history["val"].append(val_metric)
            auc_history["val"].append(val_auc)
            if self.lr_scheduler != None:
                self.lr_scheduler.step(val_loss)
            if current_lr != self.get_lr(self.opt):
                print("Loading best model weights!")
                model.load_state_dict(best_model_wts)
            

            print("train loss: %.6f, dev loss: %.6f,  train accuracy: %.2f,valid accuracy: %.2f" %(train_loss,val_loss, 100*train_metric,100*val_metric))
            print("train auc: %.2f,valid auc: %.2f" %( 100*train_auc,100*val_auc))
            print("-"*10) 
        model.load_state_dict(best_model_wts)
            
        return loss_history, metric_history, auc_history


class test_model:
    def __init__(self,):
            self.model = None
            self.cuda_status = None
            self.model_type = None
            self.opt = None
            self.loss_func = None
            self.lr_scheduler = None
            self.multi_model_check = False
            self.prob_label = [None,None]
    def model_config(self, model_type = 'cnn', data_shape = 1000, class_num = 1, use_cuda = True):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if class_num == 1:
            self.multi_model_check = False
        else:
            self.multi_model_check = True

        self.model_type = model_type
        if use_cuda == True:
            self.cuda_status = device
        else:
            self.cuda_status = 'cpu'

        if self.model_type =='cnn':
            model = resnet18(pretrained=False, num_classes=class_num)
            self.model = model.to(self.cuda_status)
        else:
            self.model = MLP(class_num, data_shape[1]).to(self.cuda_status)
        # model.initialize_weights()
    def get_model(self,):
        return self.model

    def ck_loader(self, path):
        self.model.load_state_dict(torch.load(path))

    def test_process(self, dl, output='numpy'):
        with torch.no_grad():
            for (data_, label_) in dl:
                prob = torch.sigmoid(self.model(data_.to(self.cuda_status)))
        self.prob_label[0] = prob.detach().cpu().numpy()
        self.prob_label[1] = label_.detach().cpu().numpy()
        if output == 'numpy':
            return self.prob_label[0], self.prob_label[1]
        else:
            return prob, label_
    def get_metric(self, save_name = None, fig_name = 'ROC Plot'):
        fpr, tpr, _ = metrics.roc_curve(self.prob_label[1], self.prob_label[0], pos_label=1)
        auc = np.round(metrics.auc(fpr, tpr), 5)
        plt.plot(fpr, tpr, color='red', label=f'ROC [area = {auc}]',)
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(fig_name)
        plt.legend()
        
        if save_name!=None:
            plt.savefig(save_name)
        plt.show()