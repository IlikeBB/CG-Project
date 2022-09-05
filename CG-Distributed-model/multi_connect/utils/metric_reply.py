import numpy as np
import torch
from sklearn import metrics


class logs_realtime_reply:
    def __init__(self):
        self.avg_dice = 0.0
        self.avg_loss=np.inf
        self.avg_tn = 0
        self.avg_fp = 0
        self.avg_fn = 0
        # self.running_metic = {"Loss":0, "TP":0, "FP":0, "FN": 0, "Spec": 0, "Sens": 0}
        # self.running_metic = {"Loss":0, "Accuracy":0, "Spec": 0, "Sens": 0, "AUC": 0}
        self.running_metic = {"Loss":0,"Accuracy":0, "AUC": 0}
        self.end_epoch_metric = None
    def metric_stack(self, inputs, targets, loss):
        with torch.no_grad():
            self.running_metic['Loss'] +=loss
            # metric setting
            SR = inputs.cpu().data.numpy()
            GT = targets.cpu().data.numpy()
            # print("SR", SR)
            # print("GT", GT)
            acc = metrics.accuracy_score(SR>0.5, GT)
            fpr, tpr, thresholds = metrics.roc_curve(GT, SR, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            self.running_metic['Accuracy'] += round((acc), 5)
            self.running_metic['AUC'] += round((auc), 5)
    def mini_batch_reply(self, current_step, epoch, iter_len):
        # avg_reply_metric = {"Loss":None, "TP":None, "FP":None, "FN": None, "Spec": None, "Sens": None}
        avg_reply_metric = {"Loss":None, "Accuracy": None, "AUC": None}
        # avg_reply_metric = {"Loss":None, "Accuracy": None,"Spec": None, "Sens": None, "AUC": None}
        for j in avg_reply_metric:
            avg_reply_metric[j] = round(self.running_metic[j]/int(current_step),5)
        
        if current_step ==iter_len:
            self.end_epoch_metric = avg_reply_metric
        return avg_reply_metric

    def epoch_reply(self):
        return self.end_epoch_metric