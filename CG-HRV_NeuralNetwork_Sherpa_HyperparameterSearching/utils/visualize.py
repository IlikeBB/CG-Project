from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc,make_scorer,accuracy_score,precision_score,recall_score,f1_score,cohen_kappa_score,roc_auc_score,confusion_matrix

def train_history_visual(history):
        # plots
    print(history)
    pyplot.figure(figsize=(15,5))
    pyplot.plot(history.history['loss'],'bo')
    pyplot.plot(history.history['val_loss'],'b')
    pyplot.title('model loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')
    pyplot.show()
    
#     pyplot.figure(figsize=(15,5))
#     pyplot.plot(history.history['auroc'],'bo')
#     pyplot.plot(history.history['val_auroc'],'b')
#     pyplot.title('Auroc')
#     pyplot.ylabel('auroc')
#     pyplot.xlabel('epoch')
#     pyplot.legend(['train', 'val'], loc='upper left')
#     pyplot.show()
    
    pyplot.figure(figsize=(15,5))
    pyplot.plot(history.history['accuracy'],'bo')
    pyplot.plot(history.history['val_accuracy'],'b')
    pyplot.title('Accuracy')
    pyplot.ylabel('accuracy')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'val'], loc='upper left')
    pyplot.show()
    
    
def visualize(yhat_probs_train,yhat_probs_test,model_best,train_x,train_y,test_x,test_y):
    # predict crisp classes for test set
    yhat_classes_train = model_best.predict_classes(train_x, verbose=0)
    yhat_classes_test = model_best.predict_classes(test_x, verbose=0)
    # reduce to 1d array
    yhat_probs_train = yhat_probs_train[:, 0]
    yhat_classes_train = yhat_classes_train[:, 0]

    yhat_probs_test = yhat_probs_test[:, 0]
    yhat_classes_test = yhat_classes_test[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy_train = accuracy_score(train_y, yhat_classes_train)
    accuracy_test = accuracy_score(test_y, yhat_classes_test)
    print('=== Accuracy ===')
    print('Accuracy (train): %f' % accuracy_train)
    print('Accuracy (test): %f' % accuracy_test)
    print('\n')

    # precision tp / (tp + fp)
    precision_train = precision_score(train_y, yhat_classes_train)
    precision_test = precision_score(test_y, yhat_classes_test)
    print('=== Precision ===')
    print('Precision (train): %f' % precision_train)
    print('Precision (test): %f' % precision_test)
    print('\n')

    # recall: tp / (tp + fn)
    recall_train = recall_score(train_y, yhat_classes_train)
    recall_test = recall_score(test_y, yhat_classes_test)
    print('=== Recall ===')
    print('Recall (train): %f' % recall_train)
    print('Recall (test): %f' % recall_test)
    print('\n')

    # f1: 2 tp / (2 tp + fp + fn)
    f1_train = f1_score(train_y, yhat_classes_train)
    f1_test = f1_score(test_y, yhat_classes_test)
    print('=== F1 score ===')
    print('F1 score (train): %f' % f1_train)
    print('F1 score (test): %f' % f1_test)
    print('\n') 

    # kappa
    kappa_train = cohen_kappa_score(train_y, yhat_classes_train)
    kappa_test = cohen_kappa_score(test_y, yhat_classes_test)
    print('=== Cohens Kappa ===')
    print('Cohens kappa (train): %f' % kappa_train)
    print('Cohens kappa (test): %f' % kappa_test)
    print('\n')

    # ROC AUC
    auc_train = roc_auc_score(train_y, yhat_probs_train)
    auc_test = roc_auc_score(test_y, yhat_probs_test)
    print('=== ROC AUC ===')
    print('ROC AUC for training dataset: %f' % auc_train)
    print('ROC AUC for testing dataset: %f' % auc_test)
    print('\n')

    # confusion matrix
    matrix_train = confusion_matrix(train_y, yhat_classes_train)
    print('=== Confusion matrix for train datset ===')
    print(matrix_train)
    print('\n')
    matrix_test = confusion_matrix(test_y, yhat_classes_test)
    print('=== Confusion matrix for test datset ===')
    print(matrix_test)

    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(train_y, yhat_probs_train)
    fpr1, tpr1, thresholds1 =  metrics.roc_curve(test_y, yhat_probs_test)
    pyplot.plot(fpr, tpr, color='blue', label='train ROC area = %0.2f)' % auc_train)
    pyplot.plot(fpr1, tpr1, color='red', label='test ROC area = %0.2f)' % auc_test)
    pyplot.xlim([0.0, 1.0])
    pyplot.ylim([0.0, 1.0])
    pyplot.rcParams['font.size'] = 12
    pyplot.title('ROC curve for flu classifier in testing set')
    pyplot.xlabel('False Positive Rate (1 - Specificity)')
    pyplot.ylabel('True Positive Rate (Sensitivity)')
    pyplot.legend(loc="lower right")
    pyplot.grid(True)
    return yhat_classes_train,yhat_classes_test