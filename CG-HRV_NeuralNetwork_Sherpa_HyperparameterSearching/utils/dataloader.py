from sklearn import preprocessing
from matplotlib import pyplot
import pandas as pd
def csv_loader(S_3,S_1,S_7):
    data_shape=[(S_3.shape[1]),(S_1.shape[1]),(S_7.shape[1])]
    
    
    return S_3,S_1,S_7,data_shape

def normalized(X):
    header=X.columns
    headers=header.values.tolist()

    train_D=trans_minmax(X,header)
    return train_D


def trans_minmax(X,header):
        ## Select the continuous variables
    conti_var=["Mean.rate","LF.HF.ratio.LombScargle"] #sepsis3
#     conti_var=["Mean.rate","histSI","KLPE","fFdP","pL","SDLEalpha","IoV","QSE"] #siopver1
#     conti_var=["Mean.rate","PSeo","Correlation.dimension","shannEn","QSE"] #siopver1
#     conti_var=["Mean.rate","LF.HF.ratio.LombScargle"]
    X_conti=X[conti_var]
    ## Standardied
    min_max_scaler=preprocessing.MinMaxScaler()
    X_conti=min_max_scaler.fit_transform(X_conti)
    X_conti=pd.DataFrame(X_conti,columns=[conti_var])
    for x in conti_var:
        X[conti_var] = X_conti[conti_var].values

    
#     min_max_scaler=preprocessing.MinMaxScaler()
#     x_scaled=min_max_scaler.fit_transform(X)
#     Transformed_X=pd.DataFrame(x_scaled,columns=[header])
    return X