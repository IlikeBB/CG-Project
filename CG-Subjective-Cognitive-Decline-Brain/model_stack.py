from import_library import *
# model list

def model_list(grid=False, seed=123):
    # svr_param = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')}
    # SVR_ = SVR()
    
    xgbr_param = {'nthread':[4], 'objective':['reg:squarederror'], 'learning_rate': [.03, 0.05, .07], 'max_depth': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 'min_child_weight': [4], 
                                    'subsample': [0.7], 'colsample_bytree': [0.7], 'n_estimators': [400, 450, 500, 550, 600, 660]}

    # xgbr_param = {'nthread':[4], 'objective':['reg:squarederror'], 'learning_rate': [.07], 'max_depth': [1]}

    XGBR_ =XGBRegressor(random_state = seed)

    lgbmr_param =  {'num_leaves': [7, 14, 21], 'learning_rate': [0.01, 0.05, 0.001, 0.005], 'max_depth': [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31], 
                                        'min_data_in_leaf':[10, 15, 25], 'feature_fraction': [0.6, 0.8, 0.9],'cat_smooth': [1,10, 15, 20, 35], 'verbose': [-1]}
    
    # lgbmr_param =  {'learning_rate': [0.01, 0.05, 0.001,0.005], 'max_depth': [1,3,5,7], 'verbose': [1]}
    
    
    LGBMR_ = LGBMRegressor(random_state=seed, verbose=-1)

    # mlpr_param =  {'hidden_layer_sizes': [(25), (50), (100)]}

    mlpr_param =  {'hidden_layer_sizes': [(25), (50), (75),(100), (25, 50), (50,100), (25, 50, 75),(50, 75, 100)], 'learning_rate_init': [0.01, 0.03, 0.001, 0.003, 0.0001, 0.0003],
                                        'activation': ['relu', 'tanh'], 'solver': ['adam', 'lbfgs'], 'learning_rate' : ['constant', 'adaptive', 'invscaling']}

    MLPR_ = MLPRegressor(activation = 'relu', solver = 'adam', max_iter=20000, random_state=seed, early_stopping=True, validation_fraction=0.50, verbose=0)

    if grid ==True:
        # model_stack = {"SVR":[SVR_, svr_param], "XBGR":[XGBR_, xgbr_param], "LGBMR": [LGBMR_, lgbmr_param]}
        model_stack = {"MLPR":[MLPR_, mlpr_param], "XBGR":[XGBR_, xgbr_param], "LGBMR": [LGBMR_, lgbmr_param]}
        # model_stack = {"LGBMR": [LGBMR_, lgbmr_param]}
        # model_stack = {"XBGR":[XGBR_, xgbr_param], "LGBMR": [LGBMR_, lgbmr_param]}
    else:
        # model_stack = {"SVR":SVR_, "XBGR":XGBR_, "LGBMR": LGBMR_}
        # model_stack = {"XBGR":XGBR_, "LGBMR": LGBMR_}
        model_stack = {"MLPR":MLPR_, "XBGR":XGBR_, "LGBMR": LGBMR_}
    return model_stack