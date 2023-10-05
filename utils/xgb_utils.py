import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import json
import pandas as pd
from skopt.space import Real, Integer
from skopt import BayesSearchCV

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

def grid_search_cv_acc(features_train: pd.DataFrame, labels_train: pd.Series, cv_inidces_list: list, 
            param_grid: dict=None, sw_train: pd.Series=None, scale_pos_weight: float=None, saving_file: str=None) -> dict:
    
    if param_grid == None:
        param_grid = {
                'max_depth': [3, 6, 9],
                'n_estimators': [100, 250, 500],
                'learning_rate': [0.05, 0.15, 0.3],
                'colsample_bytree': [0.5, 0.7, 1],
                'subsample': [0.6, 0.8, 1],
                'min_child_weight': [1, 10, 50]
                }
    xgb_model = xgb.XGBClassifier()
    if scale_pos_weight is not None:
        xgb_model.set_params(scale_pos_weight=scale_pos_weight)
    clf = GridSearchCV(estimator=xgb_model, 
                    param_grid=param_grid,
                    scoring='accuracy',
                    n_jobs=-1,
                    cv=cv_inidces_list,
                    verbose=3)
    fit_params ={}
    if sw_train is not None:
        fit_params['sample_weight'] = sw_train
    clf.fit(features_train, labels_train, **fit_params)
    if scale_pos_weight is not None:
        clf.best_params_['scale_pos_weight'] = scale_pos_weight

    print('Best parameters:', clf.best_params_)
    print('Best accuracy score: {}%'.format(clf.best_score_*100))

    if saving_file is not None:
        with open(saving_file, 'w') as fp:
            json.dump(clf.best_params_, fp)

    return clf.best_params_

def test_model(X_train: pd.DataFrame, X_test: pd.DataFrame, Y_train: pd.Series,
               Y_test: pd.Series, model_file_path: str, refit_frequency: int=5, sw_train: pd.Series=None, 
               sw_test: pd.Series=None) -> dict:
    
    with open(model_file_path, 'r') as fp:
        model_params = json.load(fp)
    xgb_model = xgb.XGBClassifier(**model_params)

    fit_params ={}
    if sw_train is not None:
        fit_params['sample_weight'] = sw_train
    xgb_model.fit(X_train, Y_train, **fit_params)

    test_params ={}
    if sw_test is not None:
        test_params['sample_weight'] = sw_test
        
    counter = 0
    predictions = []
    for i in range(len(X_test)):
        counter += 1
        if counter == refit_frequency:
            #print(X_train.shape, Y_train.shape)
            xgb_model.fit(X_train, Y_train)
            counter = 0
            
        prediction = xgb_model.predict(X_test.iloc[[i]])[0]
        predictions.append(prediction)
        X_train = pd.concat([X_train, X_test.iloc[i:i+1]], ignore_index=True)
        Y_train = pd.concat([Y_train, pd.Series([Y_test[i]])], ignore_index=True)
    
    metrics = {
        'accuracy': accuracy_score(Y_test, predictions,**test_params),
        'f1': f1_score(Y_test, predictions,**test_params),
        'mse': mean_squared_error(Y_test, predictions,**test_params),
        'auc': roc_auc_score(Y_test, predictions,**test_params)
    }

    return metrics

def bayes_search_cv_custom(features_train: pd.DataFrame, labels_train: pd.Series, cv_inidces_list: list, 
            param_grid: dict=None, sw_train: pd.Series=None, scale_pos_weight: float=None, saving_file: str=None) -> dict:
    
    if param_grid == None:
        param_grid = {
                'n_estimators': Integer(low=50, high=2000, prior='uniform'),
                'max_depth': Integer(low=3, high=20, prior='uniform'),
                'gamma': Real(low=0.1, high=1, prior='uniform'),
                'learning_rate': Real(low=0.01, high=0.35, prior='uniform'),
                'reg_alpha': Real(low=0, high=1, prior='uniform'),
                'reg_lambda': Real(low=0, high=5, prior='uniform'),
                'min_child_weight': Integer(low=0, high=10, prior='uniform'),
                'subsample': Real(low=0.7, high=1.0, prior='uniform'),
                }
        
    xgb_model = xgb.XGBClassifier()
    if scale_pos_weight is not None:
        xgb_model.set_params(scale_pos_weight=scale_pos_weight)
    clf = BayesSearchCV(
            xgb_model,
            param_grid,
            n_iter=50,  # Number of optimization iterations
            cv=cv_inidces_list,       # Cross-validation folds
            scoring='roc_auc',
            random_state=42,
            verbose=3
        )
    fit_params ={}
    if sw_train is not None:
        fit_params['sample_weight'] = sw_train
    clf.fit(features_train, labels_train, **fit_params)
    if scale_pos_weight is not None:
        clf.best_params_['scale_pos_weight'] = scale_pos_weight

    print('Best parameters:', clf.best_params_)
    print('Best accuracy score: {}%'.format(clf.best_score_*100))

    if saving_file is not None:
        with open(saving_file, 'w') as fp:
            json.dump(clf.best_params_, fp)

    return clf.best_params_