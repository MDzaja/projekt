import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error

def grid_search_cv_acc(features_train, labels_train, cv_inidces_list, param_grid=None, sw_train=None, scale_pos_weight=None, saving_file=None):
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

def test_model(features_train, features_test, labels_train, labels_test, model_file_path, sw_train=None, sw_test=None):
    with open(model_file_path, 'r') as fp:
        model_params = json.load(fp)
    xgb_model = xgb.XGBClassifier(**model_params)

    fit_params ={}
    if sw_train is not None:
        fit_params['sample_weight'] = sw_train
    xgb_model.fit(features_train, labels_train, **fit_params)

    test_params ={}
    if sw_test is not None:
        test_params['sample_weight'] = sw_test
    
    predictions = xgb_model.predict(features_test)
    metrics = {
        'accuracy': accuracy_score(labels_test, predictions,**test_params),
        'f1': f1_score(labels_test, predictions,**test_params),
        'mse': mean_squared_error(labels_test, predictions,**test_params),
        'auc': roc_auc_score(labels_test, predictions,**test_params)
    }

    return metrics