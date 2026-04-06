import optuna
from optuna.samplers import TPESampler
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from src.train import train_xgboost, train_lightgbm, train_logistic, train_tabnet, train_ft_transformer

def objective_xgboost(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    model = train_xgboost(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

def objective_lightgbm(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
    }
    model = train_lightgbm(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

def objective_logistic(trial, X_train, y_train, X_val, y_val):
    params = {
        'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
        'penalty': 'l2',
        'solver': 'lbfgs'
    }
    model = train_logistic(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

def objective_tabnet(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
    }
    model = train_tabnet(X_train, y_train, X_val, y_val, params)
    y_pred = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_pred)

def objective_ft_transformer(trial, X_num_train, X_cat_train, y_train, X_num_val, X_cat_val, y_val, cat_cardinalities):
    params = {
        'embedding_dim': trial.suggest_int('embedding_dim', 32, 128),
        'n_blocks': trial.suggest_int('n_blocks', 2, 6),
        'n_heads': trial.suggest_int('n_heads', 4, 16),
        'ff_dim': trial.suggest_int('ff_dim', 64, 256),
        'dropout': trial.suggest_float('dropout', 0.0, 0.3),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'epochs': 50,
    }
    model = train_ft_transformer(X_num_train, X_cat_train, y_train,
                                 X_num_val, X_cat_val, y_val,
                                 cat_cardinalities, params)
    device = next(model.parameters()).device
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        from torch.utils.data import DataLoader, TensorDataset
        val_dataset = TensorDataset(torch.FloatTensor(X_num_val), torch.LongTensor(X_cat_val))
        val_loader = DataLoader(val_dataset, batch_size=256)
        for x_num, x_cat in val_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            logits = model(x_num, x_cat)
            probs = torch.sigmoid(logits)
            y_pred_list.append(probs.cpu().numpy())
    y_pred = np.concatenate(y_pred_list)
    return roc_auc_score(y_val, y_pred)

def tune_model(model_name, train_data, val_data, cat_cardinalities=None, n_trials=30):
    if model_name == 'xgboost':
        X_train, y_train = train_data
        X_val, y_val = val_data
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective_xgboost(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
        return study.best_params
    elif model_name == 'lightgbm':
        X_train, y_train = train_data
        X_val, y_val = val_data
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective_lightgbm(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
        return study.best_params
    elif model_name == 'logistic':
        X_train, y_train = train_data
        X_val, y_val = val_data
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective_logistic(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
        return study.best_params
    elif model_name == 'tabnet':
        X_train, y_train = train_data
        X_val, y_val = val_data
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective_tabnet(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
        return study.best_params
    elif model_name == 'ft_transformer':
        (X_num_train, X_cat_train), y_train = train_data
        (X_num_val, X_cat_val), y_val = val_data
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective_ft_transformer(t, X_num_train, X_cat_train, y_train,
                                                          X_num_val, X_cat_val, y_val,
                                                          cat_cardinalities), n_trials=n_trials)
        return study.best_params
    else:
        raise ValueError(f"Unknown model {model_name}")
