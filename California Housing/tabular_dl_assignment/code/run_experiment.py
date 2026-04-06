"""
Deep Learning for Tabular Data Assignment
Dataset: California Housing (Regression)
Models: Ridge, RandomForest, XGBoost, LightGBM, CatBoost, TabNet, FT-Transformer
Features:
  - Checkpoint/Resume: saves after each model, auto-resumes on restart
  - Model persistence: saves model files to disk
  - CPU multi-core: n_jobs=16 for all sklearn/boosting models
  - GPU acceleration: XGBoost/CatBoost CUDA, FT-Transformer CUDA+AMP
  - TabNet: CPU mode (faster for small datasets, avoids GPU transfer overhead)
  - AMP (mixed precision): FP16 for FT-Transformer (2x speedup on RTX 4090)
  - torch.compile: JIT compilation for FT-Transformer
  - DataLoader: num_workers=4, pin_memory, prefetch for FT-Transformer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time, os, json, sys, joblib
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import rtdl
from pytorch_tabnet.tab_model import TabNetRegressor as TabNetLib

# CPU speed optimization (for TabNet CPU mode)
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.backends.cudnn.enabled = False  # not needed on CPU

# ============================================================
# TabNet - Official pytorch-tabnet library (CPU mode for small datasets)
# Arik & Pfenning (2021) "TabNet: Attentive Interpretable Tabular Learning"
# Using CPU mode: avoids GPU transfer overhead for 20K row dataset
# ============================================================
# (TabNetLib imported above as TabNetRegressor from pytorch_tabnet)

# ============================================================
# Config
# ============================================================
SEEDS    = [42, 123, 456]
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
N_TRIALS = 50
WORKDIR  = '/root/tabular_dl_assignment'
N_CPU    = min(os.cpu_count() or 8, 16)  # cap at 16 to avoid thread contention on 208-core server

os.makedirs(f'{WORKDIR}/results', exist_ok=True)
os.makedirs(f'{WORKDIR}/figures', exist_ok=True)
os.makedirs(f'{WORKDIR}/models',  exist_ok=True)

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Device: {DEVICE}, CPU cores: {N_CPU}")

# Enable TF32 for extra speed on Ampere+ GPUs (RTX 4090)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# ============================================================
# Load data
# ============================================================
print("\nLoading California Housing dataset...")
df = pd.read_csv(f'{WORKDIR}/california_housing.csv')
feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].values.astype(np.float32)
y = df['target'].values.astype(np.float32)
print(f"Shape: {X.shape}, Target: {y.shape}")
print(f"Features: {feature_names}")

# ============================================================
# Utilities
# ============================================================
def prepare_data(seed):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=seed)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)
    X_test_s  = scaler.transform(X_test).astype(np.float32)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test

def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return rmse, mae, r2

def inf_time(predict_fn, X, n=5):
    times = []
    for _ in range(n):
        t = time.time(); predict_fn(X); times.append(time.time()-t)
    return np.mean(times)*1000/len(X)

# ============================================================
# Checkpoint & Model Save/Load
# ============================================================
CHECKPOINT_FILE = f'{WORKDIR}/results/checkpoint.json'

def save_checkpoint(all_results, all_preds, all_params, completed):
    ckpt = {
        'all_results': all_results,
        'all_preds':   all_preds,
        'all_params':  all_params,
        'completed':   completed,
    }
    # Write to temp file first, then rename (atomic write)
    tmp = CHECKPOINT_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(ckpt, f)
    os.replace(tmp, CHECKPOINT_FILE)
    print(f"  [CHECKPOINT] Saved {len(completed)} tasks", flush=True)

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                ckpt = json.load(f)
            print(f"[RESUME] Found checkpoint: {len(ckpt['completed'])} tasks done")
            for c in ckpt['completed']:
                print(f"  - {c}")
            return (ckpt['all_results'], ckpt['all_preds'],
                    ckpt['all_params'], ckpt['completed'])
        except Exception as e:
            print(f"[WARN] Checkpoint load failed: {e}, starting fresh")
    return None

def save_model(model, name, seed):
    path = f'{WORKDIR}/models/{name}_seed{seed}'
    try:
        if name == 'TabNet':
            # Official pytorch-tabnet: use built-in save_model
            model.save_model(path)  # saves path.zip
        elif name in ('FT-Transformer', 'ResNet'):
            torch.save({
                'state_dict': model.model.state_dict(),
                'cfg': model.cfg,
            }, path + '.pt')
        else:
            joblib.dump(model, path + '.joblib', compress=3)
        print(f"  [MODEL SAVED] {name} seed={seed}", flush=True)
    except Exception as e:
        print(f"  [MODEL SAVE WARN] {name}: {e}", flush=True)

def load_model(name, seed, n_features=None):
    path = f'{WORKDIR}/models/{name}_seed{seed}'
    try:
        if name == 'TabNet':
            # Official pytorch-tabnet: use built-in load_model
            m = TabNetLib(device_name='cpu', verbose=0)
            m.load_model(path + '.zip')
            return m
        elif name == 'FT-Transformer':
            ckpt = torch.load(path + '.pt', map_location=DEVICE)
            cfg = ckpt['cfg']
            m = FTTransformerModel(**cfg)
            m.model = m._build()
            m.model.load_state_dict(ckpt['state_dict'])
            return m
        elif name == 'ResNet':
            ckpt = torch.load(path + '.pt', map_location=DEVICE)
            cfg = ckpt['cfg']
            m = ResNetModel(**cfg)
            m.model = m._build()
            m.model.load_state_dict(ckpt['state_dict'])
            return m
        else:
            return joblib.load(path + '.joblib')
    except Exception as e:
        print(f"  [MODEL LOAD WARN] {name}: {e}", flush=True)
        return None

# ============================================================
# Classical Models (CPU multi-core)
# ============================================================
def tune_ridge(Xtr, ytr, Xvl, yvl, seed):
    def obj(trial):
        m = Ridge(alpha=trial.suggest_float('alpha', 1e-4, 1e4, log=True))
        m.fit(Xtr, ytr)
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    m = Ridge(**s.best_params); m.fit(Xtr, ytr)
    return m, s.best_params

def tune_rf(Xtr, ytr, Xvl, yvl, seed):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 50, 400),
            max_depth=trial.suggest_int('max_depth', 3, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features',
                                                   ['sqrt', 'log2', 0.5]))
        m = RandomForestRegressor(**p, random_state=seed, n_jobs=N_CPU)
        m.fit(Xtr, ytr)
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS, n_jobs=1)
    m = RandomForestRegressor(**s.best_params, random_state=seed, n_jobs=N_CPU)
    m.fit(Xtr, ytr)
    return m, s.best_params

def tune_xgb(Xtr, ytr, Xvl, yvl, seed):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            reg_alpha=trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-8, 10, log=True))
        m = xgb.XGBRegressor(**p, random_state=seed,
                              tree_method='hist', device='cuda', verbosity=0)
        m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)], verbose=False)
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    m = xgb.XGBRegressor(**s.best_params, random_state=seed,
                          tree_method='hist', device='cuda', verbosity=0)
    m.fit(Xtr, ytr)
    return m, s.best_params

def tune_lgb(Xtr, ytr, Xvl, yvl, seed):
    def obj(trial):
        p = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 800),
            max_depth=trial.suggest_int('max_depth', 3, 12),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            num_leaves=trial.suggest_int('num_leaves', 20, 200),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
            min_child_samples=trial.suggest_int('min_child_samples', 5, 100))
        m = lgb.LGBMRegressor(**p, random_state=seed, n_jobs=N_CPU, verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    m = lgb.LGBMRegressor(**s.best_params, random_state=seed,
                           n_jobs=N_CPU, verbose=-1)
    m.fit(Xtr, ytr)
    return m, s.best_params

def tune_cat(Xtr, ytr, Xvl, yvl, seed):
    def obj(trial):
        p = dict(
            iterations=trial.suggest_int('iterations', 100, 800),
            depth=trial.suggest_int('depth', 3, 10),
            learning_rate=trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1e-8, 10, log=True),
            bagging_temperature=trial.suggest_float('bagging_temperature', 0, 1))
        m = cb.CatBoostRegressor(**p, random_seed=seed, task_type='GPU', verbose=0)
        m.fit(Xtr, ytr, eval_set=(Xvl, yvl), verbose=0)
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    m = cb.CatBoostRegressor(**s.best_params, random_seed=seed,
                              task_type='GPU', verbose=0)
    m.fit(Xtr, ytr)
    return m, s.best_params

# ============================================================
# TabNet - Official pytorch-tabnet library (CPU mode)
# Arik & Pfenning (2021) "TabNet: Attentive Interpretable Tabular Learning"
# ============================================================
def tune_tabnet(Xtr, ytr, Xvl, yvl, seed):
    # Ensure float32 numpy arrays (CPU-friendly)
    Xtr = Xtr.astype('float32')
    Xvl = Xvl.astype('float32')
    ytr = ytr.astype('float32')
    yvl = yvl.astype('float32')
    def obj(trial):
        p = dict(
            n_d=trial.suggest_categorical('n_d', [8, 16, 32]),
            n_a=trial.suggest_categorical('n_a', [8, 16, 32]),
            n_steps=trial.suggest_int('n_steps', 2, 5),
            n_independent=trial.suggest_int('n_independent', 1, 2),
            n_shared=trial.suggest_int('n_shared', 1, 2),
            gamma=trial.suggest_float('gamma', 1.0, 2.0),
            lambda_sparse=0,
            optimizer_params=dict(
                lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True)))
        m = TabNetLib(
            seed=seed, device_name='cpu',
            verbose=0, **p)
        m.fit(
            Xtr, ytr.reshape(-1, 1),
            eval_set=[(Xvl, yvl.reshape(-1, 1))],
            eval_name=['val'], eval_metric=['rmse'],
            max_epochs=150, patience=15,
            batch_size=256, virtual_batch_size=64,
            num_workers=0,
            drop_last=False)
        preds = m.predict(Xvl).flatten()
        return np.sqrt(mean_squared_error(yvl, preds))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    bp = s.best_params
    lr = bp.pop('lr')
    m = TabNetLib(
        seed=seed, device_name='cpu',
        verbose=0,
        lambda_sparse=0,
        optimizer_params=dict(lr=lr),
        **bp)
    m.fit(
        Xtr, ytr.reshape(-1, 1),
        eval_set=[(Xvl, yvl.reshape(-1, 1))],
        eval_name=['val'], eval_metric=['rmse'],
        max_epochs=150, patience=15,
        batch_size=256, virtual_batch_size=64,
        num_workers=0,
        drop_last=False)
    bp['lr'] = lr
    return m, bp

# ============================================================
# FT-Transformer (GPU + AMP + torch.compile + multi-worker)
# ============================================================
class FTTransformerModel:
    def __init__(self, n_features, d_token=128, n_blocks=3, n_heads=8,
                 attn_drop=0.1, ffn_hidden=256, ffn_drop=0.1,
                 lr=3e-4, batch_size=512, max_epochs=200, patience=20,
                 seed=42, device=DEVICE):
        # n_heads is stored but not passed to make_baseline (not supported in this rtdl version)
        self.cfg = dict(
            n_features=n_features, d_token=d_token, n_blocks=n_blocks,
            n_heads=n_heads, attn_drop=attn_drop, ffn_hidden=ffn_hidden,
            ffn_drop=ffn_drop, lr=lr, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience, seed=seed)
        self.device = device
        self.model = None

    def _build(self):
        cfg = self.cfg
        torch.manual_seed(cfg['seed'])
        d, h = cfg['d_token'], cfg['n_heads']
        if d % h != 0:
            h = 4 if d % 4 == 0 else 1
        m = rtdl.FTTransformer.make_baseline(
            n_num_features=cfg['n_features'],
            cat_cardinalities=None,
            d_token=d,
            n_blocks=cfg['n_blocks'],
            attention_dropout=cfg['attn_drop'],
            ffn_d_hidden=cfg['ffn_hidden'],
            ffn_dropout=cfg['ffn_drop'],
            residual_dropout=0.0,
            last_layer_query_idx=[-1],
            d_out=1,
        )
        return m.to(self.device)

    def fit(self, Xtr, ytr, Xvl, yvl):
        cfg = self.cfg
        torch.manual_seed(cfg['seed'])
        self.model = self._build()

        # NOTE: torch.compile disabled - causes hang with rtdl FTTransformer
        # GPU + AMP is already fast enough on RTX 4090
        opt   = torch.optim.AdamW(self.model.parameters(),
                                   lr=cfg['lr'], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=cfg['max_epochs'])
        crit  = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()  # AMP scaler

        # CPU tensors for DataLoader, move batches to GPU with non_blocking
        Xtr_t = torch.FloatTensor(Xtr)
        ytr_t = torch.FloatTensor(ytr)
        Xvl_t = torch.FloatTensor(Xvl).to(self.device)
        yvl_t = torch.FloatTensor(yvl).to(self.device)

        loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t),
            batch_size=cfg['batch_size'], shuffle=True,
            num_workers=0, pin_memory=False)

        best_val, best_state, patience_cnt = float('inf'), None, 0

        for epoch in range(cfg['max_epochs']):
            self.model.train()
            for Xb, yb in loader:
                Xb = Xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                opt.zero_grad()
                # AMP mixed precision (FP16 on RTX 4090 Tensor Cores)
                with torch.cuda.amp.autocast():
                    pred = self.model(Xb, None).squeeze(-1)
                    loss = crit(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                vl = crit(self.model(Xvl_t, None).squeeze(-1), yvl_t).item()
            sched.step()

            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone()
                              for k, v in self.model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= cfg['patience']:
                break

        if best_state:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.model(
                torch.FloatTensor(X).to(self.device), None).squeeze(-1)
        return out.cpu().float().numpy()

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())


def tune_ftt(Xtr, ytr, Xvl, yvl, seed):
    nf = Xtr.shape[1]
    def obj(trial):
        d = trial.suggest_categorical('d_token', [64, 128, 192, 256])
        h = trial.suggest_categorical('n_heads', [4, 8])
        if d % h != 0: h = 4
        p = dict(
            d_token=d, n_blocks=trial.suggest_int('n_blocks', 1, 4),
            n_heads=h,
            attn_drop=trial.suggest_float('attn_drop', 0.0, 0.4),
            ffn_hidden=trial.suggest_categorical('ffn_hidden', [128, 256, 512]),
            ffn_drop=trial.suggest_float('ffn_drop', 0.0, 0.4),
            lr=trial.suggest_float('lr', 1e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical('batch_size', [256, 512, 1024]))
        m = FTTransformerModel(n_features=nf, max_epochs=80,
                               patience=12, seed=seed, **p)
        m.fit(Xtr, ytr, Xvl, yvl)
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    bp = s.best_params
    if bp['d_token'] % bp['n_heads'] != 0: bp['n_heads'] = 4
    m = FTTransformerModel(n_features=nf, max_epochs=200,
                           patience=25, seed=seed, **bp)
    m.fit(Xtr, ytr, Xvl, yvl)
    return m, bp

# ============================================================
# ResNet (GPU + AMP)
# ============================================================
class ResNetModel:
    def __init__(self, n_features, hidden=256, n_layers=4, dropout=0.1,
                 lr=1e-3, batch_size=512, max_epochs=200, patience=20,
                 seed=42, device=DEVICE):
        self.cfg = dict(
            n_features=n_features, hidden=hidden, n_layers=n_layers,
            dropout=dropout, lr=lr, batch_size=batch_size,
            max_epochs=max_epochs, patience=patience, seed=seed)
        self.device = device
        self.model = None

    def _build(self):
        cfg = self.cfg
        torch.manual_seed(cfg['seed'])
        layers = [nn.Linear(cfg['n_features'], cfg['hidden']), nn.ReLU(), nn.Dropout(cfg['dropout'])]
        for _ in range(cfg['n_layers'] - 1):
            layers += [nn.Linear(cfg['hidden'], cfg['hidden']), nn.BatchNorm1d(cfg['hidden']),
                       nn.ReLU(), nn.Dropout(cfg['dropout'])]
        layers.append(nn.Linear(cfg['hidden'], 1))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, Xtr, ytr, Xvl, yvl):
        cfg = self.cfg
        torch.manual_seed(cfg['seed'])
        self.model = self._build()
        opt = torch.optim.AdamW(self.model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['max_epochs'])
        crit = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        Xtr_t = torch.FloatTensor(Xtr)
        ytr_t = torch.FloatTensor(ytr)
        Xvl_t = torch.FloatTensor(Xvl).to(self.device)
        yvl_t = torch.FloatTensor(yvl).to(self.device)
        loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t),
            batch_size=cfg['batch_size'], shuffle=True,
            num_workers=0, pin_memory=False)
        best_val, best_state, patience_cnt = float('inf'), None, 0
        for epoch in range(cfg['max_epochs']):
            self.model.train()
            for Xb, yb in loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    pred = self.model(Xb).squeeze(-1)
                    loss = crit(pred, yb)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            self.model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast():
                vl = crit(self.model(Xvl_t).squeeze(-1), yvl_t).item()
            sched.step()
            if vl < best_val:
                best_val = vl
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_cnt = 0
            else:
                patience_cnt += 1
            if patience_cnt >= cfg['patience']:
                break
        if best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.model(torch.FloatTensor(X).to(self.device)).squeeze(-1)
        return out.cpu().float().numpy()


def tune_resnet(Xtr, ytr, Xvl, yvl, seed):
    nf = Xtr.shape[1]
    def obj(trial):
        p = dict(
            hidden=trial.suggest_categorical('hidden', [128, 256, 512]),
            n_layers=trial.suggest_int('n_layers', 2, 6),
            dropout=trial.suggest_float('dropout', 0.0, 0.4),
            lr=trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            batch_size=trial.suggest_categorical('batch_size', [256, 512, 1024]))
        m = ResNetModel(n_features=nf, max_epochs=80, patience=12, seed=seed, **p)
        m.fit(Xtr, ytr, Xvl, yvl)
        return np.sqrt(mean_squared_error(yvl, m.predict(Xvl)))
    s = optuna.create_study(direction='minimize',
                            sampler=optuna.samplers.TPESampler(seed=seed))
    s.optimize(obj, n_trials=N_TRIALS)
    bp = s.best_params
    m = ResNetModel(n_features=nf, max_epochs=200, patience=25, seed=seed, **bp)
    m.fit(Xtr, ytr, Xvl, yvl)
    return m, bp


# ============================================================
# Run all models
# ============================================================
MODEL_FUNS = {
    'Ridge':          tune_ridge,
    'RandomForest':   tune_rf,
    'XGBoost':        tune_xgb,
    'LightGBM':       tune_lgb,
    'CatBoost':       tune_cat,
    'TabNet':         tune_tabnet,
    'FT-Transformer': tune_ftt,
    'ResNet':         tune_resnet,
}

# Load checkpoint if exists
ckpt = load_checkpoint()
if ckpt:
    all_results, all_preds, all_params, completed = ckpt
    completed = [tuple(c) for c in completed]
else:
    all_results = {name: {'RMSE': [], 'MAE': [], 'R2': [],
                          'train_s': [], 'inf_ms': []}
                   for name in MODEL_FUNS}
    all_preds  = {name: {} for name in MODEL_FUNS}
    all_params = {name: [] for name in MODEL_FUNS}
    completed  = []

final_models = {}

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print(f"{'='*60}")
    Xtr, Xvl, Xte, ytr, yvl, yte = prepare_data(seed)

    for name, fn in MODEL_FUNS.items():
        if (seed, name) in completed:
            print(f"  [{name}] SKIPPED (already done)", flush=True)
            if seed == 42:
                m = load_model(name, seed, n_features=Xtr.shape[1])
                if m is not None:
                    final_models[name] = m
            continue

        print(f"  [{name}] training...", end='', flush=True)
        t0 = time.time()
        model, params = fn(Xtr, ytr, Xvl, yvl, seed)
        elapsed = time.time() - t0

        # pytorch-tabnet returns 2D array for regression, flatten to 1D
        raw_pred = model.predict(Xte)
        ypred = raw_pred.flatten() if hasattr(raw_pred, 'flatten') else raw_pred
        def _predict_fn(X):
            p = model.predict(X)
            return p.flatten() if hasattr(p, 'flatten') else p
        it = inf_time(_predict_fn, Xte)

        rmse, mae, r2 = metrics(yte, ypred)
        all_results[name]['RMSE'].append(rmse)
        all_results[name]['MAE'].append(mae)
        all_results[name]['R2'].append(r2)
        all_results[name]['train_s'].append(elapsed)
        all_results[name]['inf_ms'].append(it)
        all_preds[name][str(seed)] = ypred.tolist()
        all_params[name].append(params)

        if seed == 42:
            final_models[name] = model

        print(f" RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} "
              f"t={elapsed:.1f}s inf={it:.4f}ms")

        # Save model to disk
        save_model(model, name, seed)

        # Save checkpoint (atomic write)
        completed.append((seed, name))
        save_checkpoint(all_results, all_preds, all_params, completed)

# ============================================================
# Save final results
# ============================================================
with open(f'{WORKDIR}/results/all_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
with open(f'{WORKDIR}/results/all_params.json', 'w') as f:
    json.dump(all_params, f, indent=2)
print("\nResults saved!")

# ============================================================
# Summary table
# ============================================================
print("\n" + "="*100)
print("FINAL RESULTS SUMMARY")
print("="*100)
rows = []
for name in MODEL_FUNS:
    r = all_results[name]
    row = {
        'Model':     name,
        'Type':      'DL' if name in ['ResNet', 'FT-Transformer'] else 'Classical',
        'RMSE_mean': np.mean(r['RMSE']), 'RMSE_std': np.std(r['RMSE']),
        'MAE_mean':  np.mean(r['MAE']),  'MAE_std':  np.std(r['MAE']),
        'R2_mean':   np.mean(r['R2']),   'R2_std':   np.std(r['R2']),
        'train_s':   np.mean(r['train_s']),
        'inf_ms':    np.mean(r['inf_ms']),
    }
    rows.append(row)
    print(f"{name:<20} RMSE={row['RMSE_mean']:.4f}±{row['RMSE_std']:.4f}  "
          f"MAE={row['MAE_mean']:.4f}±{row['MAE_std']:.4f}  "
          f"R2={row['R2_mean']:.4f}±{row['R2_std']:.4f}  "
          f"Train={row['train_s']:.1f}s  Inf={row['inf_ms']:.4f}ms")

summary_df = pd.DataFrame(rows)
summary_df.to_csv(f'{WORKDIR}/results/summary.csv', index=False)

# ============================================================
# Figures
# ============================================================
model_names = list(MODEL_FUNS.keys())
colors = ['#FF5722' if n in ['ResNet', 'FT-Transformer']
          else '#2196F3' for n in model_names]
from matplotlib.patches import Patch
leg = [Patch(facecolor='#2196F3', label='Classical'),
       Patch(facecolor='#FF5722', label='Deep Learning')]

# Fig 1: Performance comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, title, better in zip(
        axes, ['RMSE', 'MAE', 'R2'], ['RMSE', 'MAE', 'R²'],
        ['lower', 'lower', 'higher']):
    means = [np.mean(all_results[n][metric]) for n in model_names]
    stds  = [np.std(all_results[n][metric])  for n in model_names]
    idx   = np.argsort(means) if better == 'lower' else np.argsort(means)[::-1]
    snames = [model_names[i] for i in idx]
    smeans = [means[i] for i in idx]
    sstds  = [stds[i]  for i in idx]
    scols  = [colors[i] for i in idx]
    ax.barh(snames, smeans, xerr=sstds, color=scols, alpha=0.85,
            capsize=4, edgecolor='black', linewidth=0.5)
    ax.set_xlabel(f'{title} ({better} is better)', fontsize=11)
    ax.set_title(f'Test {title}\n(mean ± std, 3 seeds)', fontsize=12,
                 fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
fig.legend(handles=leg, loc='lower center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, -0.04))
plt.suptitle('Model Performance on California Housing (Regression)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORKDIR}/figures/performance_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: performance_comparison.png")

# Fig 2: Training time vs RMSE
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, xkey, xlabel in [
        (axes[0], 'train_s', 'Training Time (s)'),
        (axes[1], 'inf_ms',  'Inference Time (ms/sample)')]:
    for n, c in zip(model_names, colors):
        xv = np.mean(all_results[n][xkey])
        yv = np.mean(all_results[n]['RMSE'])
        ye = np.std(all_results[n]['RMSE'])
        mk = '^' if n in ['ResNet', 'FT-Transformer'] else 'o'
        ax.scatter(xv, yv, s=180, color=c, marker=mk, zorder=5,
                   edgecolors='black')
        ax.errorbar(xv, yv, yerr=ye, fmt='none', color=c, capsize=3)
        ax.annotate(n, (xv, yv), textcoords='offset points',
                    xytext=(6, 4), fontsize=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Test RMSE', fontsize=12)
    ax.set_title(f'{xlabel} vs RMSE', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
fig.legend(handles=leg, loc='lower center', ncol=2, fontsize=11,
           bbox_to_anchor=(0.5, -0.04))
plt.suptitle('Computational Cost vs Predictive Performance',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORKDIR}/figures/tradeoff.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: tradeoff.png")

# Fig 3: Sensitivity (CV)
fig, ax = plt.subplots(figsize=(10, 6))
cvs = [(n, np.std(all_results[n]['RMSE']) /
           np.mean(all_results[n]['RMSE']) * 100)
       for n in model_names]
cvs.sort(key=lambda x: x[1])
ns, vs = zip(*cvs)
cs = ['#FF5722' if n in ['TabNet', 'FT-Transformer']
      else '#2196F3' for n in ns]
ax.barh(list(ns), list(vs), color=cs, alpha=0.85,
        edgecolor='black', linewidth=0.5)
ax.set_xlabel('Coefficient of Variation (%) — lower = more stable', fontsize=12)
ax.set_title('Model Stability Across 3 Random Seeds', fontsize=13,
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.legend(handles=leg, fontsize=10)
plt.tight_layout()
plt.savefig(f'{WORKDIR}/figures/sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: sensitivity.png")

# Fig 4: Feature importance
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fi_models = [
    ('XGBoost',      final_models['XGBoost'].feature_importances_,      '#4CAF50'),
    ('LightGBM',     final_models['LightGBM'].feature_importances_ /
                     final_models['LightGBM'].feature_importances_.sum(), '#2196F3'),
    ('RandomForest', final_models['RandomForest'].feature_importances_,  '#9C27B0'),
    ('TabNet',       final_models['TabNet'].feature_importances_[0] if hasattr(final_models.get('TabNet', None), 'feature_importances_') else np.ones(len(feature_names))/len(feature_names),      '#FF5722'),
]
for ax, (name, imp, col) in zip(axes.flatten(), fi_models):
    idx = np.argsort(imp)
    ax.barh([feature_names[i] for i in idx], imp[idx], color=col, alpha=0.85,
            edgecolor='black', linewidth=0.5)
    ax.set_title(f'{name} Feature Importance', fontsize=12, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
plt.suptitle('Feature Importance Comparison (seed=42)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORKDIR}/figures/feature_importance.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: feature_importance.png")

# Fig 5: Predicted vs Actual
_, _, Xte42, _, _, yte42 = prepare_data(42)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
viz_names = ['XGBoost', 'LightGBM', 'FT-Transformer', 'ResNet']
viz_cols  = ['#4CAF50', '#2196F3', '#FF9800', '#FF5722']
for ax, nm, col in zip(axes.flatten(), viz_names, viz_cols):
    yp = final_models[nm].predict(Xte42)
    if hasattr(yp, 'flatten'): yp = yp.flatten()
    r, _, r2v = metrics(yte42, yp)
    ax.scatter(yte42, yp, alpha=0.25, s=5, color=col)
    lo = min(yte42.min(), yp.min()); hi = max(yte42.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Actual', fontsize=11)
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_title(f'{nm}\nRMSE={r:.4f}, R²={r2v:.4f}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.suptitle('Predicted vs Actual — California Housing (seed=42)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORKDIR}/figures/pred_vs_actual.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: pred_vs_actual.png")

# Fig 6: Residual distribution
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, nm, col in zip(axes.flatten(), viz_names, viz_cols):
    yp = final_models[nm].predict(Xte42)
    if hasattr(yp, 'flatten'): yp = yp.flatten()
    residuals = yte42 - yp
    ax.hist(residuals, bins=60, color=col, alpha=0.75,
            edgecolor='black', linewidth=0.3)
    ax.axvline(0, color='red', linestyle='--', lw=2)
    ax.set_xlabel('Residual (Actual - Predicted)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'{nm} Residuals\n'
                 f'mean={residuals.mean():.4f}, std={residuals.std():.4f}',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
plt.suptitle('Residual Distributions (seed=42)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{WORKDIR}/figures/residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: residuals.png")

print("\n" + "="*60)
print("ALL EXPERIMENTS COMPLETE!")
print(f"Results: {WORKDIR}/results/")
print(f"Figures: {WORKDIR}/figures/")
print(f"Models:  {WORKDIR}/models/")
print("="*60)
