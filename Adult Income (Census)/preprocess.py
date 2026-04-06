import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

def load_adult_data(data_path='data/raw/adult.data'):
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    df = pd.read_csv(data_path, names=column_names, na_values='?', skipinitialspace=True)
    df = df.dropna().reset_index(drop=True)
    y = (df['income'] == '>50K').astype(int)
    X = df.drop('income', axis=1)
    return X, y

def get_feature_types(X):
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    return num_cols, cat_cols

def split_data(X, y, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )
    val_relative = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative, random_state=random_state, stratify=y_temp
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_for_tree(X_train, X_val, X_test, cat_cols):
    X_train_tree = X_train.copy()
    X_val_tree = X_val.copy()
    X_test_tree = X_test.copy()
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train_tree[col], X_val_tree[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_tree[col] = le.transform(X_train_tree[col].astype(str))
        X_val_tree[col] = le.transform(X_val_tree[col].astype(str))
        X_test_tree[col] = le.transform(X_test_tree[col].astype(str))
        le_dict[col] = le
    return X_train_tree, X_val_tree, X_test_tree, le_dict

def preprocess_for_linear(X_train, X_val, X_test, num_cols, cat_cols):
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_val_num = num_imputer.transform(X_val[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = scaler.transform(X_val_num)
    X_test_num = scaler.transform(X_test_num)

    from sklearn.preprocessing import OneHotEncoder
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = ohe.fit_transform(X_train[cat_cols].astype(str))
    X_val_cat = ohe.transform(X_val[cat_cols].astype(str))
    X_test_cat = ohe.transform(X_test[cat_cols].astype(str))

    X_train_lr = np.hstack([X_train_num, X_train_cat])
    X_val_lr = np.hstack([X_val_num, X_val_cat])
    X_test_lr = np.hstack([X_test_num, X_test_cat])
    return X_train_lr, X_val_lr, X_test_lr, scaler, ohe

def preprocess_for_dl(X_train, X_val, X_test, num_cols, cat_cols):
    num_imputer = SimpleImputer(strategy='median')
    X_train_num = num_imputer.fit_transform(X_train[num_cols])
    X_val_num = num_imputer.transform(X_val[num_cols])
    X_test_num = num_imputer.transform(X_test[num_cols])
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train_num)
    X_val_num = scaler.transform(X_val_num)
    X_test_num = scaler.transform(X_test_num)

    cat_encoders = {}
    cat_cardinalities = []
    X_train_cat = []
    X_val_cat = []
    X_test_cat = []
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_val[col]], axis=0).astype(str)
        le.fit(combined)
        X_train_cat.append(le.transform(X_train[col].astype(str)))
        X_val_cat.append(le.transform(X_val[col].astype(str)))
        X_test_cat.append(le.transform(X_test[col].astype(str)))
        cat_encoders[col] = le
        cat_cardinalities.append(len(le.classes_))
    X_train_cat = np.column_stack(X_train_cat) if cat_cols else np.empty((len(X_train), 0))
    X_val_cat = np.column_stack(X_val_cat) if cat_cols else np.empty((len(X_val), 0))
    X_test_cat = np.column_stack(X_test_cat) if cat_cols else np.empty((len(X_test), 0))
    return (X_train_num, X_train_cat), (X_val_num, X_val_cat), (X_test_num, X_test_cat), scaler, cat_cardinalities
