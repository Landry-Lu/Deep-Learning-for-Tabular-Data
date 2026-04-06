import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
import joblib
import torch
from src.preprocess import *
from src.train import *
from src.tune import tune_model
from src.utils import set_seed, save_results

def run_experiment(dataset_name='adult', data_path='data/raw/adult.data', seeds=[42, 123, 2024]):
    X, y = load_adult_data(data_path)
    num_cols, cat_cols = get_feature_types(X)

    all_results = []
    for seed in seeds:
        print(f"\n========== Seed {seed} ==========")
        set_seed(seed)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y, random_state=seed)

        # Tree preprocessing
        X_train_tree, X_val_tree, X_test_tree, _ = preprocess_for_tree(X_train, X_val, X_test, cat_cols)
        # Linear preprocessing
        X_train_lr, X_val_lr, X_test_lr, _, _ = preprocess_for_linear(X_train, X_val, X_test, num_cols, cat_cols)
        # DL preprocessing
        (X_num_train, X_cat_train), (X_num_val, X_cat_val), (X_num_test, X_cat_test), _, cat_cardinalities = preprocess_for_dl(
            X_train, X_val, X_test, num_cols, cat_cols)

        models = ['xgboost', 'lightgbm', 'logistic', 'tabnet', 'ft_transformer']
        for model_name in models:
            print(f"  Running {model_name}...")
            if model_name in ['xgboost', 'lightgbm', 'logistic', 'tabnet']:
                # 调参次数改为 2 以便快速测试（正式运行时改为 30）
                best_params = tune_model(model_name, (X_train_tree, y_train), (X_val_tree, y_val), n_trials=2)
                if model_name == 'xgboost':
                    model = train_xgboost(X_train_tree, y_train, X_val_tree, y_val, best_params, seed)
                    X_test_model = X_test_tree
                elif model_name == 'lightgbm':
                    model = train_lightgbm(X_train_tree, y_train, X_val_tree, y_val, best_params, seed)
                    X_test_model = X_test_tree
                elif model_name == 'logistic':
                    model = train_logistic(X_train_lr, y_train, X_val_lr, y_val, best_params, seed)
                    X_test_model = X_test_lr
                elif model_name == 'tabnet':
                    model = train_tabnet(X_train_tree, y_train, X_val_tree, y_val, best_params, seed)
                    X_test_model = X_test_tree
            else:  # ft_transformer
                best_params = tune_model(model_name, ((X_num_train, X_cat_train), y_train),
                                         ((X_num_val, X_cat_val), y_val),
                                         cat_cardinalities=cat_cardinalities, n_trials=2)
                model = train_ft_transformer(X_num_train, X_cat_train, y_train,
                                             X_num_val, X_cat_val, y_val,
                                             cat_cardinalities, best_params, seed)
                X_test_model = (X_num_test, X_cat_test)

            eval_metrics = evaluate_model(model, X_test_model, y_test, model_name, cat_cardinalities, X_cat_test if model_name=='ft_transformer' else None)
            eval_metrics['model'] = model_name
            eval_metrics['seed'] = seed
            eval_metrics['best_params'] = best_params
            all_results.append(eval_metrics)

            os.makedirs(f'results/models/{dataset_name}/seed{seed}', exist_ok=True)
            if model_name == 'tabnet':
                model.save_model(f'results/models/{dataset_name}/seed{seed}/{model_name}.zip')
            elif model_name == 'ft_transformer':
                torch.save(model.state_dict(), f'results/models/{dataset_name}/seed{seed}/{model_name}.pt')
            else:
                joblib.dump(model, f'results/models/{dataset_name}/seed{seed}/{model_name}.pkl')

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(f'results/{dataset_name}_results.csv', index=False)
    print("\nAll experiments completed. Results saved to results/")
    return results_df

if __name__ == '__main__':
    run_experiment()
