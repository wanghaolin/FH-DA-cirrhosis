from functools import partial
import os
from lightgbm import LGBMClassifier
import numpy as np
import optuna
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import json
from datetime import datetime
from pathlib import Path


def load_dataset(file_path, target_col=-1):
    """
    从指定文件路径加载单个数据集
    :param file_path: 文件路径
    :param target_col: 目标变量列（默认最后一列）
    :return: (X, y) 元组
    """
    df = pd.read_csv(file_path)
    
    # 分离特征和目标变量
    if isinstance(target_col, str):
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = df.iloc[:, target_col]
        X = df.drop(df.columns[target_col], axis=1)
    
    return X, y


def load_cv_datasets(data_dir, dataset_name, num_folds=5):
    """
    加载五折交叉验证的训练集和测试集
    :param data_dir: 数据目录
    :param dataset_name: 数据集名称（如'PD'）
    :param num_folds: 折数，默认为5
    :return: (train_datasets, test_datasets) 元组，每个都是包含(num_folds)个(X, y)元组的列表
    """
    train_datasets = []
    test_datasets = []
    
    for fold in range(1, num_folds + 1):
        # 加载训练数据
        train_file = os.path.join(data_dir, dataset_name, f"{dataset_name}_final_train_fold_{fold}.csv")
        if os.path.exists(train_file):
            X_train, y_train = load_dataset(train_file)
            train_datasets.append((X_train, y_train))
        else:
            print(f"警告：训练文件不存在: {train_file}")
        
        # 加载测试数据
        test_file = os.path.join(data_dir, dataset_name, f"{dataset_name}_final_test_fold_{fold}.csv")
        if os.path.exists(test_file):
            X_test, y_test = load_dataset(test_file)
            test_datasets.append((X_test, y_test))
        else:
            print(f"警告：测试文件不存在: {test_file}")
    
    return train_datasets, test_datasets


def objective(trial, model_name, train_datasets, val_datasets):
    """
    使用预分割五折数据的目标函数
    :param trial: Optuna trial对象
    :param model_name: 模型名称
    :param train_datasets: 列表形式的训练数据集 (X_train, y_train)
    :param val_datasets: 列表形式的验证数据集 (X_val, y_val)
    :return: 五折平均AUC得分
    """
    # 创建模型
    if model_name == 'NB':
        model = GaussianNB(
            var_smoothing=trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)
        )
    elif model_name == 'KNN':
        algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
        leaf_size = 30
        if algorithm in ['ball_tree', 'kd_tree']:
            X_train, _ = train_datasets[0]
            max_samples = X_train.shape[0]
            max_leaf = min(50, max_samples)
            leaf_size = trial.suggest_int('leaf_size', 10, max_leaf)
        n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
        model = KNeighborsClassifier(
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_neighbors=n_neighbors,
            weights=trial.suggest_categorical('weights', ['uniform', 'distance']),
            p=trial.suggest_int('p', 1, 2),
            metric='minkowski'
        )
    elif model_name == 'LR':
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
        # 确保参数组合有效性
        if penalty == 'l1':
            solver = 'liblinear'
        else:
            solver = trial.suggest_categorical('solver', ['lbfgs', 'newton-cg', 'newton-cholesky', 'saga'])
        
        model = LogisticRegression(
            penalty=penalty,
            C=trial.suggest_float('C', 0.01, 10.0, log=True),
            solver=solver,
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            tol=trial.suggest_float('tol', 1e-6, 1e-2, log=True),
            max_iter=trial.suggest_int('max_iter', 50, 500),
            random_state=42
        )
    elif model_name == 'LDA':
        solver = trial.suggest_categorical('solver', ['svd', 'lsqr', 'eigen'])
        shrinkage = None
        tol = 1e-4
        
        if solver in ['lsqr', 'eigen']:
            shrinkage = trial.suggest_categorical('shrinkage', ['auto', None])
            if shrinkage != 'auto':
                shrinkage = trial.suggest_float('shrinkage_value', 0.01, 0.99) if shrinkage is None else shrinkage
            tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        
        model = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage,
            tol=tol
        )
    elif model_name == 'QDA':
        reg_param = trial.suggest_float('reg_param', 1e-3, 1.0, log=True)
        tol = trial.suggest_float('tol', 1e-6, 1e-2, log=True)
        model = QuadraticDiscriminantAnalysis(
            reg_param=reg_param,
            tol=tol,
            store_covariance=True
        )
    elif model_name == 'RF':
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_categorical('max_depth', [None] + list(range(3, 31)))
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        # 确保参数有效性
        if min_samples_leaf > min_samples_split // 2:
            raise optuna.TrialPruned(f"min_samples_leaf={min_samples_leaf} 不能超过 min_samples_split//2={min_samples_split//2}")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'ET':
        X_train, _ = train_datasets[0]
        n_samples, n_features = X_train.shape
        
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        max_depth = trial.suggest_int('max_depth', 3, min(20, int(np.log2(n_features) * 3 + 5)))
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, min(20, n_samples // 100))
        min_samples_split = trial.suggest_int('min_samples_split', min_samples_leaf + 2, min(30, n_samples // 50))
        max_features = trial.suggest_float('max_features', 0.1, 1.0, step=0.1)
        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
        
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            criterion=trial.suggest_categorical('criterion', ['gini', 'entropy']),
            random_state=42,
            n_jobs=-1
        )
    elif model_name == 'ADA':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
        model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
            algorithm=trial.suggest_categorical('algorithm', ['SAMME', 'SAMME.R'])
        )
    elif model_name == 'GBC':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        subsample = trial.suggest_float('subsample', 0.6, 1.0)
        
        model = GradientBoostingClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            subsample=subsample,
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            criterion='friedman_mse',
            random_state=42
        )
    elif model_name == 'LGBM':
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
        n_estimators = trial.suggest_int('n_estimators', 50, 500, step=50)
        num_leaves = trial.suggest_int('num_leaves', 10, 100)
        
        model = LGBMClassifier(
            class_weight=trial.suggest_categorical('class_weight', [None, 'balanced']),
            boosting_type=trial.suggest_categorical('boosting_type', ['gbdt', 'dart']),
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 1.0, log=True),
            random_state=42,
            force_col_wise=True,
            verbosity=-1
        )
    elif model_name == 'XGB':
        max_depth = trial.suggest_int('max_depth', 3, 10)
        eta = trial.suggest_float('eta', 0.01, 0.3, log=True)
        
        model = XGBClassifier(
            eta=eta,
            max_depth=max_depth,
            reg_alpha=trial.suggest_float('reg_alpha', 1e-4, 5.0, log=True),
            reg_lambda=trial.suggest_float('reg_lambda', 1e-4, 5.0, log=True),
            eval_metric='logloss',
            random_state=42,
            use_label_encoder=False
        )
    
    scores = []
    # 遍历每个预分割的fold
    for (X_train, y_train), (X_val, y_val) in zip(train_datasets, val_datasets):
        try:
            # 训练模型并计算AUC
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_proba)
            scores.append(score)
        except Exception as e:
            print(f"Fold 训练失败: {str(e)}")
            scores.append(0.0)  # 失败时给予最低分
    
    # 返回五折的平均得分
    return np.mean(scores)


def get_best_params(model_name, train_datasets, val_datasets, n_trials=50):
    """
    获取单个模型的最佳参数配置
    :param model_name: 模型名称
    :param train_datasets: 列表形式的训练数据集 (X_train, y_train)
    :param val_datasets: 列表形式的验证数据集 (X_val, y_val)
    :param n_trials: 试验次数
    :return: (best_params, best_score) 元组
    """
    objective_with_params = partial(
        objective, 
        model_name=model_name, 
        train_datasets=train_datasets, 
        val_datasets=val_datasets
    )
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42), pruner=HyperbandPruner())
    
    try:
        study.optimize(objective_with_params, n_trials=n_trials)
        
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid_trials:
            print(f"{model_name} 模型未找到有效试验")
            return None, 0.0
        
        best_params = study.best_params
        best_score = study.best_value
        
        print(f"{model_name} 最佳参数: {best_params}")
        print(f"{model_name} 最佳AUC得分: {best_score:.4f}")
        
        # 处理特殊参数情况（如LDA的shrinkage参数）
        if model_name == 'LDA' and 'shrinkage_value' in best_params:
            if best_params.get('shrinkage') == 'auto':
                best_params['shrinkage'] = 'auto'
                del best_params['shrinkage_value']
            else:
                best_params['shrinkage'] = best_params.pop('shrinkage_value')
        
        # 为所有模型添加random_state以确保可重复性
        if model_name != 'NB' and model_name != 'LDA' and model_name != 'QDA':
            best_params['random_state'] = 42
        
        return best_params, best_score
        
    except Exception as e:
        print(f"{model_name} 模型优化发生错误: {str(e)}")
        return None, 0.0


def create_best_params_dict(models, best_params_dict):
    """
    创建包含partial函数的最佳参数配置字典
    :param models: 模型名称到类的映射
    :param best_params_dict: 模型名称到最佳参数的映射
    :return: 包含partial函数的配置字典
    """
    result_dict = {}
    
    for model_name, model_class in models.items():
        if model_name in best_params_dict and best_params_dict[model_name] is not None:
            # 创建partial函数
            result_dict[model_name] = {
                'class': model_class.__name__,
                'params': best_params_dict[model_name]
            }
        else:
            # 如果没有找到最佳参数，使用默认配置
            result_dict[model_name] = {
                'class': model_class.__name__,
                'params': {}
            }
    
    return result_dict


def save_best_params_to_json(best_params_dict, output_file):
    """
    将最佳参数保存为JSON文件
    :param best_params_dict: 包含模型配置的字典
    :param output_file: 输出文件路径
    """
    # 转换为可JSON序列化的格式
    serializable_dict = {}
    
    for model_name, config in best_params_dict.items():
        # 转换numpy类型为Python原生类型
        serializable_params = {}
        for param_name, param_value in config['params'].items():
            if isinstance(param_value, np.generic):
                serializable_params[param_name] = param_value.item()
            else:
                serializable_params[param_name] = param_value
        
        serializable_dict[model_name] = {
            'class': config['class'],
            'params': serializable_params
        }
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        Path(output_dir).mkdir(exist_ok=True)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"最佳参数配置已保存至 {output_file}")
    
    # 同时打印出Python代码格式的配置，方便直接复制使用
    print("\nPython代码格式的最佳参数配置:")
    print("{")
    for model_name, config in best_params_dict.items():
        if model_name in best_params_dict and best_params_dict[model_name] is not None:
            params_str = ", ".join([f"{k}={repr(v)}" for k, v in config['params'].items()])
            print(f"    '{model_name}': partial({config['class']}, {params_str}),")
    print("}")


def main():
    # 定义异构的基分类器池
    models = {
        'NB': GaussianNB,
        'KNN': KNeighborsClassifier,
        'LR': LogisticRegression,
        'LDA': LinearDiscriminantAnalysis,
        'QDA': QuadraticDiscriminantAnalysis,
        'RF': RandomForestClassifier,
        'ET': ExtraTreesClassifier,
        'ADA': AdaBoostClassifier,
        'GBC': GradientBoostingClassifier,
        'LGBM': LGBMClassifier,
        'XGB': XGBClassifier
    }
    
    # 设置数据目录和数据集名称
    data_dir = 'filtered_data'  # 数据目录
    dataset_name = 'GY'  # 数据集名称，可以根据需要修改
    
    print(f"正在加载 {dataset_name} 数据集的五折交叉验证数据...")
    # 加载五折交叉验证数据
    train_datasets, test_datasets = load_cv_datasets(data_dir, dataset_name)
    
    if len(train_datasets) != 5 or len(test_datasets) != 5:
        print(f"警告：未找到完整的五折数据。找到 {len(train_datasets)} 个训练集和 {len(test_datasets)} 个测试集")
    
    # 为每个模型寻找最佳参数
    best_params_dict = {}
    best_scores = {}
    
    print("\n开始为每个模型寻找最佳参数配置...")
    for model_name in models.keys():
        print(f"\n正在优化 {model_name} 模型...")
        best_params, best_score = get_best_params(model_name, train_datasets, test_datasets)
        
        if best_params is not None:
            best_params_dict[model_name] = best_params
            best_scores[model_name] = best_score
    
    # 显示所有模型的最佳得分
    print("\n所有模型的最佳AUC得分:")
    for model_name, score in sorted(best_scores.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {score:.4f}")
    
    # 创建包含partial函数的配置字典
    result_dict = create_best_params_dict(models, best_params_dict)
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join('best_params/GY_baselines', dataset_name, f"best_params_{timestamp}.json")
    
    # 保存最佳参数配置
    save_best_params_to_json(result_dict, output_file)


if __name__ == "__main__":
    main()