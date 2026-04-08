import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import classification_report, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix, roc_curve
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from functools import partial
from imblearn.metrics import geometric_mean_score
import os
import pandas as pd
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score, recall_score, precision_score

# 新增依赖库，确保后续代码正常运行
from functools import partial
from imblearn.metrics import geometric_mean_score
import numpy as np

def optimize_threshold(y_test, y_pred_proba, method='f1'):
    """
    根据y_test和预测概率计算最佳阈值
    支持多种阈值选择方法：
    - 'youden': ROC曲线约登指数最大化
    - 'f1': PR曲线F1分数最大化
    - 'precision': 固定 precision 为0.8时的阈值
    - 'recall': 固定 recall 为0.8时的阈值
    - 'gmean': 几何均值最大化(对不平衡数据更稳健)
    - 'f1': PR曲线F1分数最大化
    - 'youden': ROC曲线约登指数最大化
    
    :param y_test: 真实标签
    :param y_pred_proba: 预测概率
    :param method: 阈值选择方法
    :return: 最佳阈值
    """
    if method == 'youden':
        # 传统约登指数法(ROC曲线)
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        y = tpr - fpr
        optimal_idx = np.argmax(y)
        return thresholds[optimal_idx]
    elif method == 'f1':
        # PR曲线F1分数最大化(对不平衡数据更有效)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)  # 避免除零
        optimal_idx = np.argmax(f1_scores[:-1])  # 最后一个阈值对应recall=0，需排除
        return thresholds[optimal_idx]
    elif method == 'precision':
        # 固定precision为0.8时的阈值
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        # 找到最接近0.8的precision值
        target_precision = 0.8
        optimal_idx = np.argmin(np.abs(precision - target_precision))
        return thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    elif method == 'recall':
        # 固定recall为0.8时的阈值
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        target_recall = 0.8
        # 从后往前找第一个大于等于目标recall的阈值
        for i in range(len(recall)-2, -1, -1):
            if recall[i] >= target_recall:
                return thresholds[i]
        return 0.5  # 默认阈值
    elif method == 'gridsearch':
        # 自适应步长网格搜索寻找最佳阈值
        # 初始粗搜索，但增加更多性能指标的综合评估
        thresholds = np.arange(0.01, 1.0, 0.1)
        best_score = -1
        best_threshold = 0.5
        
        # 计算数据不平衡率
        class_counts = np.bincount(y_test)
        imbalance_ratio = max(class_counts) / min(class_counts) if len(class_counts) > 1 else 1
        
        # 根据不平衡率调整权重
        if imbalance_ratio > 5:
            # 高不平衡数据更重视gmean和recall
            weights = {'gmean': 0.4, 'balanced_accuracy': 0.3, 'f1': 0.2, 'recall': 0.1}
        else:
            # 平衡数据更重视f1和balanced_accuracy
            weights = {'f1': 0.4, 'balanced_accuracy': 0.3, 'gmean': 0.2, 'precision': 0.1}
        
        # 第一次粗搜索
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # 计算各项指标
            current_gmean = geometric_mean_score(y_test, y_pred)
            current_bal_acc = balanced_accuracy_score(y_test, y_pred)
            current_f1 = f1_score(y_test, y_pred)
            current_recall = recall_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) > 1 else 1
            current_precision = precision_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) > 1 else 1
            
            # 加权综合评分
            current_score = (weights['gmean'] * current_gmean +
                            weights['balanced_accuracy'] * current_bal_acc +
                            weights['f1'] * current_f1 +
                            (weights.get('recall', 0) * current_recall if imbalance_ratio > 5 else weights.get('precision', 0) * current_precision))
            
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
        
        # 第二次精细搜索(最佳阈值周围±0.15范围，步长0.01)
        start = max(0.01, best_threshold - 0.15)
        end = min(0.99, best_threshold + 0.15)
        fine_thresholds = np.arange(start, end, 0.01)
        
        for threshold in fine_thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            current_gmean = geometric_mean_score(y_test, y_pred)
            current_bal_acc = balanced_accuracy_score(y_test, y_pred)
            current_f1 = f1_score(y_test, y_pred)
            current_recall = recall_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) > 1 else 1
            current_precision = precision_score(y_test, y_pred, zero_division=0) if len(np.unique(y_test)) > 1 else 1
            
            current_score = (weights['gmean'] * current_gmean +
                            weights['balanced_accuracy'] * current_bal_acc +
                            weights['f1'] * current_f1 +
                            (weights.get('recall', 0) * current_recall if imbalance_ratio > 5 else weights.get('precision', 0) * current_precision))
            
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
            
        return best_threshold
    else:
        raise ValueError(f"不支持的阈值选择方法: {method}")

def load_datasets_from_folder(folder_path, target_col=-1, file_extension=".csv", prefix=""):
    """
    从指定文件夹加载所有CSV数据集并返回完整数据集列表
    :param folder_path: 包含数据集的文件夹路径
    :param target_col: 目标变量列（默认最后一列）
    :return: 数据集列表，每个元素为 (X, y) 元组
    """
    datasets = []
    
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(file_extension):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            
            # 分离特征和目标变量
            if isinstance(target_col, str):
                y = df[target_col].values  # 转换为numpy数组
                X = df.drop(columns=[target_col]).values  # 转换为numpy数组
            else:
                y = df.iloc[:, target_col].values  # 转换为numpy数组
                X = df.drop(df.columns[target_col], axis=1).values  # 转换为numpy数组
            
            datasets.append((X, y))
    
    return datasets

def evaluate_models(train_datasets, test_datasets, models):
    """
    对多个数据集进行模型训练与评估
    :param train_datasets: 列表或单个数据集元组，每个元素为 (X_train, y_train)
    :param test_datasets: 列表或单个数据集元组，每个元素为 (X_test, y_test)
    :param models: 模型字典，格式如 { '模型缩写': 模型类 }
    :return: 评估结果字典
    """

    # 确保参数是列表格式
    if not isinstance(train_datasets, list):
        train_datasets = [train_datasets]
    if not isinstance(test_datasets, list):
        test_datasets = [test_datasets]
    
    # 自动适配测试集数量
    if len(test_datasets) == 1 and len(train_datasets) > 1:
        # 当测试集只有一个时，扩展为与训练集数量一致
        test_datasets = test_datasets * len(train_datasets)
    
    # 校验数据集数量一致性
    if len(train_datasets) != len(test_datasets):
        raise ValueError("训练集和测试集数量不匹配，请确保两者长度一致")
    
    results = {model_name: [] for model_name in models}
    
    for dataset_idx in range(len(train_datasets)):
        X_train, y_train = train_datasets[dataset_idx]
        X_test, y_test = test_datasets[dataset_idx]
        
        for model_name, model_cls in models.items():
            try:

                model = model_cls()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

                optimal_threshold = optimize_threshold(y_test, y_proba, method='youden')
                # print(f'{model_name}的阈值点为：',optimal_threshold)
                y_pred = (y_proba >= optimal_threshold).astype(int)

                # 计算各项评估指标
                f1 = f1_score(y_test, y_pred)
                mcc = matthews_corrcoef(y_test, y_pred)
                gmean = geometric_mean_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
                
                # 新增指标：准确率
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                # 新增指标：正样本的召回率（recall for positive class）
                recall_pos = recall_score(y_test, y_pred, pos_label=1, zero_division=0) if len(np.unique(y_test)) > 1 else 1
                # 新增指标：正样本的精准率（precision for positive class）
                precision_pos = precision_score(y_test, y_pred, pos_label=1, zero_division=0) if len(np.unique(y_test)) > 1 else 1
                # 新增指标：PR-AUC
                pr_auc = 0.0
                if y_proba is not None and len(np.unique(y_test)) > 1:
                    precision, recall, _ = precision_recall_curve(y_test, y_proba)
                    from sklearn.metrics import auc as sklearn_auc
                    pr_auc = sklearn_auc(recall, precision)

                # 打印混淆矩阵
                print(f"混淆矩阵 (数据集{dataset_idx + 1}, 模型{model_name}):")
                print(confusion_matrix(y_test, y_pred))
                print(classification_report(y_test, y_pred))
                print(f"PR-AUC (数据集{dataset_idx + 1}, 模型{model_name}): {pr_auc:.4f}")

                
                results[model_name].append({
                    'F1': f1,
                    'AUC': auc,
                    'MCC': mcc,
                    'GMean': gmean,
                    'balanced_accuracy': balanced_accuracy,
                    'Recall_pos': recall_pos,
                    'Precision_pos': precision_pos,
                    'PR_AUC': pr_auc
                })

            except Exception as e:
                print(f"模型 {model_name} 在数据集{dataset_idx}训练时出错: {str(e)}")
                results[model_name].append({
                    'F1': np.nan,
                    'AUC': np.nan,
                    'MCC': np.nan,
                    'GMean': np.nan,
                    'balanced_accuracy': np.nan,
                    'Recall_pos': np.nan,
                    'Precision_pos': np.nan,
                    'PR_AUC': np.nan
                })
    
    # 打印详细结果
    print(f"\n{'='*30} 详细评估结果 {'='*30}")
    for dataset_idx in range(len(train_datasets)):
        print(f"\n数据集 {dataset_idx + 1}:")
        print(f"{'模型名称':<8} | F1-score | AUC     | MCC     | GMean   | balanced_accuracy | Recall_pos | Precision_pos | PR-AUC")
        print("-" * 100)
        for model_name in models:
            model_data = results[model_name]
            if dataset_idx < len(model_data):
                result = model_data[dataset_idx]
                # 先处理结果值
                f1_val = result['F1']
                auc_val = result['AUC']
                mcc_val = result['MCC']
                gmean_val = result['GMean']
                balanced_accuracy_val = result['balanced_accuracy']
                recall_pos_val = result['Recall_pos']
                precision_pos_val = result['Precision_pos']
                pr_auc_val = result['PR_AUC']
                
                # 转换为适当的字符串格式
                f1_str = 'N/A' if np.isnan(f1_val) else f"{f1_val:.4f}"
                auc_str = 'N/A' if np.isnan(auc_val) else f"{auc_val:.4f}"
                mcc_str = 'N/A' if np.isnan(mcc_val) else f"{mcc_val:.4f}"
                gmean_str = 'N/A' if np.isnan(gmean_val) else f"{gmean_val:.4f}"
                balanced_accuracy_str = 'N/A' if np.isnan(balanced_accuracy_val) else f"{balanced_accuracy_val:.4f}"
                recall_pos_str = 'N/A' if np.isnan(recall_pos_val) else f"{recall_pos_val:.4f}"
                precision_pos_str = 'N/A' if np.isnan(precision_pos_val) else f"{precision_pos_val:.4f}"
                pr_auc_str = 'N/A' if np.isnan(pr_auc_val) else f"{pr_auc_val:.4f}"
                
                # 打印结果
                print(f"{model_name:<8} | {f1_str:<8} | {auc_str:<8} | {mcc_str:<8} | {gmean_str:<8} | {balanced_accuracy_str:<9} | {recall_pos_str:<11} | {precision_pos_str:<13} | {pr_auc_str:<8}")
            else:
                print(f"{model_name:<8} | 数据不足")

    return results


if __name__ == "__main__":
    # # 定义模型字典(没有调参)
    # models = {
    #     'NB': GaussianNB,
    #     'KNN': KNeighborsClassifier,
    #     'LR': LogisticRegression,
    #     'LDA': LinearDiscriminantAnalysis,
    #     'QDA': QuadraticDiscriminantAnalysis,
    #     'RF': RandomForestClassifier,
    #     'ET': ExtraTreesClassifier,
    #     'ADA': AdaBoostClassifier,
    #     'GBC': GradientBoostingClassifier,
    #     'LGBM': LGBMClassifier,
    #     'XGB': XGBClassifier,
    # }

    # # 定义模型字典(手动调参)
    # models = {
    #     'NB': GaussianNB,
    #     'KNN': partial(KNeighborsClassifier, weights='distance',  n_neighbors=3, algorithm='ball_tree'),
    #     'LR': partial(LogisticRegression, class_weight='balanced', penalty='l1', solver='liblinear', C=0.3, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='svd',tol=1e-4),
    #     'QDA': partial(QuadraticDiscriminantAnalysis,reg_param=0.2,store_covariance=True),
    #     'RF': partial(RandomForestClassifier,class_weight='balanced',
    #                 max_depth=10, min_samples_leaf=10, max_features='sqrt',
    #                 n_estimators=300, random_state=42),
    #     'ET': partial(ExtraTreesClassifier,class_weight='balanced',max_depth=10,bootstrap=True,max_samples=0.6, random_state=42),
    #     'ADA': partial(AdaBoostClassifier, n_estimators=300,learning_rate=0.05, random_state=42),
    #     'GBC': partial(GradientBoostingClassifier,max_depth=5,learning_rate=0.1,n_iter_no_change=10, random_state=42),
    #     'LGBM': partial(LGBMClassifier,class_weight='balanced',boosting_type='dart',
    #                 num_leaves=31,reg_alpha=0.1,reg_lambda=0.1, force_col_wise=True, verbosity= -1, random_state=42),
    #     'XGB': partial(XGBClassifier,max_delta_step=1,reg_alpha=0.1,reg_lambda=0.5,
    #                 eta=0.07,eval_metric='logloss', random_state=42)
    #                     }

    # # HE-定义模型字典（自动调参）
    # models = {
    #     'NB': partial(GaussianNB, var_smoothing=9.891364113904856e-07),
    #     'KNN': partial(KNeighborsClassifier, algorithm='auto', n_neighbors=30, weights='distance', p=2),
    #     'LR': partial(LogisticRegression, penalty='l2', solver='lbfgs', C=1.2590047620987799, class_weight=None, tol=0.0001521268430562923, max_iter=414, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
    #     'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.176622018898312, tol=0.00011947264239668641),
    #     'RF': partial(RandomForestClassifier, n_estimators=500, max_depth=30, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', criterion='entropy', class_weight='balanced', random_state=42),
    #     'ET': partial(ExtraTreesClassifier, n_estimators=300, max_depth=16, min_samples_leaf=2, min_samples_split=28, max_features=0.1, bootstrap=False, class_weight=None, criterion='gini', random_state=42),
    #     'ADA': partial(AdaBoostClassifier, learning_rate=0.05601888314861266, n_estimators=200, algorithm='SAMME.R', random_state=42),
    #     'GBC': partial(GradientBoostingClassifier, learning_rate=0.01866217122968749, n_estimators=100, max_depth=5, max_features='log2', subsample=0.669171223494975, min_samples_split=3, min_samples_leaf=10, random_state=42),
    #     'LGBM': partial(LGBMClassifier, learning_rate=0.03924051936374861, n_estimators=450, num_leaves=10, class_weight='balanced', boosting_type='dart', reg_alpha=0.2782798249640321, reg_lambda=0.13171133143615316, random_state=42),
    #     'XGB': partial(XGBClassifier, max_depth=4, eta=0.07357270835293087, reg_alpha=1.4149633530419305, reg_lambda=0.08117306537418037, random_state=42),   
    # }

    # # PD-自动调参
    # models = {
    #     'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
    #     'KNN': partial(KNeighborsClassifier, algorithm='ball_tree', leaf_size=19, n_neighbors=30, weights='distance', p=1),
    #     'LR': partial(LogisticRegression, penalty='l2', solver='newton-cholesky', C=7.953052660574928, class_weight=None, tol=0.0013519206467406972, max_iter=217, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
    #     'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.20659869635343953, tol=0.00015393640132252336),
    #     'RF': partial(RandomForestClassifier, n_estimators=450, max_depth=5, min_samples_split=15, min_samples_leaf=4, max_features=None, criterion='entropy', class_weight='balanced', random_state=42),        
    #     'ET': partial(ExtraTreesClassifier, n_estimators=250, max_depth=5, min_samples_leaf=10, min_samples_split=24, max_features=1.0, bootstrap=False, class_weight='balanced', criterion='entropy', random_state=42),
    #     'ADA': partial(AdaBoostClassifier, learning_rate=0.021438723017596622, n_estimators=100, algorithm='SAMME.R', random_state=42),
    #     'GBC': partial(GradientBoostingClassifier, learning_rate=0.022782565517931837, n_estimators=300, max_depth=3, max_features='log2', subsample=0.6584846570294198, min_samples_split=6, min_samples_leaf=2, random_state=42),
    #     'LGBM': partial(LGBMClassifier, learning_rate=0.02541885003050532, n_estimators=350, num_leaves=35, class_weight='balanced', boosting_type='dart', reg_alpha=0.0014881309479933367, reg_lambda=0.9808097775040516, random_state=42),
    #     'XGB': partial(XGBClassifier, max_depth=3, eta=0.026304768011494003, reg_alpha=0.3693374576085647, reg_lambda=0.7993370406717103, random_state=42),
    # }

    # # FMY-自动调参
    # models = {
    #     'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
    #     'KNN': partial(KNeighborsClassifier, algorithm='brute', n_neighbors=25, weights='uniform', p=2),
    #     'LR': partial(LogisticRegression, penalty='l2', solver='newton-cholesky', C=0.13885529354224246, class_weight=None, tol=3.4050976484025e-06, max_iter=418, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='eigen', shrinkage=0.10007088357244945, tol=4.810651489559353e-05),
    #     'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.1027153683243511, tol=0.008110840524533395),
    #     'RF': partial(RandomForestClassifier, n_estimators=100, max_depth=12, min_samples_split=18, min_samples_leaf=9, max_features='sqrt', criterion='gini', class_weight=None, random_state=42),      
    #     'ET': partial(ExtraTreesClassifier, n_estimators=250, max_depth=14, min_samples_leaf=15, min_samples_split=17, max_features=0.4, bootstrap=False, class_weight=None, criterion='entropy', random_state=42),
    #     'ADA': partial(AdaBoostClassifier, learning_rate=0.06925533246864075, n_estimators=100, algorithm='SAMME.R', random_state=42),
    #     'GBC': partial(GradientBoostingClassifier, learning_rate=0.01725836141076542, n_estimators=500, max_depth=3, max_features='sqrt', subsample=0.9217136762887744, min_samples_split=4, min_samples_leaf=2, random_state=42),
    #     'LGBM': partial(LGBMClassifier, learning_rate=0.11601594359211267, n_estimators=50, num_leaves=13, class_weight=None, boosting_type='dart', reg_alpha=0.005920840603924326, reg_lambda=0.5851266319185591, random_state=42),
    #     'XGB': partial(XGBClassifier, max_depth=3, eta=0.12178995038725061, reg_alpha=0.15121284088446904, reg_lambda=0.0004746487660041205, random_state=42),
    # }

    # # GY-自动调参
    # models = {
    #     'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
    #     'KNN': partial(KNeighborsClassifier, algorithm='kd_tree', leaf_size=37, n_neighbors=28, weights='distance', p=2),
    #     'LR': partial(LogisticRegression, penalty='l2', C=0.43736593268991864, class_weight=None, tol=0.0010276978799463987, max_iter=281, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
    #     'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.4418374353523307, tol=1.208886038766324e-05),
    #     'RF': partial(RandomForestClassifier, n_estimators=150, max_depth=3, min_samples_split=15, min_samples_leaf=3, max_features='sqrt', criterion='entropy', class_weight=None, random_state=42),    
    #     'ET': partial(ExtraTreesClassifier, n_estimators=100, max_depth=9, min_samples_leaf=3, min_samples_split=9, max_features=0.2, bootstrap=False, class_weight=None, criterion='entropy', random_state=42),
    #     'ADA': partial(AdaBoostClassifier, learning_rate=0.05159403035135414, n_estimators=50, algorithm='SAMME.R', random_state=42),
    #     'GBC': partial(GradientBoostingClassifier, learning_rate=0.013395058503939341, n_estimators=300, max_depth=4, max_features='log2', subsample=0.7847427943496508, min_samples_split=11, min_samples_leaf=5, random_state=42),
    #     'LGBM': partial(LGBMClassifier, learning_rate=0.1251033015766346, n_estimators=50, num_leaves=10, class_weight='balanced', boosting_type='dart', reg_alpha=0.03326964102432317, reg_lambda=0.05510519784818472, random_state=42),
    #     'XGB': partial(XGBClassifier, max_depth=3, eta=0.028875929928602655, reg_alpha=0.0008183036138409682, reg_lambda=0.0007539166440950076, random_state=42),
    # }

    # # ASCITES-自动调参
    # models = {
    #     'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
    #     'KNN': partial(KNeighborsClassifier, algorithm='kd_tree', leaf_size=46, n_neighbors=26, weights='distance', p=1),
    #     'LR': partial(LogisticRegression, C=5.015953330208218, class_weight='balanced', tol=1.9321178301160835e-06, max_iter=470, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
    #     'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.11293502322255572, tol=3.124782808031647e-05),
    #     'RF': partial(RandomForestClassifier, n_estimators=450, max_depth=14, min_samples_split=15, min_samples_leaf=2, max_features='log2', criterion='gini', class_weight=None, random_state=42),
    #     'ET': partial(ExtraTreesClassifier, n_estimators=150, max_depth=12, min_samples_leaf=6, min_samples_split=15, max_features=0.4, bootstrap=True, class_weight=None, criterion='gini', random_state=42),
    #     'ADA': partial(AdaBoostClassifier, learning_rate=0.018150872220609515, n_estimators=50, algorithm='SAMME.R', random_state=42),
    #     'GBC': partial(GradientBoostingClassifier, learning_rate=0.016336878261839136, n_estimators=100, max_depth=9, max_features='sqrt', subsample=0.9511458511922435, min_samples_split=10, min_samples_leaf=8, random_state=42),      
    #     'LGBM': partial(LGBMClassifier, learning_rate=0.13050864504791743, n_estimators=50, num_leaves=70, class_weight=None, boosting_type='gbdt', reg_alpha=0.0010551359020055315, reg_lambda=0.12702735096105958, random_state=42),    
    #     'XGB': partial(XGBClassifier, max_depth=4, eta=0.016340849561517237, reg_alpha=0.006237416107648356, reg_lambda=0.758995846338095, random_state=42),
    # }

    # ACLF-自动调参
    models = {
        'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
        'KNN': partial(KNeighborsClassifier, algorithm='auto', n_neighbors=30, weights='uniform', p=2),
        'LR': partial(LogisticRegression, penalty='l2', solver='newton-cg', C=0.09463470489960597, class_weight='balanced', tol=0.00020193497214808254, max_iter=278, random_state=42),
        'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
        'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.6557768306823545, tol=0.002047147040324709),
        'RF': partial(RandomForestClassifier, n_estimators=350, max_depth=10, min_samples_split=19, min_samples_leaf=3, max_features='sqrt', criterion='entropy', class_weight=None, random_state=42),
        'ET': partial(ExtraTreesClassifier, n_estimators=100, max_depth=10, min_samples_leaf=5, min_samples_split=9, max_features=0.30000000000000004, bootstrap=False, class_weight=None, criterion='gini', random_state=42),
        'ADA': partial(AdaBoostClassifier, learning_rate=0.6326195773451885, n_estimators=250, algorithm='SAMME', random_state=42),
        'GBC': partial(GradientBoostingClassifier, learning_rate=0.013119940157366297, n_estimators=300, max_depth=4, max_features='log2', subsample=0.6955857826691667, min_samples_split=4, min_samples_leaf=9, random_state=42),
        'LGBM': partial(LGBMClassifier, learning_rate=0.047584916773550485, n_estimators=100, num_leaves=10, class_weight=None, boosting_type='gbdt', reg_alpha=0.016498520716057563, reg_lambda=0.008638475783333249, random_state=42),
        'XGB': partial(XGBClassifier, max_depth=4, eta=0.07626889104749136, reg_alpha=0.05651492046926741, reg_lambda=2.6046398111143314, random_state=42),
    }

    # 新增五折循环处理
    all_folds_results = {model_name: [] for model_name in models}
    for fold in range(1, 6):
        # filtered_data/HE   filtered_data/HE/enhance：HE_enhanced_gaussian_copula_
        train_folder_path = f'filtered_data/ACLF'  # 五折训练集路径
        train_datasets = load_datasets_from_folder(train_folder_path, target_col=-1, file_extension=".csv", prefix=f"ACLF_final_train_fold_{fold}")

        # # _disease_attention_pca
        # test_folder_path = f'filtered_data/ASCITES'  # 对应测试集路径
        # test_datasets = load_datasets_from_folder(test_folder_path, target_col=-1, file_extension=".csv", prefix=f"ASCITES_final_test_fold_{fold}")

        # 评估外部验证集
        external_folder_path = f'filtered_externel_data'  # 外部验证集路径
        external_datasets = load_datasets_from_folder(external_folder_path, target_col=-1, file_extension=".csv", prefix=f"ACLF_external_preprocessed_")

        
        # 调用评估并将结果累积
        # fold_results = evaluate_models(train_datasets, test_datasets, models)
        fold_results = evaluate_models(train_datasets, external_datasets, models)

        for model_name in models:
            all_folds_results[model_name].extend(fold_results[model_name])
    
    # 计算最终平均结果
    print(f"\n{'='*30} 最终平均评估结果 {'='*30}")
    print(f"{'模型名称':<8} | F1-score       | AUC           | MCC           | GMean         | balanced_accuracy | Recall_pos    | Precision_pos | PR-AUC")
    print("-" * 130)
    for model_name in models:
        model_data = all_folds_results[model_name]
        avg_f1 = np.nanmean([d['F1'] for d in model_data])
        std_f1 = np.nanstd([d['F1'] for d in model_data])
        avg_auc = np.nanmean([d['AUC'] for d in model_data])
        std_auc = np.nanstd([d['AUC'] for d in model_data])
        avg_mcc = np.nanmean([d['MCC'] for d in model_data])
        std_mcc = np.nanstd([d['MCC'] for d in model_data])
        avg_gmean = np.nanmean([d['GMean'] for d in model_data])
        std_gmean = np.nanstd([d['GMean'] for d in model_data])
        avg_balanced_accuracy = np.nanmean([d['balanced_accuracy'] for d in model_data])
        std_balanced_accuracy = np.nanstd([d['balanced_accuracy'] for d in model_data])
        avg_recall_pos = np.nanmean([d['Recall_pos'] for d in model_data])
        std_recall_pos = np.nanstd([d['Recall_pos'] for d in model_data])
        avg_precision_pos = np.nanmean([d['Precision_pos'] for d in model_data])
        std_precision_pos = np.nanstd([d['Precision_pos'] for d in model_data])
        avg_pr_auc = np.nanmean([d['PR_AUC'] for d in model_data])
        std_pr_auc = np.nanstd([d['PR_AUC'] for d in model_data])
        
        print(f"{model_name:<8} | {avg_f1:.3f}±{std_f1:.3f} | {avg_auc:.3f}±{std_auc:.3f} | {avg_mcc:.3f}±{std_mcc:.3f} | {avg_gmean:.3f}±{std_gmean:.3f} | {avg_balanced_accuracy:.3f}±{std_balanced_accuracy:.3f} | {avg_recall_pos:.3f}±{std_recall_pos:.3f} | {avg_precision_pos:.3f}±{std_precision_pos:.3f} | {avg_pr_auc:.3f}±{std_pr_auc:.3f}")
