import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
import seaborn as sns
from test1_preprocess import DataPreprocessor
from functools import partial

matplotlib.rcParams['font.sans-serif'] = ['SimHEi']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载所有数据集
def load_all_datasets(data_folder, file_extension=".csv", prefix=""):
    """加载指定文件夹下的所有数据集"""
    datasets = []
    for filename in os.listdir(data_folder):
        if filename.startswith(prefix) and filename.endswith(file_extension):
            file_path = os.path.join(data_folder, filename)
            data = pd.read_csv(file_path)
            datasets.append((data, filename))
    return datasets

# 创建模型集合
def create_models():
    """创建并返回模型集合"""
    model1 = GaussianNB()
    model2 = KNeighborsClassifier(weights='distance',  n_neighbors=3, algorithm='ball_tree')
    model3 = LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear', C=0.3, random_state=42)
    model4 = LinearDiscriminantAnalysis(solver='svd',tol=1e-4)
    model5 = QuadraticDiscriminantAnalysis(reg_param=0.2,store_covariance=True)
    model6 = RandomForestClassifier(class_weight='balanced', random_state=42,
                            max_depth=10, min_samples_leaf=10, max_features='sqrt',
                            n_estimators=300)
    model7 = ExtraTreesClassifier(class_weight='balanced',max_depth=10,bootstrap=True,max_samples=0.6, random_state=42)
    model8 = AdaBoostClassifier(n_estimators=300,learning_rate=0.05, random_state=42)
    model9 = GradientBoostingClassifier(max_depth=5,learning_rate=0.1,n_iter_no_change=10, random_state=42)
    model10 = LGBMClassifier(class_weight='balanced',boosting_type='dart',
                            num_leaves=31,reg_alpha=0.1,reg_lambda=0.1, force_col_wise=True, random_state=42)
    model11 = XGBClassifier(max_delta_step=1,reg_alpha=0.1,reg_lambda=0.5,
                            eta=0.07,eval_metric='logloss', random_state=42)
    
    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10, model11]
    return models

# 训练并获取特征重要性
def train_and_get_importance(model, X_train: pd.DataFrame, y_train: pd.Series, 
                           X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """训练模型并返回特征重要性"""
    model.fit(X_train, y_train)
    
    result = permutation_importance(model, X_test, y_test, random_state=42, scoring='roc_auc',n_jobs=-1)
    return pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })

# 使用ROC-AUC作为评价指标，生成对比特征重要性
def compare_feature_importance(X_train, X_test, y_train, y_test, n_repeats=5):
    # 初始化模型字典（含参数配置）
    models_pools = {
        'NB': GaussianNB,
        'KNN': partial(KNeighborsClassifier, weights='distance',  n_neighbors=3, algorithm='ball_tree'),
        'LR': partial(LogisticRegression, class_weight='balanced', penalty='l1', solver='liblinear', C=0.3, random_state=42),
        'LDA': partial(LinearDiscriminantAnalysis, solver='svd',tol=1e-4),
        'QDA': partial(QuadraticDiscriminantAnalysis,reg_param=0.2,store_covariance=True),
        'RF': partial(RandomForestClassifier,class_weight='balanced',
                    max_depth=10, min_samples_leaf=10, max_features='sqrt',
                    n_estimators=300, random_state=42),
        'ET': partial(ExtraTreesClassifier,class_weight='balanced',max_depth=10,bootstrap=True,max_samples=0.6, random_state=42),
        'ADA': partial(AdaBoostClassifier, n_estimators=300,learning_rate=0.05, random_state=42),
        'GBC': partial(GradientBoostingClassifier,max_depth=5,learning_rate=0.1,n_iter_no_change=10, random_state=42),
        'LGBM': partial(LGBMClassifier,class_weight='balanced',boosting_type='dart',
                    num_leaves=31,reg_alpha=0.1,reg_lambda=0.1, force_col_wise=True, verbosity= -1, random_state=42),
        'XGB': partial(XGBClassifier,max_delta_step=1,reg_alpha=0.1,reg_lambda=0.5,
                    eta=0.07,eval_metric='logloss', random_state=42)
    }
    
    # 结果存储结构
    importance_df = pd.DataFrame(index=X_train.columns)
    
    # 训练并计算重要性（保持原有逻辑不变）
    for model_name, model_class in models_pools.items():
        # 实例化模型
        model = model_class()
        model.fit(X_train, y_train)
        
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            scoring='roc_auc',
            random_state=42
        )
        
        importance_df[f"{model_name}_mean"] = result.importances_mean
        importance_df[f"{model_name}_std"] = result.importances_std
    
    return importance_df

# 可视化特征重要性
def visualize_importance(importance_df):
    plt.figure(figsize=(16, 14))
    
    # 提取模型名称
    models = list(set([col.split('_')[0] for col in importance_df.columns]))  # 补全两个闭合括号
    
    # 绘制特征重要性热力图
    HEatmap_data = importance_df[[f"{m}_mean" for m in models]].T
    HEatmap_data.index = [m.replace('_mean', '') for m in HEatmap_data.index]
    
    plt.subplot(211)
    # sns.HEatmap(HEatmap_data, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Feature Importance'})
    sns.heatmap(HEatmap_data, annot=False, cmap="YlGnBu", cbar_kws={'label': 'Feature Importance'})
    plt.xticks(rotation=30)
    plt.title("Cross-Model Feature Importance HEatmap")
    
    # 绘制标准差对比
    plt.subplot(212)
    for model in models:
        plt.plot(
            importance_df.index,
            importance_df[f"{model}_mean"],
            'o',
            alpha=0.7,
            label=model
        )
    plt.xticks(rotation=30)
    plt.legend()
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.show()

# 从特征重要性数据框中提取正值特征
def get_positive_features(importance_df: pd.DataFrame) -> list:
    """获取重要性为正值的特征"""
    return importance_df[importance_df['Importance'] > 0]['Feature'].tolist()

# 获取排名前n的特征
def get_top_n_features(importance_df: pd.DataFrame, n: int) -> list:
    """获取排名前n的特征"""
    return importance_df.sort_values(by='Importance', ascending=False)['Feature'].head(n).tolist()

# 划分内部验证集和外部验证集
def split_internal_external_datasets(file_path, date_col, internal_start, internal_end, external_start, external_end, save_dir):
    """根据日期列分割内部/外部验证集并保存"""
    df = pd.read_csv(file_path)
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    
    # 分割数据并删除时间列
    internal = df[(df[date_col] >= internal_start) & (df[date_col] <= internal_end)].drop(columns=[date_col])
    external = df[(df[date_col] >= external_start) & (df[date_col] <= external_end)].drop(columns=[date_col])
    
    # 构建保存路径
    base_name = os.path.basename(file_path).replace('.csv', '')
    internal_path = os.path.join(save_dir, f"{base_name}_internal.csv")
    external_path = os.path.join(save_dir, f"{base_name}_external.csv")
    
    # 保存文件
    internal.to_csv(internal_path, index=False)
    external.to_csv(external_path, index=False)
    
    return internal_path, external_path

# 分层采样并划分五折的训练集（训练集和动态选择集）和测试集
def split_data(df, target_column, target_pre, output_folder='split_data'):
    
    numeric_features = df.select_dtypes(include=[np.number]).columns
    categorical_features = df.select_dtypes(include=[object]).columns

    for col in numeric_features:
        if df[col].nunique() <= 2:
            categorical_features = categorical_features.append(pd.Index([col]))
            numeric_features = numeric_features.drop(col)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 使用分层交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(skf.split(df, df[target_column])):
        # 显式创建副本避免视图问题
        train_df = df.iloc[train_index].copy() 
        test_df = df.iloc[test_index].copy()

        # # 使用 train_test_split 划分训练集和动态选择集
        train_df, desl_df = train_test_split(train_df, test_size=0.2, stratify=train_df[target_column], random_state=42)

        # # 确保desl_df也是副本（虽然train_test_split会返回副本，但显式添加更安全）
        # desl_df = desl_df.copy() 

        # 特征标准化
        scaler = StandardScaler()
        # 使用.loc方式确保明确赋值（原代码已正确使用，此处保持不变）
        train_df.loc[:, numeric_features] = np.round(scaler.fit_transform(train_df.loc[:, numeric_features]), decimals=3)
        test_df.loc[:, numeric_features] = np.round(scaler.transform(test_df.loc[:, numeric_features]), decimals=3)
        desl_df.loc[:, numeric_features] = np.round(scaler.transform(desl_df.loc[:, numeric_features]), decimals=3)

        # 保存数据集
        train_df.to_csv(os.path.join(output_folder, f'{target_pre}_train_fold_{fold+1}.csv'), index=False)
        desl_df.to_csv(os.path.join(output_folder, f'{target_pre}_desl_fold_{fold+1}.csv'), index=False)
        test_df.to_csv(os.path.join(output_folder, f'{target_pre}_test_fold_{fold+1}.csv'), index=False)


if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
    matplotlib.rcParams['axes.unicode_minus'] = False

    # 创建模型列表
    models = create_models()
    # 读取原始数据集
    original_data = pd.read_csv('preprocess_data/GY.csv')

    # 划分内部验证集和外部验证集
    split_internal_external_datasets(file_path='preprocess_data/GY.csv',
                                     date_col='入院时间（病案首页）',
                                     internal_start='2012-01-01', internal_end='2016-09-30',
                                     external_start='2016-10-01', external_end='2023-12-31',
                                     save_dir='preprocess_data')
    print('数据集内部和外部验证集划分完成，保存成功!')

    # 读取内部验证集和外部验证集
    internal_data = pd.read_csv('preprocess_data/GY_internal.csv')
    external_data = pd.read_csv('preprocess_data/GY_external.csv')

    # 对内部验证集和外部验证集进行数据预处理
    internal_preprocessor = DataPreprocessor(row_thresh=0.4, col_thresh=0.9, impute_strategy='mice', knn_neighbors=5)
    external_preprocessor = DataPreprocessor(row_thresh=0.1, col_thresh=0.9, impute_strategy='mice', knn_neighbors=5)
    internal_data = internal_preprocessor.impute_missing_values(internal_data)
    external_data = external_preprocessor.impute_missing_values(external_data)
    internal_data.to_csv('preprocess_data/GY_internal_preprocessed.csv', index=False)
    external_data.to_csv('preprocess_data/GY_external_preprocessed_0.4-0.1.csv', index=False)
    print('内部和外部验证集数据预处理完成，保存成功!')

    # 对内部验证集进行数据划分
    split_data(internal_data, target_column='GY', target_pre='GY_final', output_folder='split_data/GY')
    print('内部验证集数据划分完成，保存成功!')

    # 初始化用于存储五折训练集特征重要性的DataFrame
    all_fold_importance = {}
    models_pools = {
        'NB': GaussianNB,
        'KNN': partial(KNeighborsClassifier, weights='distance',  n_neighbors=3, algorithm='ball_tree'),
        'LR': partial(LogisticRegression, class_weight='balanced', penalty='l1', solver='liblinear', C=0.3, random_state=42),
        'LDA': partial(LinearDiscriminantAnalysis, solver='svd',tol=1e-4),
        'QDA': partial(QuadraticDiscriminantAnalysis,reg_param=0.2,store_covariance=True),
        'RF': partial(RandomForestClassifier,class_weight='balanced',
                    max_depth=10, min_samples_leaf=10, max_features='sqrt',
                    n_estimators=300, random_state=42),
        'ET': partial(ExtraTreesClassifier,class_weight='balanced',max_depth=10,bootstrap=True,max_samples=0.6, random_state=42),
        'ADA': partial(AdaBoostClassifier, n_estimators=300,learning_rate=0.05, random_state=42),
        'GBC': partial(GradientBoostingClassifier,max_depth=5,learning_rate=0.1,n_iter_no_change=10, random_state=42),
        'LGBM': partial(LGBMClassifier,class_weight='balanced',boosting_type='dart',
                    num_leaves=31,reg_alpha=0.1,reg_lambda=0.1, force_col_wise=True, verbosity= -1, random_state=42),
        'XGB': partial(XGBClassifier,max_delta_step=1,reg_alpha=0.1,reg_lambda=0.5,
                    eta=0.07,eval_metric='logloss', random_state=42)
    }
    
    for model_name in models_pools.keys():
        all_fold_importance[model_name] = []

    # 对五折训练集进行特征重要性选择
    for fold in range(1, 6):
        train_data = pd.read_csv(f'split_data/GY/GY_final_train_fold_{fold}.csv')
        test_data = pd.read_csv(f'split_data/GY/GY_final_test_fold_{fold}.csv')  # 读取测试集数据
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]  # 使用测试集特征
        y_test = test_data.iloc[:, -1]  # 使用测试集标签

        importance_df = compare_feature_importance(X_train, X_test, y_train, y_test)  # 使用测试集进行评估
        for model_name in all_fold_importance:
            all_fold_importance[model_name].append(importance_df[[f"{model_name}_mean", f"{model_name}_std"]])

    # 计算五折训练集特征重要性的均值
    final_importance_df = pd.DataFrame(index=X_train.columns)
    for model_name in all_fold_importance:
        combined_mean = pd.concat([df[f"{model_name}_mean"] for df in all_fold_importance[model_name]], axis=1)
        combined_std = pd.concat([df[f"{model_name}_std"] for df in all_fold_importance[model_name]], axis=1)
        final_importance_df[f"{model_name}_mean"] = combined_mean.mean(axis=1)
        final_importance_df[f"{model_name}_std"] = combined_std.mean(axis=1)

    # 修正文件路径拼写错误
    final_importance_df.to_csv('feature_importance/GY_five_fold_importance.csv', index=True)
    print('五折训练集特征重要性计算完成，保存成功!')

    # # 计算模型间重要性相关性矩阵
    # models_pools = ["GaussianNB", "KNeighbors", "LogisticRegression", "LDA", "QDA", 
    #                "RandomForest", "ExtraTrees", "AdaBoost", "GradientBoosting", 
    #                "LightGBM", "XGBoost"]
    # corr_matrix = final_importance_df[[f"{m}_mean" for m in models_pools]].corr()

    # # 计算特征排名差异
    # rank_diffs = {}
    # for feat in final_importance_df.index:
    #     ranks = []
    #     for model in models_pools:
    #         sorted_features = final_importance_df.sort_values(f"{model}_mean", ascending=False).index
    #         ranks.append(np.where(sorted_features == feat)[0][0])
    #     rank_diffs[feat] = np.std(ranks)  # 用标准差衡量排名波动

    # print("模型间特征重要性相关性：\n", corr_matrix)
    # print("\n特征排名波动性：\n", pd.Series(rank_diffs).sort_values())