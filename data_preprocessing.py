import matplotlib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error
from tabulate import tabulate
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

class DataPreprocessor:
    """ 
    通过处理异常值、归因缺失值、转换特征和标准化特征对数据进行预处理。

    参数：
    df （pd.DataFrame）： 输入数据帧。
    row_thresh （浮点数）： 用于删除行的缺失值比率阈值。
    col_thresh（浮点）： 删除列的缺失值比率阈值。
    impute_strategy (str)： 计算缺失值的策略（“平均值”、“中位值”、“最常见值”、“常数”、“knn”、“mice”、“贝叶斯”）。
    knn_neighbors（整数）： 用于 KNN 估算的邻接数。

    返回值：
    pd.DataFrame： 预处理后的数据帧。
    """

    def __init__(self, row_thresh=0.2, col_thresh=0.8, impute_strategy='mice', knn_neighbors=5, drop_columns=[]):
        self.row_thresh = row_thresh
        self.col_thresh = col_thresh
        self.impute_strategy = impute_strategy
        self.knn_neighbors = knn_neighbors
        self.drop_columns = drop_columns
    # 一、选择特征
    def select_features(self, df):
        df.drop(columns=self.drop_columns, inplace=True)
        return df
    
    # 二、处理异常值
    # def handle_outliers(self, df):
    #     numeric_features = df.select_dtypes(include=[np.number]).columns
    #     categorical_features = df.select_dtypes(include=[object]).columns
    #     # 必要时手动指定分类特征
    #     for col in numeric_features:
    #         if df[col].nunique() <= 2:
    #             categorical_features = categorical_features.append(pd.Index([col]))
    #             numeric_features = numeric_features.drop(col)

    #     if not numeric_features.empty:
    #         conditions = pd.DataFrame(index=df.index, columns=numeric_features, dtype=bool)
    #         for col in numeric_features:
    #             Q1 = df[col].quantile(0.25)
    #             Q3 = df[col].quantile(0.75)
    #             IQR = Q3 - Q1
    #             lower = Q1 - 1.5 * IQR
    #             upper = Q3 + 1.5 * IQR
    #             conditions[col] = (df[col] < lower) | (df[col] > upper)
    #         df = df[~conditions.any(axis=1)]
    #     return df
    
    # 二、处理异常值，将其限制在第 1 和第 99 百分位数之间
    def handle_outliers(self, df):
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[object]).columns
        # 必要时手动指定分类特征
        for col in numeric_features:
            if df[col].nunique() <= 2:
                categorical_features = categorical_features.append(pd.Index([col]))
                numeric_features = numeric_features.drop(col)

        for col in numeric_features:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        return df

    # 绘制异常值箱线图
    def plot_outliers(self, df):
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[object]).columns
        # 必要时手动指定分类特征
        for col in numeric_features:
            if df[col].nunique() <= 2:
                categorical_features = categorical_features.append(pd.Index([col]))
                numeric_features = numeric_features.drop(col)
        # 新增分页绘制逻辑
        features = numeric_features.tolist()
        n_per_page = 16  # 每页最多显示20个特征
        for i in range(0, len(features), n_per_page):
            subset = features[i:i+n_per_page]
            df[subset].plot(kind='box', subplots=True, layout=(int(np.ceil(len(subset)/4)),4), figsize=(20, 10), patch_artist=True)
            plt.tight_layout()
            plt.show()
    
    # 三、处理缺失值
    def impute_missing_values(self, df):
        # （1）删除缺失值比例大于 row_thresh 的记录
        df = df[df.isnull().mean(axis=1) < self.row_thresh]
        # （2）删除缺失值比例大于 col_thresh 的列
        df = df.loc[:, df.isnull().mean() < self.col_thresh]
        # （3）识别数字和分类特征
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[object]).columns
        # 必要时手动指定分类特征
        for col in numeric_features:
            if df[col].nunique() <= 2:
                categorical_features = categorical_features.append(pd.Index([col]))
                numeric_features = numeric_features.drop(col)
        # （4）缺失值填补
        for col in numeric_features:
            missing_ratio = df[col].isnull().mean()
            if missing_ratio < 0.05: # 0.05
                if df[col].skew() < 0.5:
                    imputer = SimpleImputer(strategy='mean')
                else:
                    imputer = SimpleImputer(strategy='most_frequent')
            else:
                if self.impute_strategy == 'knn':
                    imputer = KNNImputer(n_neighbors=self.knn_neighbors)
                elif self.impute_strategy == 'mice':
                    imputer = IterativeImputer()
                elif self.impute_strategy == 'bayesian':
                    imputer = IterativeImputer(estimator=BayesianRidge())
                else:
                    imputer = SimpleImputer(strategy=self.impute_strategy)
            df[col] = imputer.fit_transform(df[[col]]).flatten()  # 修改这里，将二维数组转换为一维数组

        for col in categorical_features:
            imputer = SimpleImputer(strategy='most_frequent')
            df[col] = imputer.fit_transform(df[[col]]).flatten()  # 修改这里，将二维数组转换为一维数组

        return df

    # 四、特征转换和标准化
    # （1）特征转换：仅对数值特征进行特征转换
    def transform_and_scale_features(self, df):
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[object]).columns

        for col in numeric_features:
            if df[col].nunique() <= 2:
                categorical_features = categorical_features.append(pd.Index([col]))
                numeric_features = numeric_features.drop(col)

        # 1、对高度偏移的类进行对数变换,使其符合正太分布
        # skewed_column = ['白细胞_白细胞数_白细胞计数_白细胞数目(WBC)','出院_白细胞_白细胞数_白细胞计数_白细胞数目(WBC)']
        # df['白细胞_白细胞数_白细胞计数_白细胞数目(WBC)'] = np.log1p(df['白细胞_白细胞数_白细胞计数_白细胞数目(WBC)'] + 1)  # 加1以避免负值或0
        # df['出院_白细胞_白细胞数_白细胞计数_白细胞数目(WBC)'] = np.log1p(df['出院_白细胞_白细胞数_白细胞计数_白细胞数目(WBC)'] + 1)  # 加1以避免负值或0
        def select_transformer(col):
            if (df[col] > 0).all():
                return PowerTransformer(method='box-cox', standardize=False)
            else:
                return PowerTransformer(method='yeo-johnson', standardize=False)

        transformers = [
            ('num', Pipeline(steps=[('transform', ColumnTransformer(
                    transformers=[(col, select_transformer(col), [col]) for col in numeric_features],
                    remainder='passthrough')),('scaler', StandardScaler())]), numeric_features)
        ]
        # 拟合和转换数据
        preprocessor = ColumnTransformer(transformers=transformers)
        
        # 将结果转换回数据帧
        df_preprocessed_num = preprocessor.fit_transform(df[numeric_features])
        df_preprocessed_num = pd.DataFrame(df_preprocessed_num, columns=numeric_features)


        # 对性别特征进行0-1转换
        if '患者性别（病案首页）' in df.columns:
            df['患者性别（病案首页）'] = df['患者性别（病案首页）'].apply(lambda x: 1 if x == '男' else 0)

        # # (2)特征标准化
        # scaler = StandardScaler()
        # df_preprocessed_num = np.round(scaler.fit_transform(df_preprocessed_num), decimals=2)

        df[numeric_features] = df_preprocessed_num

        return df

    # 数据预处理
    def preprocess_data(self, df):
        df = self.select_features(df)
        df = self.handle_outliers(df)
        # df = self.impute_missing_values(df)
        df = self.transform_and_scale_features(df)
        return df
    
    # 评估缺失值填补效果
    def evaluate_imputation(self, df, random_state=42, cv=5):
        np.random.seed(random_state)

        # 初始填补：使用填补策略对原始缺失数据进行填补，生成完整数据集
        df_imputed = self.impute_missing_values(df.copy())
        
        # 修改点：使用df_imputed代替原始df获取数值列（避免被删除的列）
        numeric_features = df_imputed.select_dtypes(include=[np.number]).columns
        for col in numeric_features:
            if df_imputed[col].nunique() <= 2:  # 使用df_imputed检查唯一值
                numeric_features = numeric_features.drop(col)

        results = {col: {'R²': [], 'MAE': [], 'KS':[]} for col in numeric_features}
        
        for _ in range(cv):
            # 从填补后的完整数据中随机抽取部分实验室项目，重新置为缺失（NaN）
            df_missing = df_imputed.copy()
            for col in numeric_features:
                # 选择原始数据中没有缺失值的行
                non_missing_indices = df[df[col].notnull()].index
                # 随机选择 10% 的行重新置为缺失
                missing_indices = np.random.choice(non_missing_indices, size=int(0.1 * len(non_missing_indices)), replace=False)
                df_missing.loc[missing_indices, col] = np.nan
            
            # 再次应用相同策略填补这些人为删除的值
            df_filled = self.impute_missing_values(df_missing.copy())
            
            # 指标计算：通过决定系数（R²）和平均绝对误差（MAE）量化填补值与真实值的差异
            for col in numeric_features:
                true_values = df_imputed.loc[df_missing[col].isnull(), col]
                imputed_values = df_filled.loc[df_missing[col].isnull(), col]
                r2 = r2_score(true_values, imputed_values)
                mae = mean_absolute_error(true_values, imputed_values)
                _, p_value = ks_2samp(true_values, imputed_values)
                results[col]['R²'].append(r2)
                results[col]['MAE'].append(mae)
                results[col]['KS'].append(p_value)
        
        # 计算平均 R² 和 MAE
        avg_results = {col: {'R²': np.mean(results[col]['R²']), 'MAE': np.mean(results[col]['MAE']), 'KS': np.mean(results[col]['KS'])} for col in numeric_features}
        
        return avg_results

    # 内部/外部验证集分割
    def split_internal_external_datasets(self, file_path, date_col, internal_start, internal_end, external_start, external_end, save_dir):
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

    # 分层采样并划分五折的训练集、验证集和测试集
    def split_data(self, df, target_column, target_pre, output_folder='split_data'):
        
        numeric_features = df.select_dtypes(include=[np.number]).columns
        categorical_features = df.select_dtypes(include=[object]).columns

        for col in numeric_features:
            if df[col].nunique() <= 2:
                categorical_features = categorical_features.append(pd.Index([col]))
                numeric_features = numeric_features.drop(col)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 使用分层交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=30)
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



# 主函数
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 设置matplotlib支持中文
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    # 读取数据集
    df = pd.read_csv('datasets/GY.csv')
    drop_columns = ['并发症_门静脉高压']

    # 调用 preprocess_data 方法进行数据预处理
    preprocessor = DataPreprocessor(row_thresh=0.1, col_thresh=0.99, impute_strategy='mice', knn_neighbors=5, drop_columns=drop_columns)
    preprocessed_df = preprocessor.preprocess_data(df)
    # preprocessor.plot_outliers(preprocessed_df)

    # 将预处理后的数据保存到文件
    preprocessed_df.to_csv('preprocess_data/GY.csv', index=False)
    print('保存完成！')




    # # 评估不同的填补策略
    # strategies = ['knn', 'mice', 'bayesian', 'mean', 'median', 'most_frequent']
    # # strategies = ['knn', 'mice', 'bayesian']
    # results_dict = {}

    # for strategy in strategies:
    #     df_copy = df.copy()  # 创建数据集的副本
    #     preprocessor = DataPreprocessor(row_thresh=0.3, col_thresh=0.7, impute_strategy=strategy, knn_neighbors=5, drop_columns=drop_columns)
    #     avg_results = preprocessor.evaluate_imputation(df_copy)
    #     avg_results = pd.DataFrame(avg_results).T
    #     results_dict[strategy] = avg_results
    #     print(f"Results for {strategy} strategy:")
    #     print(avg_results)

    # # 可视化填补策略的评估结果
    # fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # for strategy, avg_results in results_dict.items():
    #     axes[0].plot(avg_results.index, avg_results['R²'], label=strategy)
    #     axes[1].plot(avg_results.index, avg_results['MAE'], label=strategy)

    # axes[0].set_title('R\u00b2 Score for Different Imputation Strategies')
    # axes[0].set_xlabel('Features')
    # axes[0].set_ylabel('R\u00b2 Score')
    # axes[0].legend()

    # axes[1].set_title('Mean Absolute Error (MAE) for Different Imputation Strategies')
    # axes[1].set_xlabel('Features')
    # axes[1].set_ylabel('MAE')
    # axes[1].legend()

    # plt.tight_layout()
    # plt.show()
