import pandas as pd
import os
from test1_preprocess import DataPreprocessor
from test2_feature_importance import split_internal_external_datasets,split_data


def process_internal_external_data(file_path='generate_embed/preprocess_data/HE_combined.csv',
                                   date_col='入院时间（病案首页）',
                                   internal_start='2012-01-01',
                                   internal_end='2016-09-30',
                                   external_start='2016-10-01',
                                   external_end='2023-12-31',
                                   save_dir='generate_embed/preprocess_data',
                                   internal_row_thresh=0.4,
                                   external_row_thresh=0.1,
                                   col_thresh=0.9,
                                   impute_strategy='mice',
                                   knn_neighbors=5,
                                   target_column='HE',
                                   target_pre='HE_final',
                                   output_folder='generate_embed/split_data',
                                   internal_file_name=None,
                                   external_file_name=None):
    # 划分内部验证集和外部验证集
    split_internal_external_datasets(file_path=file_path,
                                     date_col=date_col,
                                     internal_start=internal_start,
                                     internal_end=internal_end,
                                     external_start=external_start,
                                     external_end=external_end,
                                     save_dir=save_dir)
    print('数据集内部和外部验证集划分完成，保存成功!')

    # 动态构造文件名，若未提供则使用默认规则
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    internal_default = f'{base_name}_internal.csv'
    external_default = f'{base_name}_external.csv'
    internal_file = internal_file_name if internal_file_name else internal_default
    external_file = external_file_name if external_file_name else external_default

    # 读取内部验证集和外部验证集
    internal_path = os.path.join(save_dir, internal_file)
    external_path = os.path.join(save_dir, external_file)
    internal_data = pd.read_csv(internal_path)
    external_data = pd.read_csv(external_path)

    # 对内部验证集和外部验证集进行数据预处理
    internal_preprocessor = DataPreprocessor(row_thresh=internal_row_thresh, col_thresh=col_thresh,
                                             impute_strategy=impute_strategy, knn_neighbors=knn_neighbors)
    external_preprocessor = DataPreprocessor(row_thresh=external_row_thresh, col_thresh=col_thresh,
                                             impute_strategy=impute_strategy, knn_neighbors=knn_neighbors)
    internal_data = internal_preprocessor.impute_missing_values(internal_data)
    external_data = external_preprocessor.impute_missing_values(external_data)

    # 保存预处理后的数据
    internal_out = os.path.join(save_dir, f'{base_name}_internal_preprocessed.csv')
    external_out = os.path.join(save_dir, f'{base_name}_external_preprocessed_{internal_row_thresh}-{external_row_thresh}.csv')
    internal_data.to_csv(internal_out, index=False)
    external_data.to_csv(external_out, index=False)
    print('内部和外部验证集数据预处理完成，保存成功!')

    # 对内部验证集进行数据划分
    split_data(internal_data, target_column=target_column, target_pre=target_pre, output_folder=output_folder)
    print('内部验证集数据划分完成，保存成功!')
# 通过限制重要性阈值获取特征
def collect_important_features(importance_df, limit=0.02):
    """
    输入构建好的最终特征重要性DataFrame（行为特征名，列为各模型的平均重要性及标准差），
    输出为所有的正向特征值DataFrame，最终的特征集合列表
    :param importance_df: 特征重要性DataFrame，行为特征名，列包含各模型的平均重要性
    :param limit: 重要性阈值，用于筛选特征
    :return: 所有正向特征值DataFrame，最终的特征集合列表
    """
    # 检查数据框的结构，确定特征名和数值列
    # 通常，CSV文件的第一列是特征名，其余列是数值
    feature_names = importance_df.iloc[:, 0].tolist()  # 获取第一列作为特征名
    # 创建一个新的数据框，使用特征名作为索引，其余列作为数据
    numeric_data = importance_df.iloc[:, 1:].copy()
    # 将特征名设置为索引
    numeric_data.index = feature_names
    # 初始化空集合用于存储最终特征
    final_feature_set = set()

    # 遍历每一列，筛选出特征值大于limit的特征
    positive_features_dict = {}
    for col in numeric_data.columns:
        try:
            # 尝试将列转换为数值类型
            numeric_col = pd.to_numeric(numeric_data[col], errors='coerce')
            # 筛选出大于阈值的特征
            positive_features = numeric_col[numeric_col > limit].sort_values(ascending=False)
            positive_features_dict[col] = positive_features
            
            print(f"Positive features for column {col} (sorted by importance):")
            print(positive_features)

            # 获取当前列筛选出的特征名
            top_features = positive_features.index.tolist()
            print(f"Selected top features for column {col}: {top_features}")

            # 更新最终特征集合
            final_feature_set.update(top_features)
        except Exception as e:
            print(f"处理列 {col} 时出错: {str(e)}")
            continue

    print(f"Final feature set (unique): {final_feature_set}")
    print(f"Final feature set size: {len(final_feature_set)}")

    # 将所有列的正向特征合并为一个DataFrame
    if positive_features_dict:
        all_positive_features = pd.DataFrame(positive_features_dict)
    else:
        all_positive_features = pd.DataFrame()

    return all_positive_features, list(final_feature_set)

def apply_final_features_to_files(importance_df, limit=0.02, data_path='split_data/HE', file_prefix='HE', target_col='target'):  
    """
    调用collect_important_features函数得到最终的特征集合，
    并将其应用于data_path路径下的前缀为指定前缀的所有文件中，仅保留最终的特征集合和目标列，
    并保留前缀为dim_的列，将得到的新文件另存独立的文件夹中
    
    :param importance_df: 特征重要性DataFrame，行为特征名，列包含各模型的平均重要性
    :param limit: 重要性阈值，用于筛选特征
    :param data_path: 数据文件所在路径，默认为'split_data'
    :param file_prefix: 文件名前缀，默认为'HE'
    :param target_col: 目标列名，默认为'target'
    """
    # 获取最终的特征集合
    _, final_feature_set = collect_important_features(importance_df, limit)
    final_feature_list = list(final_feature_set)
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"路径 {data_path} 不存在")
        return
    
    # 创建新文件夹用于保存处理后的文件
    # new_folder = f'generate_embed/filtered_data/{file_prefix}'
    new_folder = f'filtered_data/{file_prefix}'
    os.makedirs(new_folder, exist_ok=True)
    
    # 遍历路径下所有文件
    for filename in os.listdir(data_path):
        if filename.startswith(file_prefix):
            file_path = os.path.join(data_path, filename)
            try:
                # 读取文件
                df = pd.read_csv(file_path)
                # 仅保留最终特征集合中的列和前缀为dim_的列
                available_columns = [col for col in final_feature_list if col in df.columns]
                # 添加前缀为dim_的列
                dim_columns = [col for col in df.columns if col.startswith('dim_') and col not in available_columns]
                available_columns.extend(dim_columns)
                # 确保目标列在最后一列
                if target_col in df.columns and target_col not in available_columns:
                    available_columns.append(target_col)
                filtered_df = df[available_columns]
                # 保存修改后的文件到新文件夹
                new_file_path = os.path.join(new_folder, filename)
                filtered_df.to_csv(new_file_path, index=False)
                print(f"已处理文件: {filename}, 保留列数: {len(available_columns)}，保存至: {new_file_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")

    
def apply_final_features_to_external_files(importance_df, limit=0.02, data_path='preprocess_data', file_prefix='HE', target_col='target'):  
    """
    调用collect_important_features函数得到最终的特征集合，
    并将其应用于data_path路径下的前缀为指定前缀的所有文件中，仅保留最终的特征集合和目标列，
    并保留前缀为dim_的列，将得到的新文件另存独立的文件夹中
    
    :param importance_df: 特征重要性DataFrame，行为特征名，列包含各模型的平均重要性
    :param limit: 重要性阈值，用于筛选特征
    :param data_path: 数据文件所在路径，默认为'preprocess_data'
    :param file_prefix: 文件名前缀，默认为'HE'
    :param target_col: 目标列名，默认为'target'
    """
    # 获取最终的特征集合
    _, final_feature_set = collect_important_features(importance_df, limit)
    final_feature_list = list(final_feature_set)
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"路径 {data_path} 不存在")
        return
    
    # 创建新文件夹用于保存处理后的文件
    new_folder = f'filtered_externel_data'
    os.makedirs(new_folder, exist_ok=True)
    
    # 遍历路径下所有文件
    for filename in os.listdir(data_path):
        if filename.startswith(file_prefix):
            file_path = os.path.join(data_path, filename)
            try:
                # 读取文件
                df = pd.read_csv(file_path)
                # 仅保留最终特征集合中的列和前缀为dim_的列
                available_columns = [col for col in final_feature_list if col in df.columns]
                # 添加前缀为dim_的列
                dim_columns = [col for col in df.columns if col.startswith('dim_') and col not in available_columns]
                available_columns.extend(dim_columns)
                # 确保目标列在最后一列
                if target_col in df.columns and target_col not in available_columns:
                    available_columns.append(target_col)
                filtered_df = df[available_columns]
                # 保存修改后的文件到新文件夹
                new_file_path = os.path.join(new_folder, filename)
                filtered_df.to_csv(new_file_path, index=False)
                print(f"已处理文件: {filename}, 保留列数: {len(available_columns)}，保存至: {new_file_path}")
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {str(e)}")


if __name__ == "__main__":
    # # 一、合并患者数据和患者嵌入数据的代码操作
    # # 首先分别读取预处理的患者数据和患者嵌入数据
    # patient_data = pd.read_csv('preprocess_data/HE.csv')
    # disease_embed = pd.read_csv('generate_embed/HE_disease_embeddings_attention.csv')
    # # 合并患者数据和患者嵌入数据
    # combined_data = pd.concat([disease_embed, patient_data], axis=1)
    # # 保存合并后的数据
    # combined_data.to_csv('generate_embed/preprocess_data/HE_combined.csv', index=False)
    # # 读取合并后的数据
    # combined_data = pd.read_csv('generate_embed/preprocess_data/HE_combined.csv')
    # # 调用新函数划分内部数据集和外部数据集，以及训练集、动态选择集、测试集
    # process_internal_external_data()
    # # 读取特征重要性CSV文件
    # feature_importance_df = pd.read_csv('feature_importance/HE_five_fold_importance.csv')
    # # 调用函数应用特征筛选
    # apply_final_features_to_files(feature_importance_df, limit=0.01, data_path='generate_embed/split_data', file_prefix='HE', target_col='HE')



    # # 二、仅患者数据特征筛选
    # # 读取特征重要性CSV文件
    # patient_data = pd.read_csv('preprocess_data/GY.csv')
    # # 调用新函数划分内部数据集和外部数据集，以及训练集、动态选择集、测试集
    # process_internal_external_data(
    #             file_path='preprocess_data/GY.csv',
    #             date_col='入院时间（病案首页）',
    #             internal_start='2012-01-01',
    #             internal_end='2016-09-30',
    #             external_start='2016-10-01',
    #             external_end='2023-12-31',
    #             save_dir='preprocess_data',
    #             internal_row_thresh=0.4,
    #             external_row_thresh=0.1,
    #             col_thresh=0.9,
    #             impute_strategy='mice',
    #             knn_neighbors=5,
    #             target_column='GY',
    #             target_pre='GY_final',
    #             output_folder='split_data/GY'
    # )
    # 读取特征重要性CSV文件
    feature_importance_df = pd.read_csv('feature_importance/GY_five_fold_importance.csv')
    # # 调用函数应用特征筛选到内部数据集
    # apply_final_features_to_files(feature_importance_df, limit=0.01, data_path='split_data/GY', file_prefix='GY', target_col='GY')
    
    # 调用新函数应用特征筛选到外部数据集
    apply_final_features_to_external_files(feature_importance_df, limit=0.01, data_path='preprocess_data', file_prefix='GY_external_preprocessed_', target_col='GY')
