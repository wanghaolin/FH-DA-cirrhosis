import numpy as np
import pandas as pd
import os
import time
import warnings
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, roc_curve, precision_recall_curve, recall_score, precision_score, auc
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from sklearn.inspection import permutation_importance
from tabulate import tabulate
import logging
warnings.filterwarnings('ignore')

# 配置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 

from deslib.dcs import OLA, LCA, MCB, APriori, APosteriori, MLA, Rank
from deslib.des import METADES, KNORAE, DESP, KNORAU, DESClustering, DESKNN, KNOP, DESMI
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.static.static_selection import StaticSelection
from deslib.static.stacked import StackedClassifier
from my_deslib.des import DESFH
from my_deslib.des.des_FHMW_AllBoxes_vector import DESFHMW_allboxes_vector
from my_deslib.des.des_FHMW_JFB_vector import DESFHMW_JFB_vector
from my_deslib.des.des_FHMW_prior_vector import DESFHMW_prior_vector
from my_deslib.des.fh_des_AllBoxes_vector import FHDES_Allboxes_vector
from my_deslib.des.fh_des_JFB_vector import FHDES_JFB_vector
from my_deslib.des.fh_des_prior_vector import FHDES_prior_vector
# 导入基于马氏距离的超椭球体版本的类
from my_deslib.des.hyperellipsoid_versions.fh_des_prior_vector_he import FHDES_prior_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_AllBoxes_vector_he import FHDES_Allboxes_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he import FHDES_JFB_vector_he
from my_deslib.des.hyperellipsoid_versions.des_FHMW_AllBoxes_vector_he import DESFHMW_allboxes_vector_he
from my_deslib.des.hyperellipsoid_versions.des_FHMW_JFB_vector_he import DESFHMW_JFB_vector_he
from my_deslib.des.hyperellipsoid_versions.des_FHMW_prior_vector_he import DESFHMW_prior_vector_he

# 导入超盒形状变体sphere
from my_deslib.des.fh_des_JFB_vector_sphere import FHDES_JFB_vector_sphere
from my_deslib.des.fh_des_JFB_vector_rectangle import FHDES_JFB_vector_rectangle
from my_deslib.des.sphere_versions.des_FHMW_AllBoxes_vector_sphere import DESFHMW_allboxes_vector_sphere
from my_deslib.des.sphere_versions.des_FHMW_JFB_vector_sphere import DESFHMW_JFB_vector_sphere
from my_deslib.des.sphere_versions.des_FHMW_prior_vector_sphere import DESFHMW_prior_vector_sphere
from my_deslib.des.sphere_versions.fh_des_AllBoxes_vector_sphere import FHDES_AllBoxes_vector_sphere
from my_deslib.des.sphere_versions.fh_des_JFB_vector_sphere import FHDES_JFB_vector_sphere
from my_deslib.des.sphere_versions.fh_des_prior_vector_sphere import FHDES_prior_vector_sphere
# 导入超盒形状变体rectangle
from my_deslib.des.rectangle_versions.des_FHMW_AllBoxes_vector_rectangle import DESFHMW_allboxes_vector_rectangle
from my_deslib.des.rectangle_versions.des_FHMW_JFB_vector_rectangle import DESFHMW_jfb_vector_rectangle
from my_deslib.des.rectangle_versions.des_FHMW_prior_vector_rectangle import DESFHMW_prior_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_AllBoxes_vector_rectangle import FHDES_AllBoxes_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_JFB_vector_rectangle import FHDES_JFB_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_prior_vector_rectangle import FHDES_prior_vector_rectangle

from my_deslib.des.hyperellipsoid_versions.des_FHMW_AllBoxes_vector_he import DESFHMW_allboxes_vector_he

# 导入基于聚类的模糊超盒动态集成方法
from my_deslib.des.fh_des_JFB_vector_clustering import FHDES_JFB_vector_clustering
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_clustering import FHDES_AllBoxes_vector_clustering
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_he_clustering import FHDES_AllBoxes_vector_he_clustering
from my_deslib.des.fh_des_JFB_vector_clustering_advance import FHDES_JFB_vector_clustering_advance
# 导入基于聚类和密度调整的超椭球体动态集成方法
from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he_clustering import FHDES_JFB_vector_he_clustering
from my_deslib.des.fh_des_clustering.des_FHMW_AllBoxes_vector_clustering import DESFHMW_allboxes_vector_clustering
from my_deslib.des.rectangle_versions.des_FHMW_AllBoxes_vector_rectangle_clustering import DESFHMW_allboxes_vector_rectangle_clustering
from my_deslib.des.sphere_versions.des_FHMW_AllBoxes_vector_sphere_clustering import DESFHMW_allboxes_vector_sphere_clustering
# 导入基于聚类和局部稳定性的高级超椭球体动态集成方法
from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he_clustering_advance import FHDES_JFB_vector_he_clustering_advance
# 导入优化后的聚类模型
from my_deslib.des.fh_des_clustering.des_FHMW_JFB_vector_clustering_optimized import DESFHMW_JFB_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_clustering_optimized import FHDES_JFB_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_he_clustering_optimized import FHDES_JFB_vector_he_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_Allboxes_vector_he_clustering_optimized import FHDES_Allboxes_vector_he_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_Allboxes_vector_clustering_optimized import FHDES_Allboxes_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_sphere_clustering_optimized import FHDES_AllBoxes_vector_sphere_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_rectangle_clustering_optimized import FHDES_AllBoxes_vector_rectangle_clustering_optimized
# 导入新创建的聚类优化版本 - 球体和矩形
from my_deslib.des.sphere_versions.fh_des_JFB_vector_sphere_clustering_optimized import FHDES_JFB_vector_sphere_clustering_optimized
from my_deslib.des.rectangle_versions.fh_des_JFB_vector_rectangle_clustering_optimized import FHDES_JFB_vector_rectangle_clustering_optimized

from sklearn.model_selection import StratifiedKFold
from tabulate import tabulate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from functools import partial
from sklearn.metrics import pairwise_distances
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from itertools import combinations
import numpy as np
from sklearn.metrics import confusion_matrix

def load_datasets_from_folder(folder_path, file_extension=".csv", prefix="ADASYN_"):
    """加载数据集，不划分验证集"""
    train_datasets = []
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(file_extension):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1].values  # 转换为numpy数组
            y = data.iloc[:, -1].values    # 转换为numpy数组
            train_datasets.append((X, y))
    return train_datasets


class DESModelEvaluator:
    def __init__(self, models, train_datasets, X_desl, y_desl, X_test, y_test, diversity_threshold=0.3):
        self.models = models
        self.train_datasets = train_datasets
        # self.validation_sets = validation_sets  # 修改为验证集集合
        self.X_test = X_test
        self.y_test = y_test
        self.X_desl = X_desl
        self.y_desl = y_desl
        self.diversity_threshold = diversity_threshold
        self.classifier_pools = None

    def optimize_threshold(self, y_test, y_pred_proba, method='f1'):
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
     
    def _conditional_undersampling(self, X, y, sampling_strategy=0.5, random_state=42):
        """
        条件欠采样：保留所有少数类样本，仅对多数类样本进行采样
        
        参数:
            X: 特征数据
            y: 标签数据
            sampling_strategy: 多数类采样比例
            random_state: 随机种子
            
        返回:
            欠采样后的特征和标签
        """
        # 确定多数类和少数类
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        
        # 创建采样策略字典：保留所有少数类，对多数类进行指定比例采样
        minority_count = counts[np.argmin(counts)]
        majority_count = counts[np.argmax(counts)]
        target_majority_count = int(majority_count * sampling_strategy)
        
        sampling_strategy_dict = {
            majority_class: target_majority_count,
            minority_class: minority_count
        }
        
        # 执行条件欠采样
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy_dict, random_state=random_state)
        X_resampled, y_resampled = rus.fit_resample(X, y)
        
        print(f"  条件欠采样后 - 多数类: {sum(y_resampled == majority_class)}, 少数类: {sum(y_resampled == minority_class)}")
        return X_resampled, y_resampled
    
    def _generate_minority_samples(self, X, y, method='smote', minority_ratio=0.3, random_state=42):
        """
        使用不同的生成模型增强少数类样本
        
        参数:
            X: 特征数据
            y: 标签数据
            method: 生成模型方法
            minority_ratio: 最终少数类占多数类的比例
            random_state: 随机种子
            
        返回:
            增强后的特征和标签
        """
        # 确定多数类和少数类
        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]
        minority_class = classes[np.argmin(counts)]
        
        # 计算目标少数类样本数量
        current_majority_count = sum(y == majority_class)
        current_minority_count = sum(y == minority_class)
        target_minority_count = int(current_majority_count * minority_ratio)
        samples_to_generate = target_minority_count - current_minority_count
        
        if samples_to_generate <= 0:
            print(f"  无需生成样本，当前少数类比例已满足要求")
            return X, y
        
        # 执行数据增强
        try:
            if method == 'smote':
                sampler = SMOTE(sampling_strategy={minority_class: target_minority_count}, 
                               random_state=random_state, k_neighbors=min(5, current_minority_count-1))
            elif method == 'adasyn':
                sampler = ADASYN(sampling_strategy={minority_class: target_minority_count}, 
                                 random_state=random_state, n_neighbors=min(5, current_minority_count-1))
            elif method == 'borderline_smote':
                sampler = BorderlineSMOTE(sampling_strategy={minority_class: target_minority_count}, 
                                         random_state=random_state, k_neighbors=min(5, current_minority_count-1))
            elif method == 'svm_smote':
                sampler = SVMSMOTE(sampling_strategy={minority_class: target_minority_count}, 
                                  random_state=random_state, k_neighbors=min(5, current_minority_count-1))
            else:  # 基于生成模型的方法
                # 将数据转换为DataFrame格式以适应SDV库
                X_df = pd.DataFrame(X.copy())
                y_df = pd.Series(y, name='target')
                train_data = pd.concat([X_df, y_df], axis=1)
                
                # 获取元数据
                metadata = Metadata.detect_from_dataframe(data=train_data)
                
                # 初始化相应的生成模型
                if method == 'gaussian_copula':
                    model = GaussianCopulaSynthesizer(metadata=metadata)
                elif method == 'ctgan':
                    model = CTGANSynthesizer(metadata=metadata, epochs=300)
                elif method == 'tvae':
                    model = TVAESynthesizer(metadata=metadata, epochs=300)
                elif method == 'copulagan':
                    model = CopulaGANSynthesizer(metadata=metadata, epochs=300)
                else:
                    raise ValueError(f"不支持的生成方法: {method}")
                
                # 训练模型（拟合整个训练集）
                model.fit(train_data)
                
                # 确定目标少数类比例为30%
                target_minority_ratio = 0.3
                
                # 获取原始数据中多数类和少数类的信息
                unique_classes, class_counts = np.unique(y, return_counts=True)
                majority_class = unique_classes[np.argmax(class_counts)]
                minority_class = unique_classes[np.argmin(class_counts)]
                majority_count = max(class_counts)
                minority_count = min(class_counts)
                
                # 计算需要生成的少数类样本数量，确保最终比例为30%
                target_minority_count = int(majority_count * target_minority_ratio)
                minority_samples_to_add = max(0, target_minority_count - minority_count)
                
                # 如果需要添加少数类样本
                if minority_samples_to_add > 0:
                    # 增加缓冲量，确保能获得足够的少数类样本
                    buffer_multiplier = 2.0
                    total_samples_to_generate = int(minority_samples_to_add / (minority_count / len(train_data)) * buffer_multiplier)
                    
                    # 设置随机种子确保复现性
                    np.random.seed(random_state)
                    
                    # 生成合成数据
                    synthetic_data = model.sample(total_samples_to_generate)
                    
                    # 筛选合成数据中的少数类样本
                    synthetic_minority_data = synthetic_data[synthetic_data['target'] == minority_class]
                    
                    # 精确选取需要的少数类样本数量
                    actual_to_add = min(minority_samples_to_add, len(synthetic_minority_data))
                    if actual_to_add > 0:
                        selected_minority_samples = synthetic_minority_data.sample(n=actual_to_add, random_state=random_state)
                        
                        # 合并原始数据和选定的少数类合成样本
                        X_combined = pd.concat([X_df, selected_minority_samples.iloc[:, :-1]], axis=0)
                        y_combined = pd.concat([y_df, selected_minority_samples['target']], axis=0)
                    else:
                        # 如果没有足够的少数类合成样本，返回原始数据
                        X_combined = X_df.copy()
                        y_combined = y_df.copy()
                else:
                    # 如果已经达到目标比例，返回原始数据
                    X_combined = X_df.copy()
                    y_combined = y_df.copy()
                
                # 重新检查并确保精确的比例控制
                final_classes, final_counts = np.unique(y_combined, return_counts=True)
                final_majority_count = max(final_counts)
                final_minority_count = min(final_counts)
                
                # 如果少数类样本仍然过多，只删除多余的合成少数类样本
                if final_minority_count > int(final_majority_count * target_minority_ratio):
                    # 计算需要精确保留的少数类样本数量
                    exact_target_minority = int(final_majority_count * target_minority_ratio)
                    excess_minority = final_minority_count - exact_target_minority
                    
                    # 确定少数类标签
                    final_minority_class = final_classes[np.argmin(final_counts)]
                    
                    # 获取所有少数类样本索引
                    all_minority_indices = np.where(y_combined == final_minority_class)[0]
                    
                    # 区分原始少数类样本和合成少数类样本
                    original_minority_indices = np.where(y == final_minority_class)[0]
                    synthetic_minority_indices = [idx for idx in all_minority_indices if idx >= len(X)]
                    
                    # 只从合成样本中删除多余的
                    if len(synthetic_minority_indices) >= excess_minority:
                        np.random.seed(random_state)
                        indices_to_remove = np.random.choice(synthetic_minority_indices, size=excess_minority, replace=False)
                        mask = np.ones(len(X_combined), dtype=bool)
                        mask[indices_to_remove] = False
                        X_combined = X_combined[mask]
                        y_combined = y_combined[mask]
                
                # 最终检查
                final_classes, final_counts = np.unique(y_combined, return_counts=True)
                final_majority_count = max(final_counts)
                final_minority_count = min(final_counts)
                final_ratio = final_minority_count / final_majority_count if final_majority_count > 0 else 0
                
                print(f"  {method}生成增强后 - 多数类: {final_majority_count}, 少数类: {final_minority_count}, 比例: {final_ratio:.2f}")
                return X_combined.values, y_combined.values
            
            # 执行传统的过采样方法
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"  {method}生成增强后 - 多数类: {sum(y_resampled == majority_class)}, 少数类: {sum(y_resampled == minority_class)}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"  {method}生成增强时出错: {str(e)}")
            return X, y
    
    def generate_classifier_pools(self, models, top_k=7, random_state=42, fold=1):
        """
        生成基分类器池，通过条件欠采样和多种数据增强方法缓解类别不平衡问题
        
        参数:
            models: 基分类器字典
            min_mcc: 最小MCC阈值
            min_f1: 最小F1阈值
            min_auc: 最小AUC阈值
            min_gmean: 最小GMean阈值
            top_k: 选择的top AUC分类器数量
            random_state: 随机种子，用于确保实验可复现
            fold: 当前折数，用于保存不同折数的增强数据
        
        返回:
            优化后的分类器池
        """
        all_models_info = []  # 存储所有模型信息
        
        # 设置全局随机种子
        np.random.seed(random_state)
        
        # 获取基础训练数据
        base_X, base_y = self.train_datasets[0]  # 假设第一个数据集是基础数据集
        
        # 首先处理原始数据集
        print(f"\n处理原始数据集")
        for model_name, model_cls in models.items():
            try:
                unique_name = f"{model_name}_original"
                model = model_cls(random_state=random_state) if hasattr(model_cls, '__name__') and model_cls.__name__ not in ['SVC', 'KNeighborsClassifier', 'GaussianNB'] else model_cls()
                try:
                    model.fit(base_X, base_y)
                except TypeError:
                    model = model_cls()
                    model.fit(base_X, base_y)
                
                # 评估模型
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(self.X_test)
                AUC = roc_auc_score(self.y_test, y_pred_proba)
                optimal_threshold = self.optimize_threshold(self.y_test, y_pred_proba, method='youden')
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                F1 = f1_score(self.y_test, y_pred)
                MCC = matthews_corrcoef(self.y_test, y_pred)
                GMean = geometric_mean_score(self.y_test, y_pred)
                
                all_models_info.append({
                    'model': model,
                    'name': unique_name,
                    'original_name': model_name,
                    'sampling_method': 'original',
                    'F1': F1,
                    'AUC': AUC,
                    'MCC': MCC,
                    'GMean': GMean
                })
                
                print(f"  模型 {unique_name}: AUC={AUC:.4f}, F1={F1:.4f}")
                
            except Exception as e:
                print(f"  训练模型 {model_name}_original 时出错: {str(e)}")
        
        # 定义5种不同的随机种子用于条件欠采样
        undersample_seeds = [42, 43, 44, 45, 46]
        
        # 定义8种数据增强方法
        augmentation_methods = ['smote', 'adasyn', 'borderline_smote', 'svm_smote', 
                               'gaussian_copula', 'ctgan', 'tvae', 'copulagan'
                               ]
        
        # 增强数据保存路径，包含折数信息
        enhance_data_dir = os.path.join("enhance_data", "PD", f"fold{fold}")
        os.makedirs(enhance_data_dir, exist_ok=True)
        
        # 对每种条件欠采样数据集进行处理
        for i, undersample_seed in enumerate(undersample_seeds):
            print(f"\n条件欠采样数据集 {i+1}/{len(undersample_seeds)} (随机种子: {undersample_seed})")
            
            # 执行条件欠采样
            undersampled_X, undersampled_y = self._conditional_undersampling(
                base_X, base_y, sampling_strategy=0.5, random_state=undersample_seed)
            
            # 对每个增强方法进行处理
            for aug_method in augmentation_methods:
                print(f"  应用增强方法: {aug_method}")
                
                # 创建增强数据的保存文件名
                enhance_data_file = os.path.join(enhance_data_dir, f"undersample_{i+1}_{aug_method}_randomstate_{random_state}.npz")
                
                # 检查增强数据是否已存在
                if os.path.exists(enhance_data_file):
                    print(f"    加载已保存的增强数据: {enhance_data_file}")
                    # 加载已保存的增强数据
                    with np.load(enhance_data_file) as data:
                        augmented_X = data['augmented_X']
                        augmented_y = data['augmented_y']
                else:
                    print(f"    生成并保存增强数据: {enhance_data_file}")
                    # 执行数据增强
                    augmented_X, augmented_y = self._generate_minority_samples(
                        undersampled_X, undersampled_y, method=aug_method, 
                        minority_ratio=0.3, random_state=random_state)
                    # 保存增强数据
                    np.savez(enhance_data_file, augmented_X=augmented_X, augmented_y=augmented_y)
                
                # 对每个分类器在增强数据上训练
                for model_name, model_cls in models.items():
                    try:
                        # 创建唯一的模型名称
                        unique_name = f"{model_name}_undersample_{i+1}_{aug_method}"
                        
                        # 训练模型
                        model = model_cls(random_state=random_state) if hasattr(model_cls, '__name__') and model_cls.__name__ not in ['SVC', 'KNeighborsClassifier', 'GaussianNB'] else model_cls()
                        try:
                            model.fit(augmented_X, augmented_y)
                        except TypeError:
                            model = model_cls()
                            model.fit(augmented_X, augmented_y)
                        
                        # 评估模型
                        y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(self.X_test)
                        AUC = roc_auc_score(self.y_test, y_pred_proba)
                        optimal_threshold = self.optimize_threshold(self.y_test, y_pred_proba, method='youden')
                        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                        F1 = f1_score(self.y_test, y_pred)
                        MCC = matthews_corrcoef(self.y_test, y_pred)
                        GMean = geometric_mean_score(self.y_test, y_pred)
                        
                        all_models_info.append({
                            'model': model,
                            'name': unique_name,
                            'original_name': model_name,
                            'sampling_method': f'undersample_{i+1}_{aug_method}',
                            'F1': F1,
                            'AUC': AUC,
                            'MCC': MCC,
                            'GMean': GMean
                        })
                        
                        print(f"    模型 {unique_name}: AUC={AUC:.4f}, F1={F1:.4f}")
                        
                    except Exception as e:
                        print(f"    训练模型 {model_name}_undersample_{i+1}_{aug_method} 时出错: {str(e)}")
        
        # 按AUC降序排序所有模型
        # 在排序后添加额外的稳定性排序键，确保当AUC值相同时排序结果一致
        all_models_info.sort(key=lambda x: (x['AUC'], x['name']), reverse=True)
        
        # 选择top-k个模型
        selected_models_info = all_models_info[:top_k] if len(all_models_info) > top_k else all_models_info
        
        # 提取分类器池
        self.classifier_pools = [model_info['model'] for model_info in selected_models_info]
        
        # 打印选择结果
        print(f"\n{'='*80}")
        print(f"选择了Top-{len(selected_models_info)}个模型作为最终分类器池")
        print(f"{'='*80}")
        
        # 创建性能DataFrame
        performance_data = []
        for model_info in selected_models_info:
            performance_data.append({
                'Model Name': model_info['name'],
                'Original Model': model_info['original_name'],
                'Sampling Method': model_info['sampling_method'],
                'AUC': model_info['AUC'],
                'F1': model_info['F1'],
                'MCC': model_info['MCC'],
                'GMean': model_info['GMean']
            })
        
        performance_df = pd.DataFrame(performance_data)
        print(tabulate(performance_df, headers="keys", tablefmt="grid", floatfmt=".4f"))
        
        # # 保存性能结果到CSV文件
        # output_dir = os.path.join("filtered_data", "HE", "enhance")
        # os.makedirs(output_dir, exist_ok=True)
        # performance_csv_path = os.path.join(output_dir, "classifier_pool_performance.csv")
        # performance_df.to_csv(performance_csv_path, index=False)
        # print(f"\n性能结果已保存至: {performance_csv_path}")
        
        # 计算模型多样性统计
        self._calculate_diversity_stats(selected_models_info)
        
        return self.classifier_pools
    def _stratified_sample(self, X, y, sample_size=0.8, random_state=42):
        """执行分层采样，保持类别比例
        
        参数:
            X: 特征数据
            y: 标签数据
            sample_size: 采样比例
            random_state: 随机种子，确保复现性
        """
        from sklearn.model_selection import train_test_split
        X_sampled, _, y_sampled, _ = train_test_split(
            X, y, 
            test_size=1-sample_size, 
            stratify=y, 
            # random_state=np.random.randint(1, 1000)
            random_state=random_state  # 使用固定的随机种子
        )
        return X_sampled, y_sampled
    
    def _bootstrap_sample(self, X, y, sample_size=1.0, random_state=42):
        """执行分层bootstrap重采样，按类别分别进行有放回抽样后合并
        
        参数:
            X: 特征数据
            y: 标签数据
            sample_size: 采样比例，相对于每个类别的原始数量
            random_state: 随机种子，确保复现性
            
        返回:
            分层bootstrap采样后的数据
        """
        # 设置随机种子
        rng = np.random.RandomState(random_state)
        
        # 获取所有唯一类别
        unique_classes = np.unique(y)
        sampled_indices = []
        
        # 对每个类别分别进行有放回重采样
        for i, cls in enumerate(unique_classes):
            # 获取当前类别的所有索引
            cls_indices = np.where(y == cls)[0]
            # 计算当前类别的采样数量
            n_samples_cls = int(len(cls_indices) * sample_size)
            # 从当前类别中进行有放回抽样，为每个类别使用不同的子种子
            bootstrap_indices = rng.choice(cls_indices, size=n_samples_cls, replace=True)
            sampled_indices.extend(bootstrap_indices)
        
        # 打乱合并后的索引顺序，使用固定的随机种子
        rng.shuffle(sampled_indices)
        
        # 确保是DataFrame格式
        if isinstance(X, pd.DataFrame):
            X_sampled = X.iloc[sampled_indices].reset_index(drop=True)
        else:
            X_sampled = X[sampled_indices]
            
        # 确保是Series或数组格式
        if isinstance(y, pd.Series):
            y_sampled = y.iloc[sampled_indices].reset_index(drop=True)
        else:
            y_sampled = y[sampled_indices]
            
        # 计算采样后各类别分布
        sampled_classes, counts = np.unique(y_sampled, return_counts=True)
        class_distribution = dict(zip(sampled_classes, counts))
        print(f"  分层bootstrap采样后类别分布: {class_distribution}")
            
        return X_sampled, y_sampled
    
    def _calculate_diversity_stats(self, selected_models_info):
        """计算并显示分类器池的多样性统计"""
        # 统计不同原始模型的数量
        original_model_counts = {}
        for model_info in selected_models_info:
            original_model = model_info['original_name']
            original_model_counts[original_model] = original_model_counts.get(original_model, 0) + 1
        
        # 统计不同采样方法的数量
        sampling_method_counts = {}
        for model_info in selected_models_info:
            method = model_info['sampling_method']
            sampling_method_counts[method] = sampling_method_counts.get(method, 0) + 1
        
        print(f"\n{'='*60}")
        print("分类器池多样性统计")
        print(f"{'='*60}")
        print(f"原始模型分布:")
        for model, count in original_model_counts.items():
            print(f"  {model}: {count}个模型 ({count/len(selected_models_info)*100:.1f}%)")
        
        print(f"\n采样方法分布:")
        for method, count in sampling_method_counts.items():
            print(f"  {method}: {count}个模型 ({count/len(selected_models_info)*100:.1f}%)")
        
        print(f"\n原始模型种类数: {len(original_model_counts)}")
        print(f"平均每个原始模型的实例数: {len(selected_models_info)/len(original_model_counts):.1f}")
        print(f"{'='*60}")

    def calculate_iba(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        iba = (2 * acc * f1) / (acc + f1)
        return iba

    def evaluate_methods(self, random_state=42):
        """评估不同动态选择方法"""
        # 设置随机种子确保复现性
        np.random.seed(random_state)
        methods = {
            # 理论最优模型
            'Oracle': (Oracle, {}),
            # 动态选择方法
            'LCA': (LCA, {'diff_thresh':0.05, 'k': 7}),
            'Rank': (Rank, {'diff_thresh':0.05, 'k': 7}),
            'OLA': (OLA, {'diff_thresh':0.05, 'k': 7}),
            'APriori': (APriori, {'diff_thresh':0.05, 'k': 7}),
            'MCB': (MCB, {'diff_thresh': 0.05, 'k': 7}),
            # 动态集成方法
            'MLA': (MLA, {'k': 7}),
            # 'voting': 'soft',
            'METADES': (METADES, {'k': 7}),
            'KNORAE': (KNORAE, {'k': 7}),
            'KNORAU': (KNORAU, {'k': 7}),
            # 'metric_performance':'f1_score'
            'DESClustering': (DESClustering, {}),
            'KNOP': (KNOP, {'k': 7}),
            'DESP': (DESP, {'k': 7}),
            'DESKNN': (DESKNN, {'k': 7}),

            # GY-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4等参数不变，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            'FH_DES': (
                FHDES_JFB_vector,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_ellipsoidal': (
                FHDES_JFB_vector_he,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'gamma': 0.7581691436530708,
                    'reg_lambda': 1.13798107855186e-05,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_spherical': (
                FHDES_JFB_vector_sphere,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'sigma': 0.028175933150630706,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_bi-slope_rectangular': (
                FHDES_JFB_vector_rectangle,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'alpha': 1.3320075338937662,
                    'beta': 4.9713071807123175,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            # GY-改进的超盒变体
            'FH_DA': (
                FHDES_JFB_vector_clustering_optimized, 
                {
                    'theta':0.001, 'mu':.4, 'mis_sample_based':True,
                    'doContraction':True, 'thetaCheck':False, 'multiCore_process':True, 'shuffle_dataOrder':False,
                    'n_clusters': 10, 'bandwidth_selection': 'cv', 'density_estimation': 'knn_density', 'knn_k': 6, 
                    'use_pca': False, 'pca_components': 6, 'normalize_weights': False, 'normalize_correlations': True, 
                    'bandwidth': 0.315023760097458, 'density_weight': 0.15000000000000002, 'cluster_distance_weight': 0.584526702261065, 
                    'basic_weight': 0.1, 'enhancement_weight': 0.9488213816195104, 'epsilon': 4.4875730462234466e-08
                 }
            ),
            'FH_DA_ellipsoidal': (
                FHDES_JFB_vector_he_clustering_optimized,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False,
                    'n_clusters': 9,
                    'bandwidth_selection': 'adaptive',
                    'density_estimation': 'pca_kde',
                    'knn_k': 7,
                    'use_pca': True,
                    'pca_components': 3,
                    'normalize_weights': True,
                    'normalize_correlations': True,
                    'bandwidth': 2.9547412518596725,
                    'density_weight': 0.5,
                    'cluster_distance_weight': 0.689299246291069,
                    'basic_weight': 0.7000000000000001,
                    'enhancement_weight': 0.17644429085079516,
                    'epsilon': 6.459686488008564e-08,
                    'gamma': 0.010923198627741751,
                    'reg_lambda': 4.210947032191968e-06
                }
            ),
            'FH_DA_spherical': (
                FHDES_AllBoxes_vector_sphere_clustering_optimized,
                {
                    'mis_sample_based': True,
                    'doContraction': False,
                    'thetaCheck': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False,
                    'n_clusters': 3,
                    'bandwidth_selection': 'cv',
                    'density_estimation': 'kde',
                    'knn_k': 4,
                    'use_pca': True,
                    'pca_components': 9,
                    'normalize_weights': False,
                    'normalize_correlations': True,
                    'bandwidth': 1.4639562535426875,
                    'theta': 0.8277044306380984,
                    'mu': 0.45000000000000007,
                    'density_weight': 1.0,
                    'cluster_distance_weight': 0.2661541039538596,
                    'basic_weight': 0.85,
                    'enhancement_weight': 0.3875854066581769,
                    'epsilon': 4.126349943114586e-08,
                    'sigma': 0.4868663394812949
                }
            ),
            'FH_DA_bi-slope_rectangular': (
                FHDES_AllBoxes_vector_rectangle_clustering_optimized,
                {
                    'n_clusters': 5,
                    'bandwidth_selection': 'silverman',
                    'density_estimation': 'kde',
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False,
                    'knn_k': 3,
                    'use_pca': False,
                    'pca_components': 5,
                    'normalize_weights': True,
                    'normalize_correlations': False,
                    'bandwidth': 0.27887919279027584,
                    'density_weight': 0.85,
                    'cluster_distance_weight': 0.590801864085726,
                    'basic_weight': 0.9,
                    'enhancement_weight': 0.3644400902697933,
                    'epsilon': 3.70645854340426e-08,
                    'alpha': 1.181425093183144,
                    'beta': 3.270077472634777
                }
            ),

            
            # HE-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            'FH_DES': (
                FHDES_JFB_vector,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_ellipsoidal': (
                FHDES_JFB_vector_he,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'gamma': 10.165830776156998,
                    'reg_lambda': 1.7290103161667023e-05,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_spherical': (
                FHDES_JFB_vector_sphere,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'sigma': 0.9958059560727144,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_bi-slope_rectangular': (
                FHDES_JFB_vector_rectangle,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'alpha': 0.8153751093671162,
                    'beta': 4.691449548222193,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            # HE-改进的超盒变体
            # 快速计算配置 - 低计算复杂度
            'FH_DA': (
                FHDES_JFB_vector_clustering_optimized,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'n_clusters': 2,
                    'bandwidth_selection': 'silverman',
                    'use_pca': True,
                    'pca_components': 5,
                    'density_estimation': 'knn_density',
                    'knn_k': 2,
                    'density_weight': 0.5,
                    'cluster_distance_weight': 0.5,
                    'normalize_weights': True,
                    'epsilon': 1e-8,
                    'normalize_correlations': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            # 超椭球体版本 - 平衡配置 (2v配置)
            'FH_DA_ellipsoidal': (
                FHDES_JFB_vector_he_clustering_optimized,
                {
                    'theta': 0.001,
                    'mu': 0.85,
                    'gamma': 0.06101491367302711,
                    'reg_lambda': 2.310201887845294e-06,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'n_clusters': 3,
                    'bandwidth_selection': 'silverman',
                    'use_pca': True,
                    'pca_components': 10,
                    'density_estimation': 'kde',
                    'density_weight': 0.6,
                    'cluster_distance_weight': 0.4,
                    'normalize_weights': True,
                    'epsilon': 1e-8,
                    'normalize_correlations': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DA_spherical': (
                FHDES_JFB_vector_sphere_clustering_optimized,
                {
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False,
                    'n_clusters': 2,
                    'bandwidth_selection': 'adaptive',
                    'density_estimation': 'kde',
                    'knn_k': 6,
                    'use_pca': True,
                    'pca_components': 10,
                    'normalize_weights': False,
                    'normalize_correlations': True,
                    'bandwidth': 0.6082563589986993,
                    'theta': 0.17063386522055357,
                    'mu': 0.35,
                    'density_weight': 0.35,
                    'cluster_distance_weight': 0.9439845768431889,
                    'basic_weight': 0.1,
                    'enhancement_weight': 0.7734201421310619,
                    'epsilon': 5.7124175613557085e-08,
                    'sigma': 0.38070292327594585
                }
            ),
            'FH_DA_bi-slope_rectangular': (
                FHDES_AllBoxes_vector_rectangle_clustering_optimized,
                {
                    'theta': 0.001,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False,
                    'n_clusters': 3,
                    'bandwidth_selection': 'silverman',
                    'density_estimation': 'knn_density',
                    'knn_k': 4,
                    'use_pca': False,
                    'pca_components': 8,
                    'normalize_weights': False,
                    'normalize_correlations': True,
                    'bandwidth': 2.8588262369008905,
                    'density_weight': 0.65,
                    'cluster_distance_weight': 0.17930588219809357,
                    'basic_weight': 0.6,
                    'enhancement_weight': 0.21880065868715076,
                    'epsilon': 8.365122846241532e-08,
                    'alpha': 1.1719664906128544,
                    'beta': 4.54373856060449
                }
            ),

            # # PD-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            'FH_DES': (
                FHDES_Allboxes_vector,
                {
                    'theta': 0.4,
                    'mu': 0.4,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_ellipsoidal': (
                FHDES_Allboxes_vector_he,
                {
                    'theta': 0.4,
                    'mu': 0.4,
                    'gamma': 0.8072465573781535,
                    'reg_lambda': 8.724761227721179e-06,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_spherical': (
                FHDES_AllBoxes_vector_sphere,
                {
                    'theta': 0.4,
                    'mu': 0.4,
                    'sigma': 0.7969454818643931,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            'FH_DES_bi-slope_rectangular': (
                FHDES_AllBoxes_vector_rectangle,
                {
                    'theta': 0.4,
                    'mu': 0.4,
                    'alpha': 0.14049896585154187,
                    'beta': 0.6072222164274351,
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False
                }
            ),
            # # PD-改进的超盒变体
            'FH_DA': (
                FHDES_Allboxes_vector_clustering_optimized,
                {
                    'mis_sample_based': True,
                    'doContraction': False,
                    'thetaCheck': True,
                    'multiCore_process': False,
                    'shuffle_dataOrder': True,
                    'n_clusters': 2,
                    'bandwidth_selection': 'cv',
                    'density_estimation': 'pca_kde',
                    'knn_k': 7,
                    'use_pca': True,
                    'pca_components': None,
                    'normalize_weights': True,
                    'normalize_correlations': True,
                    'bandwidth': 1.1513058574930677,
                    'theta': 0.16109297688061047,
                    'mu': 0.75,
                    'density_weight': 0.45000000000000007,
                    'cluster_distance_weight': 0.34661740876334135,
                    'basic_weight': 0.30000000000000004,
                    'enhancement_weight': 0.897994362306035,
                    'epsilon': 4.2285052164024e-08
                }
            ),
            'FH_DA_ellipsoidal': (
                FHDES_Allboxes_vector_he_clustering_optimized,
                {
                    'mis_sample_based': True,
                    'doContraction': False,
                    'thetaCheck': False,
                    'n_clusters': 8,
                    'bandwidth_selection': 'cv',
                    'density_estimation': 'knn_density',
                    'knn_k': 10,
                    'use_pca': False,
                    'pca_components': None,
                    'normalize_weights': False,
                    'normalize_correlations': False,
                    'multiCore_process': False,
                    'shuffle_dataOrder': True,
                    'bandwidth': 0.24327392618081645,
                    'mu': 0.5,
                    'theta': 0.580934134852548,
                    'density_weight': 0.30000000000000004,
                    'cluster_distance_weight': 0.1,
                    'basic_weight': 0.25,
                    'enhancement_weight': 0.4,
                    'epsilon': 9.131795402684815e-08,
                    'gamma': 14.685312005983562,
                    'reg_lambda': 1.0227915932908707e-05
                }
            ),
            'FH_DA_spherical': (
                FHDES_AllBoxes_vector_sphere_clustering_optimized,
                {
                    'mis_sample_based': True,
                    'doContraction': False,
                    'thetaCheck': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': True,
                    'n_clusters': 2,
                    'bandwidth_selection': 'adaptive',
                    'density_estimation': 'pca_kde',
                    'knn_k': 5,
                    'use_pca': False,
                    'pca_components': None,
                    'normalize_weights': False,
                    'normalize_correlations': True,
                    'bandwidth': 1.0025763510308325,
                    'theta': 0.7315157902574336,
                    'mu': 0.45000000000000007,
                    'density_weight': 0.6,
                    'cluster_distance_weight': 0.8200555983272062,
                    'basic_weight': 0.35,
                    'enhancement_weight': 0.62394472757224,
                    'epsilon': 4.039819547001331e-08,
                    'sigma': 0.10881066516231722
                }
            ),
            'FH_DA_bi-slope_rectangular': (
                FHDES_AllBoxes_vector_rectangle_clustering_optimized,
                {
                    'mis_sample_based': True,
                    'doContraction': True,
                    'thetaCheck': False,
                    'n_clusters': 9,
                    'bandwidth_selection': 'adaptive',
                    'density_estimation': 'pca_kde',
                    'knn_k': 5,
                    'use_pca': False,
                    'pca_components': 3,
                    'normalize_weights': True,
                    'normalize_correlations': True,
                    'multiCore_process': True,
                    'shuffle_dataOrder': False,
                    'bandwidth': 1.8190427869701697,
                    'mu': 0.85,
                    'theta': 0.3463644376849158,
                    'density_weight': 0.6,
                    'cluster_distance_weight': 0.1,
                    'basic_weight': 0.8,
                    'enhancement_weight': 0.4,
                    'epsilon': 4.548447107982115e-08,
                    'alpha': 1.6300899902636972,
                    'beta': 4.85858264996245
                }
            ),

            # ACLF-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            'FH_DES': (FHDES_Allboxes_vector, {'theta':0.001, 'mu':.1, 'mis_sample_based':True,
                                    'doContraction':True, 'thetaCheck':False, 'multiCore_process':False, 'shuffle_dataOrder':False}),
            'FH_DES_ellipsoidal': (FHDES_Allboxes_vector_he, {'theta':0.001, 'mu':.4,'mis_sample_based':True, 'gamma': 0.010698510489380428, 'reg_lambda': 9.373922704327301e-05,
                                    'doContraction':True, 'thetaCheck':False, 'multiCore_process':True, 'shuffle_dataOrder':False}),
            'FH_DES_spherical':(FHDES_AllBoxes_vector_sphere,{'theta':0.001, 'mu':.4, 'mis_sample_based':True, 'sigma': 0.1673808578875213,
                                'doContraction':True, 'thetaCheck':False, 'multiCore_process':True, 'shuffle_dataOrder':False}),
            'FH_DES_bi-slope_rectangular':(FHDES_AllBoxes_vector_rectangle,{'theta':0.001, 'mu':.4, 'mis_sample_based':True, 'alpha': 1.0589602580430237, 'beta': 0.5077285920181165,
                                'doContraction':True, 'thetaCheck':False, 'multiCore_process':True, 'shuffle_dataOrder':False}),
            # # # ACLF-改进的超盒变体         
  
            'FH_DA': (
                FHDES_Allboxes_vector_clustering_optimized,
                {'mis_sample_based': True, 'doContraction': True, 'thetaCheck': True, 'multiCore_process': False, 
                'shuffle_dataOrder': True, 'n_clusters': 9, 'bandwidth_selection': 'adaptive', 'density_estimation': 'pca_kde', 
                'knn_k': 6, 'use_pca': False, 'pca_components': 9, 'normalize_weights': False, 'normalize_correlations': False, 
                'bandwidth': 2.7975701046328614, 'theta': 0.6870950482390474, 'mu': 0.2, 'density_weight': 0.75, 
                'cluster_distance_weight': 0.6071418086421805, 'basic_weight': 0.65, 
                'enhancement_weight': 0.5943258552355211, 'epsilon': 1.3133080552399374e-08}
                ),
            'FH_DA_ellipsoidal': (
                    FHDES_Allboxes_vector_he_clustering_optimized,
                    {'mis_sample_based': True, 'doContraction': False, 'thetaCheck': False, 'multiCore_process': False, 
                    'shuffle_dataOrder': True, 'n_clusters': 6, 'bandwidth_selection': 'silverman', 'density_estimation': 'kde', 
                    'knn_k': 4, 'use_pca': True, 'pca_components': 7, 'normalize_weights': False, 'normalize_correlations': True, 
                    'bandwidth': 0.9145187314549602, 'theta': 0.9088200252575138, 'mu': 0.6, 'density_weight': 0.45000000000000007, 
                    'cluster_distance_weight': 0.721175651859676, 'basic_weight': 0.55, 'enhancement_weight': 0.10057062807344118, 
                    'epsilon': 7.580429004146923e-08, 'gamma': 0.1177623190541896, 'reg_lambda': 4.2930177392983695e-05}
                ),  
            'FH_DA_spherical': (
                FHDES_AllBoxes_vector_sphere_clustering_optimized,
                {'mis_sample_based': True, 'doContraction': True, 'thetaCheck': False, 
                'multiCore_process': True, 'shuffle_dataOrder': False, 'n_clusters': 9, 
                'bandwidth_selection': 'cv', 'density_estimation': 'knn_density', 'knn_k': 9, 
                'use_pca': False, 'pca_components': 8, 'normalize_weights': False, 
                'normalize_correlations': True, 'bandwidth': 0.2783297834037277, 'theta': 0.5107073270264305, 
                'mu': 0.15000000000000002, 'density_weight': 0.65, 'cluster_distance_weight': 0.41129475125766113, 
                'basic_weight': 0.65, 'enhancement_weight': 0.14116783043126158, 
                'epsilon': 8.728214380908525e-08, 'sigma': 0.8850706481483924}
               ),      
            'FH_DA_bi-slope_rectangular': (
                FHDES_AllBoxes_vector_rectangle_clustering_optimized,
                {'mis_sample_based': True, 'doContraction': True, 'thetaCheck': False, 'multiCore_process': False, 
                'shuffle_dataOrder': True, 'n_clusters': 5, 'bandwidth_selection': 'adaptive', 'density_estimation': 'kde', 
                'knn_k': 4, 'use_pca': True, 'pca_components': 10, 'normalize_weights': True, 'normalize_correlations': True, 
                'bandwidth': 0.5614924866063653, 'theta': 0.5705753278606428, 'mu': 0.55, 'density_weight': 0.65, 
                'cluster_distance_weight': 0.7982400774120468, 'basic_weight': 0.35, 'enhancement_weight': 0.17896348521992975, 
                'epsilon': 4.573243428322887e-08, 'alpha': 1.0269641738184827, 'beta': 0.5108427899165016}
                ),    

        }

        results = []

        for name, (method_cls, method_params) in methods.items():
            # 合并默认参数和方法特定参数
            params = {
                'pool_classifiers': self.classifier_pools,
                'random_state': 42,
                **method_params
            }
            des = method_cls(**params)
            if name == "AdaptiveFHDE_he":
                X_train, y_train = self.train_datasets[0]
                des.fit(self.X_desl, self.y_desl, X_train, y_train, self.X_test, self.y_test)
            elif name == "FHDES_JFB_DE":
                # 为FHDES_JFB_DE使用特殊的fit_with_adaptive_de方法进行参数优化
                X_train, y_train = self.train_datasets[0]
                # 分割训练集和验证集用于DE优化
                X_train_split, X_val, y_train_split, y_val = train_test_split(
                    X_train, y_train, test_size=0.2, random_state=42
                )
                # 调用fit_with_adaptive_de方法进行自适应DE优化
                logger.info("开始FHDES_JFB_DE的自适应DE优化...")
                best_params = des.fit_with_adaptive_de(X_train_split, y_train_split, X_val, y_val, optimize_level='per_box')
                logger.info(f"自适应DE优化完成，验证集准确率: {best_params['val_accuracy']:.4f}, F1分数: {best_params['val_macro_f1']:.4f}")
            else:
                des.fit(self.X_desl, self.y_desl)
            # 测试集评估
            if name == ('Oracle' or 'single_best' or 'StaticSelection' or 'StackedClassifier'):
                y_pred_test = des.predict(self.X_test, self.y_test)
                y_pred_test_proba = des.predict_proba(self.X_test, self.y_test)[:, 1]
            else:
                y_pred_test = des.predict(self.X_test)
                y_pred_test_proba = np.array(des.predict_proba(self.X_test))[:, 1]

            # 检查是否存在 NaN 值
            if np.isnan(y_pred_test_proba).any():
                print(f"Warning: NaN values encountered in {name}. Replacing NaN with 0.")
                y_pred_test_proba = np.nan_to_num(y_pred_test_proba, nan=0.0)

            optimal_threshold = self.optimize_threshold(self.y_test, y_pred_test_proba, method='youden')  # 使用新阈值优化方法
            y_pred_test = (y_pred_test_proba >= optimal_threshold).astype(int)

            # # 打印混淆矩阵
            # print(f"{name} Confusion Matrix:")  # 混淆矩阵  
            # print(confusion_matrix(self.y_test, y_pred_test))
            # print(classification_report(self.y_test, y_pred_test))

            # 计算PR-AUC
            precision, recall, _ = precision_recall_curve(self.y_test, y_pred_test_proba)
            pr_auc = auc(recall, precision)
            
            scores = {
                'Method': name,
                'F1 (test)': f1_score(self.y_test, y_pred_test),
                'AUC (test)': roc_auc_score(self.y_test, y_pred_test_proba),
                'MCC (test)': matthews_corrcoef(self.y_test, y_pred_test),
                'GMean (test)': geometric_mean_score(self.y_test, y_pred_test),
                'balanced_accuracy (test)': balanced_accuracy_score(self.y_test, y_pred_test),
                'Recall_pos (test)': recall_score(self.y_test, y_pred_test, pos_label=1),
                'Precision_pos (test)': precision_score(self.y_test, y_pred_test, pos_label=1),
                'PR_AUC (test)': pr_auc
            }
            results.append(scores)

        headers = ['Method', 'F1 (test)', 'AUC (test)', 'MCC (test)', 'GMean (test)', 'balanced_accuracy (test)', 'Recall_pos (test)', 'Precision_pos (test)', 'PR_AUC (test)']
        print(tabulate(results, headers="keys", tablefmt='grid', floatfmt=".3f"))

        return results

if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    num_folds = 5
    overall_results = []
    
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
    #                 }

    # # HE-自动化参数优化
    # models = {
    #         'NB': partial(GaussianNB, var_smoothing=9.891364113904856e-07),
    #         'KNN': partial(KNeighborsClassifier, algorithm='auto', n_neighbors=30, weights='distance', p=2),
    #         'LR': partial(LogisticRegression, penalty='l2', solver='lbfgs', C=1.2590047620987799, class_weight=None, tol=0.0001521268430562923, max_iter=414, random_state=42),
    #         'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
    #         'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.176622018898312, tol=0.00011947264239668641),
    #         'RF': partial(RandomForestClassifier, n_estimators=500, max_depth=30, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', criterion='entropy', class_weight='balanced', random_state=42),
    #         'ET': partial(ExtraTreesClassifier, n_estimators=300, max_depth=16, min_samples_leaf=2, min_samples_split=28, max_features=0.1, bootstrap=False, class_weight=None, criterion='gini', random_state=42),
    #         'ADA': partial(AdaBoostClassifier, learning_rate=0.05601888314861266, n_estimators=200, algorithm='SAMME.R', random_state=42),
    #         'GBC': partial(GradientBoostingClassifier, learning_rate=0.01866217122968749, n_estimators=100, max_depth=5, max_features='log2', subsample=0.669171223494975, min_samples_split=3, min_samples_leaf=10, random_state=42),
    #         'LGBM': partial(LGBMClassifier, learning_rate=0.03924051936374861, n_estimators=450, num_leaves=10, class_weight='balanced', boosting_type='dart', reg_alpha=0.2782798249640321, reg_lambda=0.13171133143615316, random_state=42),
    #         'XGB': partial(XGBClassifier, max_depth=4, eta=0.07357270835293087, reg_alpha=1.4149633530419305, reg_lambda=0.08117306537418037, random_state=42),   
    #     }

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
    
    # PD-自动参数调参
    models = {
        'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
        'KNN': partial(KNeighborsClassifier, algorithm='ball_tree', leaf_size=19, n_neighbors=30, weights='distance', p=1),
        'LR': partial(LogisticRegression, penalty='l2', solver='newton-cholesky', C=7.953052660574928, class_weight=None, tol=0.0013519206467406972, max_iter=217, random_state=42),
        'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
        'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.20659869635343953, tol=0.00015393640132252336),
        'RF': partial(RandomForestClassifier, n_estimators=450, max_depth=5, min_samples_split=15, min_samples_leaf=4, max_features=None, criterion='entropy', class_weight='balanced', random_state=42),        
        'ET': partial(ExtraTreesClassifier, n_estimators=250, max_depth=5, min_samples_leaf=10, min_samples_split=24, max_features=1.0, bootstrap=False, class_weight='balanced', criterion='entropy', random_state=42),
        'ADA': partial(AdaBoostClassifier, learning_rate=0.021438723017596622, n_estimators=100, algorithm='SAMME.R', random_state=42),
        'GBC': partial(GradientBoostingClassifier, learning_rate=0.022782565517931837, n_estimators=300, max_depth=3, max_features='log2', subsample=0.6584846570294198, min_samples_split=6, min_samples_leaf=2, random_state=42),
        'LGBM': partial(LGBMClassifier, learning_rate=0.02541885003050532, n_estimators=350, num_leaves=35, class_weight='balanced', boosting_type='dart', reg_alpha=0.0014881309479933367, reg_lambda=0.9808097775040516, random_state=42),
        'XGB': partial(XGBClassifier, max_depth=3, eta=0.026304768011494003, reg_alpha=0.3693374576085647, reg_lambda=0.7993370406717103, random_state=42),
    }

    # # ACLF-自动调参
    # models = {
    #     'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
    #     'KNN': partial(KNeighborsClassifier, algorithm='auto', n_neighbors=30, weights='uniform', p=2),
    #     'LR': partial(LogisticRegression, penalty='l2', solver='newton-cg', C=0.09463470489960597, class_weight='balanced', tol=0.00020193497214808254, max_iter=278, random_state=42),
    #     'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
    #     'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.6557768306823545, tol=0.002047147040324709),
    #     'RF': partial(RandomForestClassifier, n_estimators=350, max_depth=10, min_samples_split=19, min_samples_leaf=3, max_features='sqrt', criterion='entropy', class_weight=None, random_state=42),
    #     'ET': partial(ExtraTreesClassifier, n_estimators=100, max_depth=10, min_samples_leaf=5, min_samples_split=9, max_features=0.30000000000000004, bootstrap=False, class_weight=None, criterion='gini', random_state=42),
    #     'ADA': partial(AdaBoostClassifier, learning_rate=0.6326195773451885, n_estimators=250, algorithm='SAMME', random_state=42),
    #     'GBC': partial(GradientBoostingClassifier, learning_rate=0.013119940157366297, n_estimators=300, max_depth=4, max_features='log2', subsample=0.6955857826691667, min_samples_split=4, min_samples_leaf=9, random_state=42),
    #     'LGBM': partial(LGBMClassifier, learning_rate=0.047584916773550485, n_estimators=100, num_leaves=10, class_weight=None, boosting_type='gbdt', reg_alpha=0.016498520716057563, reg_lambda=0.008638475783333249, random_state=42),
    #     'XGB': partial(XGBClassifier, max_depth=4, eta=0.07626889104749136, reg_alpha=0.05651492046926741, reg_lambda=2.6046398111143314, random_state=42),
    # }

    for fold in range(1, num_folds + 1):
        print(f"Processing fold {fold}...")

        # filtered_data/HE  filtered_data/ASCITES/enhance/default
        synthetic_datasets = load_datasets_from_folder(
            folder_path='filtered_data/PD',
            # _disease_attention_pca    /enhance/default：PD_final_train_fold_  PD_enhanced_gaussian_copula_fold_
            file_extension='.csv', prefix=f'PD_final_train_fold_{fold}')


        # 得到外部验证集
        test_set = pd.read_csv(f"filtered_externel_data/PD_external_preprocessed_0.4-0.1.csv")
        X_test = test_set.iloc[:, :-1].values  # 转换为numpy数组
        y_test = test_set.iloc[:, -1].values    # 转换为numpy数组

        # 得到动态选择集
        desl_set = pd.read_csv(f'filtered_data/PD/PD_final_desl_fold_{fold}.csv')
        X_desl = desl_set.iloc[:, :-1].values   # 转换为numpy数组
        y_desl = desl_set.iloc[:, -1].values     # 转换为numpy数组
        
        evaluator = DESModelEvaluator(models, synthetic_datasets, X_desl,y_desl, X_test, y_test, diversity_threshold=1)
        evaluator.generate_classifier_pools(models, top_k=7, random_state=42)
        fold_results = evaluator.evaluate_methods(random_state=42)
        overall_results.append(fold_results)

    # 计算每种方法的平均结果
    avg_results = []
    # 获取所有方法名称
    methods = set()
    for result in overall_results[0]:
        methods.add(result['Method'])
    methods = list(methods)

    # 修改最终平均结果计算方式
    # 新增总平均计算，包含准确率、正样本召回率、正样本精准率和PR-AUC指标
    overall_avg_results = []
    for method in methods:
        f1_values = []
        auc_values = []
        mcc_values = []
        gmean_values = []
        balanced_accuracy_values = []
        recall_pos_values = []
        precision_pos_values = []
        pr_auc_values = []
        for fold_result in overall_results:
            for result in fold_result:
                if result['Method'] == method:
                    f1_values.append(result['F1 (test)'])
                    auc_values.append(result['AUC (test)'])
                    mcc_values.append(result['MCC (test)'])
                    gmean_values.append(result['GMean (test)'])
                    balanced_accuracy_values.append(result['balanced_accuracy (test)'])
                    recall_pos_values.append(result['Recall_pos (test)'])
                    precision_pos_values.append(result['Precision_pos (test)'])
                    pr_auc_values.append(result['PR_AUC (test)'])
        # 计算各指标的平均值和标准差
        avg_f1 = np.mean(f1_values)
        std_f1 = np.std(f1_values)
        avg_auc = np.mean(auc_values)
        std_auc = np.std(auc_values)
        avg_mcc = np.mean(mcc_values)
        std_mcc = np.std(mcc_values)
        avg_gmean = np.mean(gmean_values)
        std_gmean = np.std(gmean_values)
        avg_balanced_accuracy = np.mean(balanced_accuracy_values)
        std_balanced_accuracy = np.std(balanced_accuracy_values)
        avg_recall_pos = np.mean(recall_pos_values)
        std_recall_pos = np.std(recall_pos_values)
        avg_precision_pos = np.mean(precision_pos_values)
        std_precision_pos = np.std(precision_pos_values)
        avg_pr_auc = np.mean(pr_auc_values)
        std_pr_auc = np.std(pr_auc_values)
        
        # 格式化结果为平均值±标准差
        formatted_results = {
            'Method': method,
            'F1 (test)': f"{avg_f1:.3f}±{std_f1:.3f}",
            'AUC (test)': f"{avg_auc:.3f}±{std_auc:.3f}",
            'MCC (test)': f"{avg_mcc:.3f}±{std_mcc:.3f}",
            'GMean (test)': f"{avg_gmean:.3f}±{std_gmean:.3f}",
            'balanced_accuracy (test)': f"{avg_balanced_accuracy:.3f}±{std_balanced_accuracy:.3f}",
            'Recall_pos (test)': f"{avg_recall_pos:.3f}±{std_recall_pos:.3f}",
            'Precision_pos (test)': f"{avg_precision_pos:.3f}±{std_precision_pos:.3f}",
            'PR_AUC (test)': f"{avg_pr_auc:.3f}±{std_pr_auc:.3f}"
        }
        overall_avg_results.append(formatted_results)

    print(f"\n{'='*30} 全局平均评估结果 {'='*30}")
    print(tabulate(overall_avg_results, headers="keys", tablefmt="grid"))
