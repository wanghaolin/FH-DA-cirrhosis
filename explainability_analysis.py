# ========================== 标准库 ==========================
import os
import time
import warnings
from functools import partial
from itertools import combinations
# ========================== 第三方数值/科学计算 ==========================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
# ========================== sklearn 基础工具 ==========================
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, auc, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve
)
# ========================== sklearn 分类器 ==========================
from sklearn.ensemble import (
    AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
)
# ========================== 高级集成/提升库 ==========================
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# ========================== 可解释性工具 ==========================
import shap
# ========================== 高级集成/提升库 ==========================
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# ========================== 不平衡学习 ==========================
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import (
    ADASYN, BorderlineSMOTE, SMOTE, SVMSMOTE
)
from imblearn.under_sampling import RandomUnderSampler
# ========================== 合成数据生成 ==========================
from sdv.metadata import Metadata
from sdv.single_table import (
    CopulaGANSynthesizer, CTGANSynthesizer, GaussianCopulaSynthesizer,
    TVAESynthesizer
)
# ========================== deslib 动态集成 ==========================
from deslib.dcs import APriori, APosteriori, LCA, MCB, MLA, OLA, Rank
from deslib.des import (
    DESClustering, DESKNN, DESMI, DESP, KNOP, KNORAE, KNORAU, METADES
)
# ========================== 自定义 DES/FH 系列 ==========================
from my_deslib.des import DESFH
from my_deslib.des.des_FHMW_AllBoxes_vector import DESFHMW_allboxes_vector
from my_deslib.des.des_FHMW_JFB_vector import DESFHMW_JFB_vector
from my_deslib.des.des_FHMW_prior_vector import DESFHMW_prior_vector
from my_deslib.des.fh_des_AllBoxes_vector import FHDES_Allboxes_vector
from my_deslib.des.fh_des_JFB_vector import FHDES_JFB_vector
from my_deslib.des.fh_des_prior_vector import FHDES_prior_vector
# -------------- 超椭球体版本 --------------
from my_deslib.des.hyperellipsoid_versions.des_FHMW_AllBoxes_vector_he import DESFHMW_allboxes_vector_he
from my_deslib.des.hyperellipsoid_versions.des_FHMW_JFB_vector_he import DESFHMW_JFB_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_AllBoxes_vector_he import FHDES_Allboxes_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he import FHDES_JFB_vector_he
# -------------- 球体版本 --------------
from my_deslib.des.sphere_versions.fh_des_AllBoxes_vector_sphere import FHDES_AllBoxes_vector_sphere
from my_deslib.des.sphere_versions.fh_des_JFB_vector_sphere import FHDES_JFB_vector_sphere
# -------------- 矩形版本 --------------
from my_deslib.des.rectangle_versions.fh_des_AllBoxes_vector_rectangle import FHDES_AllBoxes_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_JFB_vector_rectangle import FHDES_JFB_vector_rectangle
# -------------- 聚类扩展版本 --------------
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
# ========================== 辅助函数/工具 ==========================
from tabulate import tabulate
from sklearn.metrics import pairwise_distances

# ========================== 日志配置 ==========================
import logging
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置matplotlib字体，支持中文显示
# 尝试多种常用中文字体
plt.rcParams['font.family'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Times New Roman']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 12  # 设置默认字体大小

# 调试：打印当前使用的字体
print("Matplotlib font settings:")
print(f"Font family: {plt.rcParams['font.family']}")
print(f"Axes unicode minus: {plt.rcParams['axes.unicode_minus']}")

# 确保pictures目录存在
os.makedirs('pictures', exist_ok=True)


def load_datasets_from_folder(folder_path, file_extension=".csv", prefix="ADASYN_"):
    """加载数据集，不划分验证集"""
    train_datasets = []
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(file_extension):
            file_path = os.path.join(folder_path, filename)
            # 尝试不同编码来解决解码错误
            try:
                data = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    data = pd.read_csv(file_path, encoding='gbk')
                except UnicodeDecodeError:
                    try:
                        data = pd.read_csv(file_path, encoding='latin1')
                    except Exception as e:
                        print(f"无法读取文件 {file_path}: {e}")
                        continue
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            train_datasets.append((X, y))
    return train_datasets


class FHDAExplainabilityAnalyzer:
    def __init__(self, models, train_datasets, X_desl, y_desl, X_test, y_test, feature_names=None):
        self.models = models
        self.train_datasets = train_datasets
        self.X_desl = X_desl
        self.y_desl = y_desl
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names if feature_names is not None else [f'Feature_{i}' for i in range(X_test.shape[1])]
        self.classifier_pools = None
        self.method_results = {}
        self.feature_importance_results = {}
        
        # 确保X_test是pandas DataFrame，便于SHAP分析
        if isinstance(self.X_test, np.ndarray):
            self.X_test = pd.DataFrame(self.X_test, columns=self.feature_names)
        if isinstance(self.X_desl, np.ndarray):
            self.X_desl = pd.DataFrame(self.X_desl, columns=self.feature_names)
    
    def optimize_threshold(self, y_true, y_pred_proba, method='youden'):
        """
        根据不同方法优化分类阈值
        
        参数:
            y_true: 真实标签
            y_pred_proba: 预测概率
            method: 优化方法，可选 'youden', 'f1', 'precision', 'recall'
            
        返回:
            最优阈值
        """
        if method == 'youden':
            # Youden指数法
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden = tpr - fpr
            optimal_idx = np.argmax(youden)
            return thresholds[optimal_idx]
        elif method == 'f1':
            # F1分数最大化
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx]
        elif method == 'precision':
            # 精确率最大化
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(precision)
            return thresholds[optimal_idx]
        elif method == 'recall':
            # 召回率最大化
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(recall)
            return thresholds[optimal_idx]
        else:
            # 默认使用0.5
            return 0.5
    
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
                
                # 确保返回带有列名的DataFrame
                if isinstance(X, pd.DataFrame):
                    return X_combined, y_combined
                else:
                    return X_combined.values, y_combined.values
            
            # 执行传统的过采样方法
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            print(f"  {method}生成增强后 - 多数类: {sum(y_resampled == majority_class)}, 少数类: {sum(y_resampled == minority_class)}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"  {method}生成增强时出错: {str(e)}")
            return X, y
    
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
    
    def generate_classifier_pools(self, topk=7, random_state=42, fold=1):
        """
        生成基分类器池，通过条件欠采样和多种数据增强方法缓解类别不平衡问题
        
        参数:
            topk: 选择的top AUC分类器数量
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
        for model_name, model_cls in self.models.items():
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
                    
                    # 确保如果原始数据是DataFrame，加载的数据也转换为DataFrame并保留列名
                    if isinstance(undersampled_X, pd.DataFrame):
                        augmented_X = pd.DataFrame(augmented_X, columns=undersampled_X.columns)
                else:
                    print(f"    生成并保存增强数据: {enhance_data_file}")
                    # 执行数据增强
                    augmented_X, augmented_y = self._generate_minority_samples(
                        undersampled_X, undersampled_y, method=aug_method, 
                        minority_ratio=0.3, random_state=random_state)
                    # 保存增强数据
                    np.savez(enhance_data_file, augmented_X=augmented_X, augmented_y=augmented_y)
                
                # 对每个分类器在增强数据上训练
                for model_name, model_cls in self.models.items():
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
        selected_models_info = all_models_info[:topk] if len(all_models_info) > topk else all_models_info
        
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
        
        # 计算模型多样性统计
        self._calculate_diversity_stats(selected_models_info)
        
        return self.classifier_pools
    
    def train_and_evaluate_method(self, method_name, method_cls, method_params):
        """训练并评估指定的动态选择方法"""
        print(f"\n{'='*60}")
        print(f"训练并评估方法: {method_name}")
        print(f"{'='*60}")
        print(f"DEBUG: 分类器池大小: {len(self.classifier_pools)}")
        print(f"DEBUG: 方法类: {method_cls}")
        print(f"DEBUG: 方法参数: {method_params}")
        
        # 合并默认参数和方法特定参数
        params = {
            'pool_classifiers': self.classifier_pools,
            'random_state': 42,
            **method_params
        }
        
        # 初始化动态选择方法
        des = method_cls(**params)
        
        # 针对不同方法使用不同的训练逻辑
        X_train, y_train = self.train_datasets[0]
        if method_name == "AdaptiveFHDE_he":
            des.fit(self.X_desl, self.y_desl, X_train, y_train, self.X_test, self.y_test)
        elif method_name == "FHDES_JFB_DE":
            # 为FHDES_JFB_DE使用特殊的fit_with_adaptive_de方法进行参数优化
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
        
        # 预测和概率获取
        if method_name in ['Oracle', 'single_best', 'StaticSelection', 'StackedClassifier']:
            y_pred_test = des.predict(self.X_test, self.y_test)
            y_pred_test_proba = des.predict_proba(self.X_test, self.y_test)[:, 1]
        else:
            y_pred_test = des.predict(self.X_test)
            y_pred_test_proba = np.array(des.predict_proba(self.X_test))[:, 1]
        
        # 检查是否存在 NaN 值
        if np.isnan(y_pred_test_proba).any():
            print(f"Warning: NaN values encountered in {method_name}. Replacing NaN with 0.")
            y_pred_test_proba = np.nan_to_num(y_pred_test_proba, nan=0.0)
        
        # 使用youden指数优化阈值
        optimal_threshold = self.optimize_threshold(self.y_test, y_pred_test_proba, method='youden')
        y_pred_test = (y_pred_test_proba >= optimal_threshold).astype(int)
        
        # 计算PR-AUC
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_test_proba)
        pr_auc = auc(recall, precision)
        
        # 计算评估指标
        scores = {
            'Method': method_name,
            'F1 (test)': f1_score(self.y_test, y_pred_test),
            'AUC (test)': roc_auc_score(self.y_test, y_pred_test_proba),
            'MCC (test)': matthews_corrcoef(self.y_test, y_pred_test),
            'GMean (test)': geometric_mean_score(self.y_test, y_pred_test),
            'balanced_accuracy (test)': balanced_accuracy_score(self.y_test, y_pred_test),
            'Recall_pos (test)': recall_score(self.y_test, y_pred_test, pos_label=1, zero_division=0),
            'Precision_pos (test)': precision_score(self.y_test, y_pred_test, pos_label=1, zero_division=0),
            'PR_AUC (test)': pr_auc
        }
        
        print(tabulate([scores], headers="keys", tablefmt='grid', floatfmt=".3f"))
        self.method_results[method_name] = {
            'model': des,
            'scores': scores,
            'y_pred_test': y_pred_test,
            'y_pred_test_proba': y_pred_test_proba
        }
        
        print(f"DEBUG: {method_name} 训练完成，已保存到 method_results")
        print(f"DEBUG: method_results 当前包含: {list(self.method_results.keys())}")
        
        return des
    
    def calculate_permutation_importance(self, method_name, n_repeats=10, random_state=42):
        """计算排列特征重要性"""
        print(f"\n{'='*60}")
        print(f"计算 {method_name} 的排列特征重要性")
        print(f"{'='*60}")
        
        if method_name not in self.method_results:
            print(f"方法 {method_name} 尚未训练，无法计算特征重要性")
            return
        
        des = self.method_results[method_name]['model']
        
        # 确保X_test是numpy数组，避免pandas索引问题
        X_test_np = self.X_test.values if isinstance(self.X_test, pd.DataFrame) else self.X_test
        
        # 获取原始性能
        try:
            if hasattr(des, 'predict_proba'):
                y_pred_proba = des.predict_proba(X_test_np)[:, 1]
            elif hasattr(des, 'decision_function'):
                y_pred_proba = des.decision_function(X_test_np)
            else:
                raise AttributeError("Model doesn't have predict_proba or decision_function method")
            
            # 检查预测结果是否有效
            is_all_zero = np.allclose(y_pred_proba, 0)
            is_constant = np.all(y_pred_proba == y_pred_proba[0])
            
            if is_all_zero or is_constant:
                status = "全为0" if is_all_zero else "所有值相同"
                print(f"警告：模型预测结果无效，所有预测值{status}")
                return
            
            # 计算原始AUC
            original_auc = roc_auc_score(self.y_test, y_pred_proba)
            print(f"原始 AUC: {original_auc:.4f}")
            print(f"预测值范围: [{np.min(y_pred_proba):.4f}, {np.max(y_pred_proba):.4f}]")
            print(f"预测值标准差: {np.std(y_pred_proba):.4f}")
        except Exception as e:
            print(f"获取预测结果时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return
        
        # 初始化重要性结果
        n_features = X_test_np.shape[1]
        importances = np.zeros((n_repeats, n_features))
        rng = np.random.RandomState(random_state)
        
        # 对每个特征计算排列重要性
        for i in range(n_features):
            print(f"  正在计算特征 {self.feature_names[i]} 的重要性...")
            for j in range(n_repeats):
                # 复制数据
                X_permuted = X_test_np.copy()
                # 排列特征
                X_permuted[:, i] = rng.permutation(X_permuted[:, i])
                
                # 预测
                if hasattr(des, 'predict_proba'):
                    y_pred_proba_permuted = des.predict_proba(X_permuted)[:, 1]
                elif hasattr(des, 'decision_function'):
                    y_pred_proba_permuted = des.decision_function(X_permuted)
                else:
                    raise AttributeError("Model doesn't have predict_proba or decision_function method")
                
                # 计算性能变化
                permuted_auc = roc_auc_score(self.y_test, y_pred_proba_permuted)
                importances[j, i] = original_auc - permuted_auc  # 重要性 = 原始性能 - 排列后性能
        
        # 计算平均重要性和标准差
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        
        # 创建结果对象，与scikit-learn的permutation_importance结果格式一致
        result = type('obj', (object,), {
            'importances_mean': importances_mean,
            'importances_std': importances_std,
            'importances': importances
        })
        
        # 整理结果
        sorted_importances_idx = importances_mean.argsort()[::-1]
        importance_df = pd.DataFrame(
            importances[:, sorted_importances_idx],
            columns=[self.feature_names[idx] for idx in sorted_importances_idx]
        )
        
        # 保存结果
        if method_name not in self.feature_importance_results:
            self.feature_importance_results[method_name] = {}
        self.feature_importance_results[method_name]['permutation_importance'] = result
        self.feature_importance_results[method_name]['importance_df'] = importance_df
        
        # 可视化
        plt.figure(figsize=(12, 8))
        plt.title(f"{method_name} - Permutation Feature Importance", fontsize=14, fontweight='bold')
        sns.boxplot(data=importance_df, orient="h")
        plt.xlabel("Importance (AUC Reduction)", fontsize=12)
        plt.ylabel("Features", fontsize=12)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig(f"pictures/{method_name}_permutation_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 打印前10个重要特征
        print("\n前10个重要特征:")
        top_10_features = [self.feature_names[idx] for idx in sorted_importances_idx[:10]]
        top_10_importances = importances_mean[sorted_importances_idx][:10]
        top_10_stds = importances_std[sorted_importances_idx][:10]
        for i, (feature, importance, std) in enumerate(zip(top_10_features, top_10_importances, top_10_stds)):
            print(f"{i+1}. {feature}: {importance:.4f} ± {std:.4f}")
    
    def calculate_shap_importance(self, method_name, sample_size=100, shap_method='classifier_wise'):
        """使用SHAP计算特征重要性
        
        参数:
            method_name: 方法名称
            sample_size: 采样大小
            shap_method: SHAP计算方法，可选值:
                - 'classifier_wise': 分classifier解释 + 聚合（默认）
                - 'black_box': 将整个DES封装为黑盒函数 + 使用KernelExplainer
        """
        print(f"\n{'='*60}")
        print(f"计算 {method_name} 的SHAP特征重要性，使用方法: {shap_method}")
        print(f"{'='*60}")
        
        if method_name not in self.method_results:
            print(f"方法 {method_name} 尚未训练，无法计算SHAP重要性")
            return
        
        des = self.method_results[method_name]['model']
        
        # 由于动态集成模型可能不直接支持SHAP，我们将使用其所有基分类器的平均SHAP值
        # 并根据基分类器类型选择合适的explainer
        
        try:
            # 采样数据以加速计算
            if len(self.X_test) > sample_size:
                X_sample = self.X_test.sample(n=sample_size, random_state=42)
            else:
                X_sample = self.X_test
            
            # 转换为numpy数组以兼容所有explainer类型
            X_sample_np = X_sample.values
            
            if shap_method == 'classifier_wise':
                # 方法1: 分classifier解释 + 聚合
                print("  使用方法1: 分classifier解释 + 聚合")
                
                # 收集所有基分类器的SHAP值
                all_shap_values = []
                valid_models_count = 0
                expected_features = len(self.feature_names)
                
                for i, base_model in enumerate(self.classifier_pools):
                    print(f"  正在处理基分类器 {i+1}/{len(self.classifier_pools)}...")
                    
                    try:
                        shap_vals = None
                        
                        if hasattr(base_model, 'feature_importances_') and not hasattr(base_model, 'estimators_'):
                            # 使用TreeExplainer（适用于单一树模型，不适用于集成模型如AdaBoost）
                            try:
                                explainer = shap.TreeExplainer(base_model)
                                shap_values = explainer(X_sample_np)
                                
                                # 处理树模型的SHAP值（可能是3D数组）
                                if hasattr(shap_values, 'values'):
                                    if shap_values.values.ndim == 3:
                                        # 对于多输出模型，取第二个类（正类）的SHAP值
                                        shap_vals = shap_values.values[:, :, 1]
                                    elif shap_values.values.ndim == 2:
                                        # 对于单输出模型，直接使用SHAP值
                                        shap_vals = shap_values.values
                                    else:
                                        # 其他维度情况，尝试展平到2D
                                        shap_vals = np.squeeze(shap_values.values)
                                        if shap_vals.ndim == 1:
                                            shap_vals = shap_vals.reshape(-1, 1)
                            except Exception as te:
                                print(f"    TreeExplainer不支持此模型，尝试使用KernelExplainer: {str(te)}")
                                # 对于不支持的树模型（如AdaBoost），改用KernelExplainer
                                shap_vals = None
                        
                        elif hasattr(base_model, 'coef_'):
                            # 使用LinearExplainer（适用于线性模型）
                            explainer = shap.LinearExplainer(base_model, X_sample_np)
                            shap_values = explainer(X_sample_np)
                            
                            if hasattr(shap_values, 'values'):
                                shap_vals = shap_values.values
                                # 确保SHAP值是2D数组
                                if shap_vals.ndim == 1:
                                    shap_vals = shap_vals.reshape(-1, 1)
                        
                        else:
                            # 使用KernelExplainer（适用于其他模型）
                            # 移除使用限制，确保有足够的有效SHAP值
                            # 确保模型有predict_proba或decision_function方法
                            if hasattr(base_model, 'predict_proba'):
                                predict_fn = lambda x: base_model.predict_proba(x)[:, 1]
                            elif hasattr(base_model, 'decision_function'):
                                predict_fn = lambda x: base_model.decision_function(x)
                            else:
                                print(f"    基分类器 {i+1} 没有predict_proba或decision_function方法，跳过")
                                continue
                                
                            try:
                                explainer = shap.KernelExplainer(
                                    predict_fn,
                                    shap.sample(X_sample_np, 10)  # 使用10个样本作为背景
                                )
                                # KernelExplainer的__call__方法不接受nsamples参数，移除该参数
                                shap_values = explainer(X_sample_np)
                                
                                if hasattr(shap_values, 'values'):
                                    shap_vals = shap_values.values
                                else:
                                    shap_vals = shap_values
                                    
                                # 确保SHAP值是2D数组
                                if shap_vals.ndim == 1:
                                    shap_vals = shap_vals.reshape(-1, 1)
                            except Exception as ke:
                                print(f"    KernelExplainer处理失败: {str(ke)}")
                                continue
                            
                        if shap_vals is not None:
                            # 强制调整SHAP值的形状，确保与特征数量一致
                            if shap_vals.ndim == 1:
                                # 1D数组转为2D (n_samples, 1)
                                shap_vals = shap_vals.reshape(-1, 1)
                            
                            # 确保特征数量正确
                            if shap_vals.shape[1] < expected_features:
                                # 填充缺失的特征维度
                                padding = np.zeros((shap_vals.shape[0], expected_features - shap_vals.shape[1]))
                                shap_vals = np.hstack([shap_vals, padding])
                            elif shap_vals.shape[1] > expected_features:
                                # 截断多余的特征维度
                                shap_vals = shap_vals[:, :expected_features]
                            
                            # 验证SHAP值是否有效：不为全0且不是所有值相同
                            is_all_zero = np.allclose(shap_vals, 0)
                            is_all_same = np.all(shap_vals == shap_vals[0, 0]) if shap_vals.size > 0 else True
                            
                            if not is_all_zero and not is_all_same:
                                all_shap_values.append(shap_vals)
                                valid_models_count += 1
                                print(f"    基分类器 {i+1} SHAP分析成功")
                                print(f"        SHAP值形状: {shap_vals.shape}, 非零值数量: {np.count_nonzero(shap_vals)}")
                            else:
                                status = "全为0" if is_all_zero else "所有值相同"
                                print(f"    基分类器 {i+1} SHAP值{status}，跳过")
                            
                    except Exception as e:
                        print(f"    基分类器 {i+1} SHAP分析失败: {str(e)}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                if not all_shap_values:
                    print("所有基分类器的SHAP分析都失败了，跳过此方法")
                    return
                
                print(f"  成功处理 {valid_models_count} 个基分类器")
                print(f"  SHAP值数组形状: {[sv.shape for sv in all_shap_values[:3]]}...")
                
                # 获取DES的选择结果或权重
                print("  获取DES的选择结果或权重...")
                try:
                    # 优先使用estimate_competence方法获取竞争力分数作为权重（严格遵循FH-DES模型的选择规则）
                    if hasattr(des, 'estimate_competence'):
                        print("  优先使用estimate_competence方法获取竞争力分数作为权重...")
                        competences = des.estimate_competence(X_sample_np)
                        print(f"  成功获取竞争力分数，类型: {type(competences).__name__}")
                        
                        if hasattr(competences, 'shape'):
                            print(f"  竞争力分数形状: {competences.shape}")
                        
                        # 确保competences是2D数组，每行对应一个样本的竞争力分数向量
                        if hasattr(competences, 'shape'):
                            if competences.ndim == 1:
                                # 如果是1D数组，转换为2D (n_samples, n_classifiers)
                                # 这种情况通常是所有样本使用相同的权重
                                competences = np.tile(competences, (X_sample_np.shape[0], 1))
                            elif competences.ndim > 2:
                                # 其他形状，尝试展平到2D
                                competences = competences.reshape(X_sample_np.shape[0], -1)
                        else:
                            # 列表形式，确保长度与样本数一致
                            if len(competences) != X_sample_np.shape[0]:
                                # 如果长度不一致，复制第一个元素
                                competences = [competences[0]] * X_sample_np.shape[0]
                        
                        # 使用竞争力分数聚合SHAP值
                        mean_shap_values = np.zeros((X_sample_np.shape[0], expected_features))
                        for i in range(X_sample_np.shape[0]):
                            try:
                                # 获取当前样本的竞争力分数
                                if hasattr(competences, 'shape'):
                                    # 数组形式
                                    sample_weights = competences[i]
                                else:
                                    # 列表形式
                                    sample_weights = competences[i]
                                
                                # 确保权重长度与有效分类器数量一致
                                if len(sample_weights) > len(all_shap_values):
                                    sample_weights = sample_weights[:len(all_shap_values)]
                                elif len(sample_weights) < len(all_shap_values):
                                    # 补全权重，使用默认值1.0
                                    sample_weights = np.pad(sample_weights, (0, len(all_shap_values) - len(sample_weights)), 'constant', constant_values=1.0)
                                
                                # 归一化权重，严格遵循DES模型的normalize_weights参数
                                # 检查DES模型是否已经对竞争力分数进行了归一化
                                if hasattr(des, 'normalize_weights') and des.normalize_weights:
                                    # 如果DES模型已经归一化（使用MinMaxScaler），只需要处理零和情况
                                    if np.sum(sample_weights) == 0:
                                        sample_weights = np.ones(len(all_shap_values)) / len(all_shap_values)
                                    # 保持DES模型归一化后的结果，不进行额外归一化
                                else:
                                    # 如果DES模型没有归一化，我们需要进行归一化
                                    if np.sum(sample_weights) == 0:
                                        sample_weights = np.ones(len(all_shap_values)) / len(all_shap_values)
                                    else:
                                        sample_weights = sample_weights / np.sum(sample_weights)
                                
                                # 使用权重计算加权平均
                                sample_shap = np.average([all_shap_values[j][i] for j in range(len(all_shap_values))], 
                                                       weights=sample_weights, axis=0)
                                mean_shap_values[i] = sample_shap
                            except Exception as sample_e:
                                print(f"  使用竞争力分数处理样本 {i} 时出错: {str(sample_e)}")
                                # 使用所有有效分类器的平均值作为备选方案
                                sample_shap = np.mean([all_shap_values[j][i] for j in range(len(all_shap_values))], axis=0)
                                mean_shap_values[i] = sample_shap
                    # 尝试获取每个样本的选择结果
                    elif hasattr(des, 'select'):
                        # 如果DES模型有属性，使用它
                        selected_indices = des.select
                        print(f"  成功获取，类型: {type(selected_indices).__name__}")
                    
                        # 处理不同形状的
                        if hasattr(selected_indices, 'shape'):
                            print(f"  形状: {selected_indices.shape}")
                        else:
                            print(f"  selected_indices元素类型: {type(selected_indices[0]).__name__}" if selected_indices else "selected_indices为空")
                        
                        # 使用选择结果聚合SHAP值
                        mean_shap_values = np.zeros((X_sample_np.shape[0], expected_features))
                        for i in range(X_sample_np.shape[0]):
                            try:
                                if hasattr(selected_indices, 'shape'):
                                    # 对于数组形式的selected_indices
                                    if selected_indices.ndim == 2:
                                        # 每行是一个样本的选择结果
                                        sample_indices = selected_indices[i]
                                        # 过滤掉无效索引
                                        sample_indices = [idx for idx in sample_indices if idx >= 0 and idx < len(all_shap_values)]
                                    elif selected_indices.ndim == 1:
                                        # 单一索引情况（每个样本选择一个分类器）
                                        sample_indices = [selected_indices[i]] if 0 <= selected_indices[i] < len(all_shap_values) else []
                                    else:
                                        # 其他形状，使用所有有效分类器的平均值
                                        sample_indices = []
                                else:
                                    # 对于列表形式的selected_indices
                                    sample_indices = selected_indices[i] if i < len(selected_indices) else []
                                    sample_indices = [idx for idx in sample_indices if idx >= 0 and idx < len(all_shap_values)]
                                
                                # 检查sample_indices是否为空
                                if len(sample_indices) == 0:
                                    # 如果没有选择任何分类器，使用所有有效分类器的平均值
                                    sample_shap = np.mean([all_shap_values[j][i] for j in range(len(all_shap_values))], axis=0)
                                else:
                                    # 使用选择的分类器的SHAP值的平均值
                                    sample_shap = np.mean([all_shap_values[j][i] for j in sample_indices], axis=0)
                                mean_shap_values[i] = sample_shap
                            except Exception as sample_e:
                                print(f"  处理样本 {i} 时出错: {str(sample_e)}")
                                # 使用所有有效分类器的平均值作为备选方案
                                sample_shap = np.mean([all_shap_values[j][i] for j in range(len(all_shap_values))], axis=0)
                                mean_shap_values[i] = sample_shap
                    elif hasattr(des, 'weights_'):
                        # 如果DES模型有权重属性，使用权重聚合SHAP值
                        weights = des.weights_
                        print(f"  成功获取weights_，类型: {type(weights).__name__}")
                        
                        if hasattr(weights, 'shape'):
                            print(f"  weights_形状: {weights.shape}")
                        
                        # 使用权重聚合SHAP值
                        mean_shap_values = np.zeros((X_sample_np.shape[0], expected_features))
                        for i in range(X_sample_np.shape[0]):
                            try:
                                if hasattr(weights, 'shape'):
                                    if weights.ndim == 2:
                                        # 每行是一个样本的权重向量
                                        sample_weights = weights[i] if i < weights.shape[0] else np.ones(len(all_shap_values))
                                    elif weights.ndim == 1:
                                        # 单一权重情况
                                        sample_weights = weights if len(weights) == len(all_shap_values) else np.ones(len(all_shap_values))
                                    else:
                                        # 其他形状
                                        sample_weights = np.ones(len(all_shap_values))
                                else:
                                    # 列表形式
                                    sample_weights = weights[i] if i < len(weights) else np.ones(len(all_shap_values))
                                
                                # 确保权重长度与有效分类器数量一致
                                if len(sample_weights) > len(all_shap_values):
                                    sample_weights = sample_weights[:len(all_shap_values)]
                                elif len(sample_weights) < len(all_shap_values):
                                    # 补全权重
                                    sample_weights = np.pad(sample_weights, (0, len(all_shap_values) - len(sample_weights)), 'constant', constant_values=1.0)
                                
                                # 归一化权重
                                if np.sum(sample_weights) == 0:
                                    sample_weights = np.ones(len(all_shap_values)) / len(all_shap_values)
                                else:
                                    sample_weights = sample_weights / np.sum(sample_weights)
                                
                                # 使用权重计算加权平均
                                sample_shap = np.average([all_shap_values[j][i] for j in range(len(all_shap_values))], 
                                                       weights=sample_weights, axis=0)
                                mean_shap_values[i] = sample_shap
                            except Exception as sample_e:
                                print(f"  使用权重处理样本 {i} 时出错: {str(sample_e)}")
                                # 使用所有有效分类器的平均值作为备选方案
                                sample_shap = np.mean([all_shap_values[j][i] for j in range(len(all_shap_values))], axis=0)
                                mean_shap_values[i] = sample_shap
                    else:
                        # 如果没有选择结果或权重属性，尝试获取每个样本的选择结果
                        # 对于某些DES模型，可能需要先对样本进行预测才能生成选择结果
                        print("  无法直接获取选择结果或权重，尝试生成选择结果...")
                        
                        # 尝试让DES模型对样本进行预测以生成选择结果
                        if hasattr(des, 'predict'):
                            try:
                                # 预测样本以触发选择过程
                                _ = des.predict(X_sample_np)
                                
                                # 再次检查是否生成了选择结果
                                if hasattr(des, 'select'):
                                    # 如果现在有了属性，使用它
                                    selected_indices = des.select
                                    print(f"  预测后成功获取，类型: {type(selected_indices).__name__}")
                                    
                                    # 再次调用自己处理这种情况
                                    # 这里使用递归逻辑，避免重复代码
                                    if hasattr(selected_indices, 'shape'):
                                        print(f"  形状: {selected_indices.shape}")
                                        
                                        # 使用选择结果聚合SHAP值
                                        mean_shap_values = np.zeros((X_sample_np.shape[0], expected_features))
                                        for i in range(X_sample_np.shape[0]):
                                            try:
                                                if selected_indices.ndim == 2:
                                                    # 每行是一个样本的选择结果
                                                    sample_indices = selected_indices[i]
                                                    sample_indices = [idx for idx in sample_indices if idx >= 0 and idx < len(all_shap_values)]
                                                elif selected_indices.ndim == 1:
                                                    # 单一索引情况
                                                    sample_indices = [selected_indices[i]] if 0 <= selected_indices[i] < len(all_shap_values) else []
                                                else:
                                                    sample_indices = []
                                                
                                                if len(sample_indices) == 0:
                                                    sample_shap = np.mean([all_shap_values[j][i] for j in range(len(all_shap_values))], axis=0)
                                                else:
                                                    sample_shap = np.mean([all_shap_values[j][i] for j in sample_indices], axis=0)
                                                mean_shap_values[i] = sample_shap
                                            except Exception as sample_e:
                                                print(f"  处理样本 {i} 时出错: {str(sample_e)}")
                                                sample_shap = np.mean([all_shap_values[j][i] for j in range(len(all_shap_values))], axis=0)
                                                mean_shap_values[i] = sample_shap
                                    else:
                                        # 非数组形式，使用默认方法
                                        print("  非数组形式，使用默认方法")
                                        mean_shap_values = np.mean(all_shap_values, axis=0)
                                else:
                                    # 如果预测后仍然没有选择结果，使用所有基分类器的平均值
                                    print("  预测后仍然无法获取选择结果，使用所有基分类器的平均值")
                                    mean_shap_values = np.mean(all_shap_values, axis=0)
                            except Exception as pred_e:
                                print(f"  预测样本时出错: {str(pred_e)}")
                                # 使用所有基分类器的平均值作为备选方案
                                mean_shap_values = np.mean(all_shap_values, axis=0)
                        else:
                            # 如果模型没有predict方法，使用所有基分类器的平均值
                            print("  模型没有predict方法，使用所有基分类器的平均值")
                            mean_shap_values = np.mean(all_shap_values, axis=0)
                except Exception as e:
                    print(f"  获取选择结果或权重时出错: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 使用所有基分类器的平均值作为备选方案
                    mean_shap_values = np.mean(all_shap_values, axis=0)
                    
                print(f"  聚合后SHAP值形状: {mean_shap_values.shape}")
            
            elif shap_method == 'black_box':
                # 方法2: 将整个DES封装为黑盒函数 + 使用KernelExplainer
                print("  使用方法2: 将整个DES封装为黑盒函数 + 使用KernelExplainer")
                
                # 定义黑盒函数：f(X) -> predicted_prob(class 1)
                def black_box_fn(X):
                    if hasattr(des, 'predict_proba'):
                        return des.predict_proba(X)[:, 1]
                    elif hasattr(des, 'decision_function'):
                        return des.decision_function(X)
                    else:
                        raise AttributeError("Model doesn't have predict_proba or decision_function method")
                
                # 使用KernelExplainer（model-agnostic）
                print("  初始化KernelExplainer...")
                try:
                    print(f"  X_sample_np形状: {X_sample_np.shape}")
                    # 选择背景数据集
                    background = shap.sample(X_sample_np, 10)  # 使用10个样本作为背景
                    print(f"  背景数据形状: {background.shape}")
                    
                    # 测试黑盒函数是否正常工作
                    print("  测试黑盒函数...")
                    test_pred = black_box_fn(X_sample_np[:2])
                    print(f"  黑盒函数测试结果: {test_pred}")
                    
                    # 尝试使用shap.Explainer（通用接口）
                    try:
                        print("  尝试使用shap.Explainer...")
                        explainer = shap.Explainer(black_box_fn, background)
                        shap_values = explainer(X_sample_np)
                        print("  shap.Explainer成功")
                    except Exception as e:
                        print(f"  shap.Explainer失败: {str(e)}")
                        # 如果shap.Explainer失败，使用KernelExplainer
                        print("  尝试使用shap.KernelExplainer...")
                        explainer = shap.KernelExplainer(black_box_fn, background)
                        shap_values = explainer(X_sample_np)
                        print("  shap.KernelExplainer成功")
                    
                    # 获取SHAP值
                    if hasattr(shap_values, 'values'):
                        mean_shap_values = shap_values.values
                        print(f"  SHAP值形状: {mean_shap_values.shape}")
                    else:
                        mean_shap_values = shap_values
                        print(f"  SHAP值形状: {mean_shap_values.shape}")
                    
                    # 确保SHAP值是2D数组
                    if mean_shap_values.ndim == 1:
                        mean_shap_values = mean_shap_values.reshape(-1, 1)
                        print(f"  调整后SHAP值形状: {mean_shap_values.shape}")
                    
                    print(f"  黑盒方法SHAP值形状: {mean_shap_values.shape}")
                except Exception as e:
                    print(f"  KernelExplainer处理失败: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            
            else:
                print(f"不支持的SHAP方法: {shap_method}")
                return
            
            # 确保特征数量与输入匹配（最终检查）
            expected_features = len(self.feature_names)
            if mean_shap_values.shape[1] != expected_features:
                print(f"  警告：平均SHAP值特征数量({mean_shap_values.shape[1]})与预期({expected_features})不匹配")
                if mean_shap_values.shape[1] < expected_features:
                    padding = np.zeros((mean_shap_values.shape[0], expected_features - mean_shap_values.shape[1]))
                    mean_shap_values = np.hstack([mean_shap_values, padding])
                else:
                    mean_shap_values = mean_shap_values[:, :expected_features]
                print(f"  调整后SHAP值形状: {mean_shap_values.shape}")
            
            # 可视化SHAP summary plot（与示例样式完全一致）
            plt.figure(figsize=(14, 12))
            
            # 使用与示例完全一致的样式：带颜色映射的点图
            shap.summary_plot(
                mean_shap_values, 
                X_sample, 
                feature_names=self.feature_names, 
                show=False, 
                plot_type="dot",  # 使用点图，与示例一致
                color_bar_label="Feature value",  # 颜色条标签与示例一致
                cmap="RdBu_r",  # 红蓝颜色映射，与示例一致
                alpha=0.6,  # 点的透明度，与示例一致
                max_display=30,  # 显示更多特征
                plot_size=(14, 12),  # 更大的图表尺寸
                color_bar=True  # 确保显示颜色条
            )
            
            # 添加与示例完全一致的标题
            plt.title("Complications in Cirrhosis Prediction using FH-DA Model", 
                      fontsize=20, 
                      fontweight='bold', 
                      pad=25)
            
            # 调整坐标轴标签
            plt.xlabel("SHAP value (impact on model output)", 
                      fontsize=14, 
                      fontweight='bold')
            
            # 确保图表布局与示例一致
            plt.subplots_adjust(left=0.20, right=0.98, top=0.90, bottom=0.15)
            
            # 根据使用的方法添加后缀
            method_suffix = f"_{shap_method}" if shap_method != 'classifier_wise' else ""
            
            # 保存图片
            plt.savefig(f"pictures/{method_name}_shap_summary{method_suffix}.png", 
                        dpi=600,  # 更高的分辨率
                        bbox_inches='tight', 
                        facecolor='white', 
                        edgecolor='white')
            plt.close()
            
            print(f"\nSHAP summary plot已保存为: pictures/{method_name}_shap_summary{method_suffix}.png")
            
            # 计算平均SHAP值（绝对值的平均值作为重要性）
            mean_abs_shap = np.abs(mean_shap_values).mean(axis=0)
            shap_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'SHAP_Importance': mean_abs_shap
            }).sort_values('SHAP_Importance', ascending=False)
            
            # 保存SHAP重要性结果
            if method_name not in self.feature_importance_results:
                self.feature_importance_results[method_name] = {}
            
            # 根据使用的方法保存结果
            result_key = f'shap_importance_{shap_method}'
            self.feature_importance_results[method_name][result_key] = shap_importance_df
            
            # 打印前10个重要特征
            print("\n前10个SHAP重要特征:")
            for i, (_, row) in enumerate(shap_importance_df.head(10).iterrows()):
                print(f"{i+1}. {row['Feature']}: {row['SHAP_Importance']:.4f}")
                
        except Exception as e:
            print(f"计算SHAP重要性时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def run_analysis(self, methods_to_evaluate, fold=1, shap_methods=None):
        """运行完整的可解释性分析"""
        # 生成分类器池
        print("生成分类器池...")
        self.generate_classifier_pools(topk=7, random_state=42, fold=fold)
        
        # 训练并评估每个方法
        for method_name, (method_cls, method_params) in methods_to_evaluate.items():
            self.train_and_evaluate_method(method_name, method_cls, method_params)
        
        # 设置默认SHAP方法
        if shap_methods is None:
            # ['classifier_wise', 'black_box'] # 默认使用两种方法
            shap_methods = ['classifier_wise']  
        
        # 计算每个方法的特征重要性
        for method_name in methods_to_evaluate.keys():
            # 计算permutation importance
            self.calculate_permutation_importance(method_name)
            
            # 计算SHAP importance - 使用所有指定的方法
            for shap_method in shap_methods:
                self.calculate_shap_importance(method_name, shap_method=shap_method)
        
        # 生成最终报告
        self.generate_final_report()
    
    def generate_final_report(self):
        """生成最终的特征重要性报告"""
        print(f"\n{'='*80}")
        print(f"FH-DA模型可解释性分析报告")
        print(f"{'='*80}")
        
        # 汇总所有方法的特征重要性
        all_importances = []
        
        for method_name, importance_data in self.feature_importance_results.items():
            if 'permutation_importance' in importance_data:
                result = importance_data['permutation_importance']
                sorted_idx = result.importances_mean.argsort()[::-1]
                for i, idx in enumerate(sorted_idx[:10]):  # 取前10个特征
                    all_importances.append({
                        'Method': method_name,
                        'Feature': self.feature_names[idx],
                        'Importance_Type': 'Permutation',
                        'Importance_Value': result.importances_mean[idx],
                        'Importance_Rank': i+1
                    })
            
            # 处理两种SHAP方法的结果
            for key in importance_data.keys():
                if key.startswith('shap_importance_'):
                    # 提取SHAP方法类型
                    shap_method_type = key.split('_', 2)[2]
                    shap_df = importance_data[key]
                    for i, (_, row) in enumerate(shap_df.head(10).iterrows()):
                        all_importances.append({
                            'Method': method_name,
                            'Feature': row['Feature'],
                            'Importance_Type': f'SHAP_{shap_method_type}',
                            'Importance_Value': row['SHAP_Importance'],
                            'Importance_Rank': i+1
                        })
        
        # 创建汇总DataFrame
        importance_df = pd.DataFrame(all_importances)
        
        # 保存结果到CSV
        importance_df.to_csv('pictures/fhda_feature_importance_summary.csv', index=False)
        print(f"\n特征重要性汇总已保存到: pictures/fhda_feature_importance_summary.csv")
        
        # 分析最具预测力的特征集
        print(f"\n{'='*60}")
        print(f"最具预测力的特征集分析")
        print(f"{'='*60}")
        
        # 统计每个特征在不同方法和不同重要性类型中出现的次数
        feature_counts = importance_df['Feature'].value_counts()
        
        print(f"\n特征出现频率排名 (前15名):")
        for feature, count in feature_counts.head(15).items():
            print(f"{feature}: {count}次")
        
        # 基于permutation importance的平均重要性
        permutation_importance_mean = importance_df[importance_df['Importance_Type'] == 'Permutation']
        permutation_importance_mean = permutation_importance_mean.groupby('Feature')['Importance_Value'].mean().sort_values(ascending=False)
        
        print(f"\n基于Permutation Importance的平均重要性 (前15名):")
        for feature, importance in permutation_importance_mean.head(15).items():
            print(f"{feature}: {importance:.4f}")
        
        # 基于SHAP的平均重要性 - 分别处理两种方法
        shap_types = [col for col in importance_df['Importance_Type'].unique() if col.startswith('SHAP_')]
        
        for shap_type in shap_types:
            shap_importance_mean = importance_df[importance_df['Importance_Type'] == shap_type]
            if not shap_importance_mean.empty:
                shap_importance_mean = shap_importance_mean.groupby('Feature')['Importance_Value'].mean().sort_values(ascending=False)
                
                print(f"\n基于{shap_type}的平均重要性 (前15名):")
                for feature, importance in shap_importance_mean.head(15).items():
                    print(f"{feature}: {importance:.4f}")
        
        # 所有SHAP方法的综合平均
        if len(shap_types) > 0:
            all_shap_importance = importance_df[importance_df['Importance_Type'].isin(shap_types)]
            combined_shap_mean = all_shap_importance.groupby('Feature')['Importance_Value'].mean().sort_values(ascending=False)
            
            print(f"\n基于所有SHAP方法的综合平均重要性 (前15名):")
            for feature, importance in combined_shap_mean.head(15).items():
                print(f"{feature}: {importance:.4f}")
        
        print(f"\n{'='*80}")
        print(f"Analysis completed!")
        print(f"{'='*80}")
        print(f"Generated files:")
        print(f"- Permutation Importance plots for each method: pictures/*permutation_importance.png")
        print(f"- SHAP Summary plots for each method: pictures/*shap_summary.png")
        print(f"- Feature importance summary CSV: pictures/fhda_feature_importance_summary.csv")


def main():
    """主函数"""
    warnings.filterwarnings('ignore')
    
    # 1. 配置参数
    fold = 1
    num_folds = 5
    
    # 2. 加载数据集
    print("加载数据集...")
    
    # 使用ACLF数据集
    synthetic_datasets = load_datasets_from_folder(
        folder_path='filtered_data_english/PD',
        file_extension='.csv', 
        prefix=f'PD_final_train_fold_{fold}'
    )
    
    # 得到测试集
    # 尝试不同编码来解决解码错误
    try:
        test_set = pd.read_csv(f"filtered_data_english/PD/PD_final_test_fold_{fold}.csv", encoding='utf-8')  # 使用utf-8编码读取中文
    except UnicodeDecodeError:
        try:
            test_set = pd.read_csv(f"filtered_data_english/PD/PD_final_test_fold_{fold}.csv", encoding='gbk')
        except UnicodeDecodeError:
            try:
                test_set = pd.read_csv(f"filtered_data_english/PD/PD_final_test_fold_{fold}.csv", encoding='latin1')
            except Exception as e:
                print(f"无法读取测试集文件: {e}")
                raise
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]
    
    # 得到动态选择集
    # 尝试不同编码来解决解码错误
    try:
        desl_set = pd.read_csv(f'filtered_data_english/PD/PD_final_desl_fold_{fold}.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            desl_set = pd.read_csv(f'filtered_data_english/PD/PD_final_desl_fold_{fold}.csv', encoding='gbk')
        except UnicodeDecodeError:
            try:
                desl_set = pd.read_csv(f'filtered_data_english/PD/PD_final_desl_fold_{fold}.csv', encoding='latin1')
            except Exception as e:
                print(f"无法读取设计集文件: {e}")
                raise
    X_desl = desl_set.iloc[:, :-1]
    y_desl = desl_set.iloc[:, -1]
    
    # 确保训练数据集也保留列名
    for i in range(len(synthetic_datasets)):
        X_train, y_train = synthetic_datasets[i]
        if isinstance(X_train, np.ndarray):
            synthetic_datasets[i] = (pd.DataFrame(X_train, columns=X_test.columns), y_train)
    
    # 调试：打印特征名
    print("\nLoaded feature names from test set:")
    print(f"Number of features: {len(X_test.columns)}")
    print(f"First 10 feature names: {list(X_test.columns[:10])}")
    print(f"Last 10 feature names: {list(X_test.columns[-10:])}")
    
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
    #     'SVM': partial(SVC, kernel='rbf', C=0.0001, gamma=0.0001, class_weight=None, tol=0.0001, max_iter=1000, random_state=42),
    #     }    
    
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
        'SVM': partial(SVC, kernel='rbf', C=0.0001, gamma=0.0001, class_weight=None, tol=0.0001, max_iter=1000, random_state=42),
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
    #     }
    

    # 4. 定义要评估的动态选择方法
    methods_to_evaluate = {
        # 指定的动态选择方法
        # # (1)GY
        # 'FH_DA_bi-slope_rectangular': (
        #     FHDES_AllBoxes_vector_rectangle_clustering_optimized,
        #     {
        #         'n_clusters': 5,
        #         'bandwidth_selection': 'silverman',
        #         'density_estimation': 'kde',
        #         'theta': 0.001,
        #         'mu': 0.4,
        #         'mis_sample_based': True,
        #         'doContraction': True,
        #         'thetaCheck': False,
        #         'multiCore_process': True,
        #         'shuffle_dataOrder': False,
        #         'knn_k': 3,
        #         'use_pca': False,
        #         'pca_components': 5,
        #         'normalize_weights': True,
        #         'normalize_correlations': False,
        #         'bandwidth': 0.27887919279027584,
        #         'density_weight': 0.85,
        #         'cluster_distance_weight': 0.590801864085726,
        #         'basic_weight': 0.9,
        #         'enhancement_weight': 0.3644400902697933,
        #         'epsilon': 3.70645854340426e-08,
        #         'alpha': 1.181425093183144,
        #         'beta': 3.270077472634777
        #     }
        # ),
        # # (2)HE
        # 'FH_DA': (
        #     FHDES_JFB_vector_clustering_optimized,
        #     {
        #         'theta': 0.001,
        #         'mu': 0.4,
        #         'mis_sample_based': True,
        #         'doContraction': True,
        #         'thetaCheck': False,
        #         'n_clusters': 2,
        #         'bandwidth_selection': 'silverman',
        #         'use_pca': True,
        #         'pca_components': 5,
        #         'density_estimation': 'knn_density',
        #         'knn_k': 2,
        #         'density_weight': 0.5,
        #         'cluster_distance_weight': 0.5,
        #         'normalize_weights': True,
        #         'epsilon': 1e-8,
        #         'normalize_correlations': True,
        #         'multiCore_process': True,
        #         'shuffle_dataOrder': False
        #     }
        # ),
        # (3)PD
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
        # # (4)ACLF
        # 'FH_DA_bi-slope_rectangular': (
        #     FHDES_AllBoxes_vector_rectangle_clustering_optimized,
        #     {'mis_sample_based': True, 'doContraction': True, 'thetaCheck': False, 'multiCore_process': False, 
        #     'shuffle_dataOrder': True, 'n_clusters': 5, 'bandwidth_selection': 'adaptive', 'density_estimation': 'kde', 
        #     'knn_k': 4, 'use_pca': True, 'pca_components': 10, 'normalize_weights': True, 'normalize_correlations': True, 
        #     'bandwidth': 0.5614924866063653, 'theta': 0.5705753278606428, 'mu': 0.55, 'density_weight': 0.65, 
        #     'cluster_distance_weight': 0.7982400774120468, 'basic_weight': 0.35, 'enhancement_weight': 0.17896348521992975, 
        #     'epsilon': 4.573243428322887e-08, 'alpha': 1.0269641738184827, 'beta': 0.5108427899165016}
        # ),    
        
        }
    
    # 5. 创建可解释性分析器
    # 从测试集获取真实的列名
    feature_names = X_test.columns.tolist()
    analyzer = FHDAExplainabilityAnalyzer(
        models=models,
        train_datasets=synthetic_datasets,
        X_desl=X_desl,
        y_desl=y_desl,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names
    )
    
    # 6. 运行分析
    analyzer.run_analysis(methods_to_evaluate, fold=fold)


if __name__ == "__main__":
    main()
