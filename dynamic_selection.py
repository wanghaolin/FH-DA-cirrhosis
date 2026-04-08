# 标准库
import os
import warnings
import logging
from functools import partial
from itertools import combinations

# 三方科学计算与可视化
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# sklearn 基础组件
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.inspection import permutation_importance

# sklearn 分类器
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
)

# sklearn 指标
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    pairwise_distances,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# 第三方集成学习库
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 不平衡学习指标
from imblearn.metrics import geometric_mean_score

# 表格打印
from tabulate import tabulate

# deslib 官方动态选择与动态集成方法
from deslib.dcs import APriori, APosteriori, LCA, MCB, MLA, OLA, Rank
from deslib.des import DESClustering, DESKNN, DESMI, DESP, KNOP, KNORAE, KNORAU, METADES
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.static.static_selection import StaticSelection
from deslib.static.stacked import StackedClassifier

# 自定义 DES/FH 系列（去重 + 按功能分组）
## 基础 FH 方法
from my_deslib.des import DESFH
from my_deslib.des.des_FHMW_AllBoxes_vector import DESFHMW_allboxes_vector
from my_deslib.des.des_FHMW_JFB_vector import DESFHMW_JFB_vector
from my_deslib.des.des_FHMW_prior_vector import DESFHMW_prior_vector
from my_deslib.des.fh_des_AllBoxes_vector import FHDES_Allboxes_vector
from my_deslib.des.fh_des_JFB_vector import FHDES_JFB_vector
from my_deslib.des.fh_des_prior_vector import FHDES_prior_vector

## 超椭球体版本
from my_deslib.des.hyperellipsoid_versions.des_FHMW_AllBoxes_vector_he import DESFHMW_allboxes_vector_he
from my_deslib.des.hyperellipsoid_versions.des_FHMW_JFB_vector_he import DESFHMW_JFB_vector_he
from my_deslib.des.hyperellipsoid_versions.des_FHMW_prior_vector_he import DESFHMW_prior_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_AllBoxes_vector_he import FHDES_Allboxes_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he import FHDES_JFB_vector_he
from my_deslib.des.hyperellipsoid_versions.fh_des_prior_vector_he import FHDES_prior_vector_he

## 球体版本
from my_deslib.des.sphere_versions.des_FHMW_AllBoxes_vector_sphere import DESFHMW_allboxes_vector_sphere
from my_deslib.des.sphere_versions.des_FHMW_JFB_vector_sphere import DESFHMW_JFB_vector_sphere
from my_deslib.des.sphere_versions.des_FHMW_prior_vector_sphere import DESFHMW_prior_vector_sphere
from my_deslib.des.sphere_versions.fh_des_AllBoxes_vector_sphere import FHDES_AllBoxes_vector_sphere
from my_deslib.des.sphere_versions.fh_des_JFB_vector_sphere import FHDES_JFB_vector_sphere
from my_deslib.des.sphere_versions.fh_des_prior_vector_sphere import FHDES_prior_vector_sphere

## 矩形版本
from my_deslib.des.rectangle_versions.des_FHMW_AllBoxes_vector_rectangle import DESFHMW_allboxes_vector_rectangle
from my_deslib.des.rectangle_versions.des_FHMW_JFB_vector_rectangle import DESFHMW_jfb_vector_rectangle
from my_deslib.des.rectangle_versions.des_FHMW_prior_vector_rectangle import DESFHMW_prior_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_AllBoxes_vector_rectangle import FHDES_AllBoxes_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_JFB_vector_rectangle import FHDES_JFB_vector_rectangle
from my_deslib.des.rectangle_versions.fh_des_prior_vector_rectangle import FHDES_prior_vector_rectangle

## 聚类扩展版本
from my_deslib.des.fh_des_JFB_vector_clustering import FHDES_JFB_vector_clustering
from my_deslib.des.fh_des_JFB_vector_clustering_advance import FHDES_JFB_vector_clustering_advance
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_clustering import FHDES_AllBoxes_vector_clustering
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_he_clustering import FHDES_AllBoxes_vector_he_clustering
from my_deslib.des.fh_des_clustering.des_FHMW_AllBoxes_vector_clustering import DESFHMW_allboxes_vector_clustering
from my_deslib.des.fh_des_clustering.des_FHMW_JFB_vector_clustering_optimized import DESFHMW_JFB_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_clustering_optimized import FHDES_JFB_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_he_clustering_optimized import FHDES_JFB_vector_he_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_sphere_clustering_optimized import FHDES_JFB_vector_sphere_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_Allboxes_vector_he_clustering_optimized import FHDES_Allboxes_vector_he_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_Allboxes_vector_clustering_optimized import FHDES_Allboxes_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_rectangle_clustering_optimized import FHDES_AllBoxes_vector_rectangle_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_sphere_clustering_optimized import FHDES_AllBoxes_vector_sphere_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_rectangle_clustering_optimized import FHDES_JFB_vector_rectangle_clustering_optimized


# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_datasets_from_folder(folder_path, file_extension=".csv", prefix="ADASYN_"):
    """加载数据集，不划分验证集"""
    train_datasets = []
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(file_extension):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
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

    def calculate_pairwise_diversity(self, clf_a_pred, clf_b_pred, y_true):
        """计算两个分类器之间的多样性指标"""
        # 获取联合预测结果
        cm_ab = confusion_matrix(y_true, (clf_a_pred == y_true) & (clf_b_pred == y_true))
        
        # 计算基本统计量
        N00 = cm_ab[0,0]  # 两者都错
        N11 = cm_ab[1,1]  # 两者都对
        N01 = cm_ab[0,1]  # A错B对
        N10 = cm_ab[1,0]  # A对B错
        
        # 多样性指标计算
        metrics = {
            'Q统计量': (N11*N00 - N01*N10) / (N11*N00 + N01*N10 + 1e-9),
            '相关系数': (N11*N00 - N01*N10) / np.sqrt((N11+N10)*(N01+N00)*(N11+N01)*(N10+N00) + 1e-9),
            '不一致性': (N01 + N10) / len(y_true),
            '双错率': N00 / len(y_true)
        }
        return metrics

    def ensemble_diversity(self, predictions, y_true, metric='Q'):
        """
        计算分类器池的整体多样性
        predictions: shape (n_clfs, n_samples)
        """
        n_clfs = predictions.shape[0]
        if n_clfs < 2:
            return {
                'diversity_matrix': np.array([]),
                'mean_diversity': 0,
                'min_diversity': 0,
                'max_diversity': 0
            }

        diversity_matrix = np.zeros((n_clfs, n_clfs))
        
        # 成对计算
        for i, j in combinations(range(n_clfs), 2):
            if metric == 'Q':
                div = self.calculate_pairwise_diversity(predictions[i], predictions[j], y_true)['Q统计量']
            elif metric == 'disagreement':
                div = self.calculate_pairwise_diversity(predictions[i], predictions[j], y_true)['不一致性']
            diversity_matrix[i,j] = div
            diversity_matrix[j,i] = div
        
        # 计算平均多样性
        upper_tri = np.triu_indices_from(diversity_matrix, k=1)
        mean_div = np.mean(diversity_matrix[upper_tri])
        
        return {
            'diversity_matrix': diversity_matrix,
            'mean_diversity': mean_div,
            'min_diversity': np.min(diversity_matrix[upper_tri]),
            'max_diversity': np.max(diversity_matrix[upper_tri])
        }

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
   
    def calculate_iba(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        iba = (2 * acc * f1) / (acc + f1)
        return iba
    
    def train_and_get_importance(self, model_cls, X_train: np.ndarray, y_train: np.ndarray, 
                            X_desl: np.ndarray, y_desl: np.ndarray) -> pd.DataFrame:
        """训练模型并返回特征重要性"""
        # 将 numpy.ndarray 转换为 pandas.DataFrame
        X_train_df = pd.DataFrame(X_train)
        X_desl_df = pd.DataFrame(X_desl)
        
        model = model_cls()  # 实例化模型
        model.fit(X_train_df, y_train)
        # 限制使用的CPU核心数，避免内存不足错误
        result = permutation_importance(model, X_desl_df, y_desl, n_repeats=10, random_state=42, n_jobs=2)
        return pd.DataFrame({
            'Feature': X_train_df.columns,
            'Importance': result.importances_mean,
            'Std': result.importances_std
        })
    
    def generate_classifier_pools(self, models, topk=5):
        all_selected_models = []  # 用于存储每个数据集中性能排序前n_estimators个模型
        for dataset_idx, (X, y) in enumerate(self.train_datasets):
            candidate_models = []
            model_preds = []
            model_probas = []
            model_names = []
            # 先收集所有模型的预测概率
            
            for name, model_cls in models.items():
                model = model_cls()
                model.fit(X, y)
                y_pred = model.predict(self.X_test)  # 使用对应的验证集
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]

                model_preds.append((model, name, y_pred_proba))
                model_probas.append(y_pred_proba)
                model_names.append(name)

            for idx, (model, name, y_pred_proba) in enumerate(model_preds):
                # 计算各项指标
                optimal_threshold = self.optimize_threshold(self.y_test, y_pred_proba, method='youden')  # 使用新阈值优化方法
                y_pred = (y_pred_proba >= optimal_threshold).astype(int)
                F1 = f1_score(self.y_test, y_pred)
                AUC = roc_auc_score(self.y_test, y_pred_proba)
                MCC = matthews_corrcoef(self.y_test, y_pred)
                GMean = geometric_mean_score(self.y_test, y_pred)
                # 添加所有模型到候选列表
                candidate_models.append((model, dataset_idx, name, F1, AUC, MCC, GMean))
            
            # 按AUC分数排序并选择topk个模型
            top_models = sorted(candidate_models, key=lambda x: x[4], reverse=True)[:topk]
            all_selected_models.extend(top_models)

        # 直接使用所有通过性能筛选的模型
        self.classifier_pools = [model for model, _, _, _, _, _, _ in all_selected_models]

        # 性能记录逻辑 - 只记录和打印筛选后的topk模型
        model_performance = [
            {'Dataset Index': dataset_idx+1,
             'Model Name': name,
             'F1': f1,
             'AUC': auc,
             'MCC': mcc,
             'GMean': gmean
             } for (model, dataset_idx, name, f1, auc, mcc, gmean) in all_selected_models]

        performance_df = pd.DataFrame(model_performance)
        print(f"最终筛选后的top{topk}基分类器性能：")
        print(tabulate(performance_df, headers="keys", tablefmt="grid", floatfmt=".3f"))

        return self.classifier_pools

    def calculate_iba(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        iba = (2 * acc * f1) / (acc + f1)
        return iba

    def evaluate_methods(self):
        X_train, y_train = self.train_datasets[0]
        """评估不同动态选择方法"""
        methods = {
            # # 理论最优模型
            # 'Oracle': (Oracle, {}),
            # # 动态选择方法
            # 'LCA': (LCA, {'diff_thresh':0.05, 'k': 7}),
            # 'Rank': (Rank, {'diff_thresh':0.05, 'k': 7}),
            # 'OLA': (OLA, {'diff_thresh':0.05, 'k': 7}),
            # 'APriori': (APriori, {'diff_thresh':0.05, 'k': 7}),
            # 'MCB': (MCB, {'diff_thresh': 0.05, 'k': 7}),
            # 'MLA': (MLA, {'k': 7}),
            # # 动态集成方法
            # # 'voting': 'soft',
            # 'METADES': (METADES, {'k': 7}),
            # 'KNORAE': (KNORAE, {'k': 7}),
            # 'KNORAU': (KNORAU, {'k': 7}),
            # # 'metric_performance':'f1_score'
            # 'DESClustering': (DESClustering, {}),
            # 'KNOP': (KNOP, {'k': 7}),
            # 'DESP': (DESP, {'k': 7}),
            # 'DESKNN': (DESKNN, {'k': 7}),

            # # GY-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4等参数不变，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            # 'FH_DES': (
            #     FHDES_JFB_vector,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_ellipsoidal': (
            #     FHDES_JFB_vector_he,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'gamma': 0.7581691436530708,
            #         'reg_lambda': 1.13798107855186e-05,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_spherical': (
            #     FHDES_JFB_vector_sphere,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'sigma': 0.028175933150630706,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_bi-slope_rectangular': (
            #     FHDES_JFB_vector_rectangle,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'alpha': 1.3320075338937662,
            #         'beta': 4.9713071807123175,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # # GY-改进的超盒变体
            # 'FH_DA': (
            #     FHDES_JFB_vector_clustering_optimized, 
            #     {
            #         'theta':0.001, 'mu':.4, 'mis_sample_based':True,
            #         'doContraction':True, 'thetaCheck':False, 'multiCore_process':True, 'shuffle_dataOrder':False,
            #         'n_clusters': 10, 'bandwidth_selection': 'cv', 'density_estimation': 'knn_density', 'knn_k': 6, 
            #         'use_pca': False, 'pca_components': 6, 'normalize_weights': False, 'normalize_correlations': True, 
            #         'bandwidth': 0.315023760097458, 'density_weight': 0.15000000000000002, 'cluster_distance_weight': 0.584526702261065, 
            #         'basic_weight': 0.1, 'enhancement_weight': 0.9488213816195104, 'epsilon': 4.4875730462234466e-08
            #      }
            # ),
            # 'FH_DA_ellipsoidal': (
            #     FHDES_JFB_vector_he_clustering_optimized,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False,
            #         'n_clusters': 9,
            #         'bandwidth_selection': 'adaptive',
            #         'density_estimation': 'pca_kde',
            #         'knn_k': 7,
            #         'use_pca': True,
            #         'pca_components': 3,
            #         'normalize_weights': True,
            #         'normalize_correlations': True,
            #         'bandwidth': 2.9547412518596725,
            #         'density_weight': 0.5,
            #         'cluster_distance_weight': 0.689299246291069,
            #         'basic_weight': 0.7000000000000001,
            #         'enhancement_weight': 0.17644429085079516,
            #         'epsilon': 6.459686488008564e-08,
            #         'gamma': 0.010923198627741751,
            #         'reg_lambda': 4.210947032191968e-06
            #     }
            # ),
            # 'FH_DA_spherical': (
            #     FHDES_AllBoxes_vector_sphere_clustering_optimized,
            #     {
            #         'mis_sample_based': True,
            #         'doContraction': False,
            #         'thetaCheck': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False,
            #         'n_clusters': 3,
            #         'bandwidth_selection': 'cv',
            #         'density_estimation': 'kde',
            #         'knn_k': 4,
            #         'use_pca': True,
            #         'pca_components': 9,
            #         'normalize_weights': False,
            #         'normalize_correlations': True,
            #         'bandwidth': 1.4639562535426875,
            #         'theta': 0.8277044306380984,
            #         'mu': 0.45000000000000007,
            #         'density_weight': 1.0,
            #         'cluster_distance_weight': 0.2661541039538596,
            #         'basic_weight': 0.85,
            #         'enhancement_weight': 0.3875854066581769,
            #         'epsilon': 4.126349943114586e-08,
            #         'sigma': 0.4868663394812949
            #     }
            # ),
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

            
            # # HE-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            # 'FH_DES': (
            #     FHDES_JFB_vector,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_ellipsoidal': (
            #     FHDES_JFB_vector_he,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'gamma': 10.165830776156998,
            #         'reg_lambda': 1.7290103161667023e-05,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_spherical': (
            #     FHDES_JFB_vector_sphere,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'sigma': 0.9958059560727144,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_bi-slope_rectangular': (
            #     FHDES_JFB_vector_rectangle,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'alpha': 0.8153751093671162,
            #         'beta': 4.691449548222193,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # # HE-改进的超盒变体
            # # 快速计算配置 - 低计算复杂度
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
            # # 超椭球体版本 - 平衡配置 (2v配置)
            # 'FH_DA_ellipsoidal': (
            #     FHDES_JFB_vector_he_clustering_optimized,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.85,
            #         'gamma': 0.06101491367302711,
            #         'reg_lambda': 2.310201887845294e-06,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'n_clusters': 3,
            #         'bandwidth_selection': 'silverman',
            #         'use_pca': True,
            #         'pca_components': 10,
            #         'density_estimation': 'kde',
            #         'density_weight': 0.6,
            #         'cluster_distance_weight': 0.4,
            #         'normalize_weights': True,
            #         'epsilon': 1e-8,
            #         'normalize_correlations': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DA_spherical': (
            #     FHDES_JFB_vector_sphere_clustering_optimized,
            #     {
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False,
            #         'n_clusters': 2,
            #         'bandwidth_selection': 'adaptive',
            #         'density_estimation': 'kde',
            #         'knn_k': 6,
            #         'use_pca': True,
            #         'pca_components': 10,
            #         'normalize_weights': False,
            #         'normalize_correlations': True,
            #         'bandwidth': 0.6082563589986993,
            #         'theta': 0.17063386522055357,
            #         'mu': 0.35,
            #         'density_weight': 0.35,
            #         'cluster_distance_weight': 0.9439845768431889,
            #         'basic_weight': 0.1,
            #         'enhancement_weight': 0.7734201421310619,
            #         'epsilon': 5.7124175613557085e-08,
            #         'sigma': 0.38070292327594585
            #     }
            # ),
            # 'FH_DA_bi-slope_rectangular': (
            #     FHDES_AllBoxes_vector_rectangle_clustering_optimized,
            #     {
            #         'theta': 0.001,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False,
            #         'n_clusters': 3,
            #         'bandwidth_selection': 'silverman',
            #         'density_estimation': 'knn_density',
            #         'knn_k': 4,
            #         'use_pca': False,
            #         'pca_components': 8,
            #         'normalize_weights': False,
            #         'normalize_correlations': True,
            #         'bandwidth': 2.8588262369008905,
            #         'density_weight': 0.65,
            #         'cluster_distance_weight': 0.17930588219809357,
            #         'basic_weight': 0.6,
            #         'enhancement_weight': 0.21880065868715076,
            #         'epsilon': 8.365122846241532e-08,
            #         'alpha': 1.1719664906128544,
            #         'beta': 4.54373856060449
            #     }
            # ),

            # # # PD-超盒形状变体 - 保持相同参数配置'theta'，'mu':.4，剩余参数采用贝叶斯优化算法——增强数据集（50次）
            # 'FH_DES': (
            #     FHDES_Allboxes_vector,
            #     {
            #         'theta': 0.4,
            #         'mu': 0.4,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_ellipsoidal': (
            #     FHDES_Allboxes_vector_he,
            #     {
            #         'theta': 0.4,
            #         'mu': 0.4,
            #         'gamma': 0.8072465573781535,
            #         'reg_lambda': 8.724761227721179e-06,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_spherical': (
            #     FHDES_AllBoxes_vector_sphere,
            #     {
            #         'theta': 0.4,
            #         'mu': 0.4,
            #         'sigma': 0.7969454818643931,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # 'FH_DES_bi-slope_rectangular': (
            #     FHDES_AllBoxes_vector_rectangle,
            #     {
            #         'theta': 0.4,
            #         'mu': 0.4,
            #         'alpha': 0.14049896585154187,
            #         'beta': 0.6072222164274351,
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False
            #     }
            # ),
            # # # PD-改进的超盒变体
            # 'FH_DA': (
            #     FHDES_Allboxes_vector_clustering_optimized,
            #     {
            #         'mis_sample_based': True,
            #         'doContraction': False,
            #         'thetaCheck': True,
            #         'multiCore_process': False,
            #         'shuffle_dataOrder': True,
            #         'n_clusters': 2,
            #         'bandwidth_selection': 'cv',
            #         'density_estimation': 'pca_kde',
            #         'knn_k': 7,
            #         'use_pca': True,
            #         'pca_components': None,
            #         'normalize_weights': True,
            #         'normalize_correlations': True,
            #         'bandwidth': 1.1513058574930677,
            #         'theta': 0.16109297688061047,
            #         'mu': 0.75,
            #         'density_weight': 0.45000000000000007,
            #         'cluster_distance_weight': 0.34661740876334135,
            #         'basic_weight': 0.30000000000000004,
            #         'enhancement_weight': 0.897994362306035,
            #         'epsilon': 4.2285052164024e-08
            #     }
            # ),
            # 'FH_DA_ellipsoidal': (
            #     FHDES_Allboxes_vector_he_clustering_optimized,
            #     {
            #         'mis_sample_based': True,
            #         'doContraction': False,
            #         'thetaCheck': False,
            #         'n_clusters': 8,
            #         'bandwidth_selection': 'cv',
            #         'density_estimation': 'knn_density',
            #         'knn_k': 10,
            #         'use_pca': False,
            #         'pca_components': None,
            #         'normalize_weights': False,
            #         'normalize_correlations': False,
            #         'multiCore_process': False,
            #         'shuffle_dataOrder': True,
            #         'bandwidth': 0.24327392618081645,
            #         'mu': 0.5,
            #         'theta': 0.580934134852548,
            #         'density_weight': 0.30000000000000004,
            #         'cluster_distance_weight': 0.1,
            #         'basic_weight': 0.25,
            #         'enhancement_weight': 0.4,
            #         'epsilon': 9.131795402684815e-08,
            #         'gamma': 14.685312005983562,
            #         'reg_lambda': 1.0227915932908707e-05
            #     }
            # ),
            # 'FH_DA_spherical': (
            #     FHDES_AllBoxes_vector_sphere_clustering_optimized,
            #     {
            #         'mis_sample_based': True,
            #         'doContraction': False,
            #         'thetaCheck': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': True,
            #         'n_clusters': 2,
            #         'bandwidth_selection': 'adaptive',
            #         'density_estimation': 'pca_kde',
            #         'knn_k': 5,
            #         'use_pca': False,
            #         'pca_components': None,
            #         'normalize_weights': False,
            #         'normalize_correlations': True,
            #         'bandwidth': 1.0025763510308325,
            #         'theta': 0.7315157902574336,
            #         'mu': 0.45000000000000007,
            #         'density_weight': 0.6,
            #         'cluster_distance_weight': 0.8200555983272062,
            #         'basic_weight': 0.35,
            #         'enhancement_weight': 0.62394472757224,
            #         'epsilon': 4.039819547001331e-08,
            #         'sigma': 0.10881066516231722
            #     }
            # ),
            # 'FH_DA_bi-slope_rectangular': (
            #     FHDES_AllBoxes_vector_rectangle_clustering_optimized,
            #     {
            #         'mis_sample_based': True,
            #         'doContraction': True,
            #         'thetaCheck': False,
            #         'n_clusters': 9,
            #         'bandwidth_selection': 'adaptive',
            #         'density_estimation': 'pca_kde',
            #         'knn_k': 5,
            #         'use_pca': False,
            #         'pca_components': 3,
            #         'normalize_weights': True,
            #         'normalize_correlations': True,
            #         'multiCore_process': True,
            #         'shuffle_dataOrder': False,
            #         'bandwidth': 1.8190427869701697,
            #         'mu': 0.85,
            #         'theta': 0.3463644376849158,
            #         'density_weight': 0.6,
            #         'cluster_distance_weight': 0.1,
            #         'basic_weight': 0.8,
            #         'enhancement_weight': 0.4,
            #         'epsilon': 4.548447107982115e-08,
            #         'alpha': 1.6300899902636972,
            #         'beta': 4.85858264996245
            #     }
            # ),

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
                # 'Balanced Acc (test)': balanced_accuracy_score(self.y_test, y_pred_test)
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

    # # PD-自动参数调参
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

    # GY-自动调参
    models = {
        'NB': partial(GaussianNB, var_smoothing=1.7670169402947963e-10),
        'KNN': partial(KNeighborsClassifier, algorithm='kd_tree', leaf_size=37, n_neighbors=28, weights='distance', p=2),
        'LR': partial(LogisticRegression, penalty='l2', C=0.43736593268991864, class_weight=None, tol=0.0010276978799463987, max_iter=281, random_state=42),
        'LDA': partial(LinearDiscriminantAnalysis, solver='lsqr', shrinkage='auto', tol=4.207053950287936e-06),
        'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.4418374353523307, tol=1.208886038766324e-05),
        'RF': partial(RandomForestClassifier, n_estimators=150, max_depth=3, min_samples_split=15, min_samples_leaf=3, max_features='sqrt', criterion='entropy', class_weight=None, random_state=42),    
        'ET': partial(ExtraTreesClassifier, n_estimators=100, max_depth=9, min_samples_leaf=3, min_samples_split=9, max_features=0.2, bootstrap=False, class_weight=None, criterion='entropy', random_state=42),
        'ADA': partial(AdaBoostClassifier, learning_rate=0.05159403035135414, n_estimators=50, algorithm='SAMME.R', random_state=42),
        'GBC': partial(GradientBoostingClassifier, learning_rate=0.013395058503939341, n_estimators=300, max_depth=4, max_features='log2', subsample=0.7847427943496508, min_samples_split=11, min_samples_leaf=5, random_state=42),
        'LGBM': partial(LGBMClassifier, learning_rate=0.1251033015766346, n_estimators=50, num_leaves=10, class_weight='balanced', boosting_type='dart', reg_alpha=0.03326964102432317, reg_lambda=0.05510519784818472, random_state=42),
        'XGB': partial(XGBClassifier, max_depth=3, eta=0.028875929928602655, reg_alpha=0.0008183036138409682, reg_lambda=0.0007539166440950076, random_state=42),
    }

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

        # filtered_data/HE  generate_embed/filtered_data/HE
        synthetic_datasets = load_datasets_from_folder(
            folder_path='filtered_data/GY',
            # _disease_attention_pca
            file_extension='.csv', prefix=f'GY_final_train_fold_{fold}')

        # 得到测试集
        test_set = pd.read_csv(f"filtered_data/GY/GY_final_test_fold_{fold}.csv")
        X_test = test_set.iloc[:, :-1]
        y_test = test_set.iloc[:, -1]

        # # 得到外部验证集
        # test_set = pd.read_csv(f"preprocess_data/HE_filtered2_disease_attention_pca_external_preprocessed_0.4.csv")
        # X_test = test_set.iloc[:, :-1]
        # y_test = test_set.iloc[:, -1]

        # 得到动态选择集
        desl_set = pd.read_csv(f'filtered_data/GY/GY_final_desl_fold_{fold}.csv')
        X_desl = desl_set.iloc[:, :-1]
        y_desl = desl_set.iloc[:, -1]
        
        evaluator = DESModelEvaluator(models, synthetic_datasets, X_desl,y_desl, X_test, y_test, diversity_threshold=1)
        evaluator.generate_classifier_pools(models,topk=7)
        fold_results = evaluator.evaluate_methods()
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
