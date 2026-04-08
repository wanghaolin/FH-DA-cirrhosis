import os
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.base import clone
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer, TVAESynthesizer, CopulaGANSynthesizer
from imblearn.metrics import geometric_mean_score
from tabulate import tabulate

warnings.filterwarnings('ignore')

# 导入聚类优化的超盒算法
from my_deslib.des.fh_des_clustering.fh_des_Allboxes_vector_clustering_optimized import FHDES_Allboxes_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_Allboxes_vector_he_clustering_optimized import FHDES_Allboxes_vector_he_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_sphere_clustering_optimized import FHDES_AllBoxes_vector_sphere_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_AllBoxes_vector_rectangle_clustering_optimized import FHDES_AllBoxes_vector_rectangle_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_clustering_optimized import FHDES_JFB_vector_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_he_clustering_optimized import FHDES_JFB_vector_he_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_sphere_clustering_optimized import FHDES_JFB_vector_sphere_clustering_optimized
from my_deslib.des.fh_des_clustering.fh_des_JFB_vector_rectangle_clustering_optimized import FHDES_JFB_vector_rectangle_clustering_optimized

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from functools import partial

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import json
import joblib
from datetime import datetime
import warnings
import multiprocessing

# 过滤scikit-learn的警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


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

# # HE-自动调参
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

# # 定义模糊超盒模型类和对应的参数搜索空间
# # 优化参数搜索空间，基于先验知识和经验缩小搜索范围
# base_clustering_fh_params = {
#     # 基础参数
#     'theta': (0.1, 1.0),
#     'mu': (0.1, 1.0),
#     'mis_sample_based': [True, False],
#     'doContraction': [True, False],
#     'thetaCheck': [True, False],
#     'multiCore_process': [True, False],
#     'shuffle_dataOrder': [True, False],
#     # 聚类参数
#     'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
#     'bandwidth': (0.1, 3.0),

#     # 密度估计参数
#     'density_estimation': ['kde', 'knn_density', 'pca_kde'],
#     'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
#     'use_pca': [True, False],
#     'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],

#     # 权重参数 - 所有权重参数独立优化，不再使用线性约束
#     'density_weight': (0.1, 1.0),
#     'cluster_distance_weight': (0.1, 1.0),
#     'basic_weight': (0.1, 1.0),
#     'enhancement_weight': (0.1, 1.0),
#     # 其他参数
#     'normalize_weights': [True, False],
#     'normalize_correlations': [True, False],
#     # 基于模糊椭球体的参数搜索空间
#     'gamma': (0.01, 50),
#     'reg_lambda': (1e-6, 1e-4),
#     # 基于模糊球体的参数搜索空间
#     'sigma': (0.01, 1.0),
#     # 基于模糊矩形的参数搜索空间
#     'alpha': (0.1, 2.0),
#     'beta': (0.5, 5.0),
# }

# 固定参数配置，保持四类算法的可对比性
fixed_params = {
    # # 7v
    # 'theta': 0.4,
    # 'mu': 0.4,
    # 'mis_sample_based': True,
    # 'doContraction': True,
    # 'thetaCheck': True,
    # 'multiCore_process': True,
    # 'shuffle_dataOrder': False
    
    # # 2v
    # 'theta':0.001, 
    # 'mu':.4, 
    # 'mis_sample_based':True,
    # 'doContraction':True, 
    # 'thetaCheck':False, 
    # 'multiCore_process':True, 
    # 'shuffle_dataOrder':False

    # # 4V-ACLF
    # 'theta':0.001, 
    # 'mu':.1, 
    # 'mis_sample_based':True,
    # 'doContraction':True, 
    # 'thetaCheck':False, 
    # 'multiCore_process':False, 
    # 'shuffle_dataOrder':False
}

fh_models = {
    # PD|ACLF-自动调参
    # 基于模糊超盒的模型 - 聚类优化版
    'FHDES_Allboxes_vector_clustering_optimized': {
        'class': FHDES_Allboxes_vector_clustering_optimized,
        'params': {
            # 基础参数
            'theta': (0.1, 1.0),
            'mu': (0.1, 1.0),
            'mis_sample_based': [True, False],
            'doContraction': [True, False],
            'thetaCheck': [True, False],
            'multiCore_process': [True, False],
            'shuffle_dataOrder': [True, False],

            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
            'bandwidth': (0.1, 3.0),
            'density_estimation': ['kde', 'knn_density', 'pca_kde'],
            'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
            'use_pca': [True, False],
            'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
            # 权重参数 - 独立优化，不再使用线性约束
            'density_weight': (0.1, 1.0),
            'cluster_distance_weight': (0.1, 1.0),
            'basic_weight': (0.1, 1.0),
            'enhancement_weight': (0.1, 1.0),
            'normalize_weights': [True, False],
            'normalize_correlations': [True, False],
            'epsilon': (1e-9, 1e-7)
        }
    },
    # # 基于马氏距离的超椭球体模型 - 聚类优化版
    # 'FHDES_Allboxes_vector_he_clustering_optimized': {
    #     'class': FHDES_Allboxes_vector_he_clustering_optimized,
    #     'params': {
    #         # # 基础参数
    #         # 'theta': (0.1, 1.0),
    #         # 'mu': (0.1, 1.0),
    #         # 'mis_sample_based': [True, False],
    #         # 'doContraction': [True, False],
    #         # 'thetaCheck': [True, False],
    #         # 'multiCore_process': [True, False],
    #         # 'shuffle_dataOrder': [True, False],

    #         'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
    #         'bandwidth': (0.1, 3.0),
    #         'density_estimation': ['kde', 'knn_density', 'pca_kde'],
    #         'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'use_pca': [True, False],
    #         'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
    #         # 权重参数 - 独立优化，不再使用线性约束
    #         'density_weight': (0.1, 1.0),
    #         'cluster_distance_weight': (0.1, 1.0),
    #         'basic_weight': (0.1, 1.0),
    #         'enhancement_weight': (0.1, 1.0),
    #         'normalize_weights': [True, False],
    #         'normalize_correlations': [True, False],
    #         'epsilon': (1e-9, 1e-7),
    #         'gamma': (0.01, 50),
    #         'reg_lambda': (1e-6, 1e-4)
    #     }
    # },
    # # 基于超球体模型 - 聚类优化版
    # 'FHDES_Allboxes_vector_sphere_clustering_optimized': {
    #     'class': FHDES_AllBoxes_vector_sphere_clustering_optimized,
    #     'params': {
    #         # 基础参数
    #         'theta': (0.1, 1.0),
    #         'mu': (0.1, 1.0),
    #         'mis_sample_based': [True, False],
    #         'doContraction': [True, False],
    #         'thetaCheck': [True, False],
    #         'multiCore_process': [True, False],
    #         'shuffle_dataOrder': [True, False],

    #         'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
    #         'bandwidth': (0.1, 3.0),
    #         'density_estimation': ['kde', 'knn_density', 'pca_kde'],
    #         'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'use_pca': [True, False],
    #         'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
    #         # 权重参数 - 独立优化，不再使用线性约束
    #         'density_weight': (0.1, 1.0),
    #         'cluster_distance_weight': (0.1, 1.0),
    #         'basic_weight': (0.1, 1.0),
    #         'enhancement_weight': (0.1, 1.0),
    #         'normalize_weights': [True, False],
    #         'normalize_correlations': [True, False],
    #         'epsilon': (1e-9, 1e-7),
    #         'sigma': (0.01, 1.0)
    #     }
    # },
    # 基于超矩形模型 - 聚类优化版
    'FHDES_Allboxes_vector_rectangle_clustering_optimized': {
        'class': FHDES_AllBoxes_vector_rectangle_clustering_optimized,
        'params': {
            # 基础参数
            'theta': (0.1, 1.0),
            'mu': (0.1, 1.0),
            'mis_sample_based': [True, False],
            'doContraction': [True, False],
            'thetaCheck': [True, False],
            'multiCore_process': [True, False],
            'shuffle_dataOrder': [True, False],
            
            'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
            'bandwidth': (0.1, 3.0),
            'density_estimation': ['kde', 'knn_density', 'pca_kde'],
            'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
            'use_pca': [True, False],
            'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
            # 权重参数 - 独立优化，不再使用线性约束
            'density_weight': (0.1, 1.0),
            'cluster_distance_weight': (0.1, 1.0),
            'basic_weight': (0.1, 1.0),
            'enhancement_weight': (0.1, 1.0),
            'normalize_weights': [True, False],
            'normalize_correlations': [True, False],
            'epsilon': (1e-9, 1e-7),
            'alpha': (0.1, 2.0),
            'beta': (0.5, 5.0)
        }
    },

    # HE|GY-自动调参
    # 基于模糊超盒的模型 - 聚类优化版
    # 'FHDES_JFB_vector_clustering_optimized': {
    #     'class': FHDES_JFB_vector_clustering_optimized,
    #     'params': {
    #         # # 基础参数
    #         # 'theta': (0.1, 1.0),
    #         # 'mu': (0.1, 1.0),
    #         # 'mis_sample_based': [True, False],
    #         # 'doContraction': [True, False],
    #         # 'thetaCheck': [True, False],
    #         # 'multiCore_process': [True, False],
    #         # 'shuffle_dataOrder': [True, False],

    #         'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
    #         'bandwidth': (0.1, 3.0),
    #         'density_estimation': ['kde', 'knn_density', 'pca_kde'],
    #         'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'use_pca': [True, False],
    #         'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
    #         # 权重参数 - 独立优化，不再使用线性约束
    #         'density_weight': (0.1, 1.0),
    #         'cluster_distance_weight': (0.1, 1.0),
    #         'basic_weight': (0.1, 1.0),
    #         'enhancement_weight': (0.1, 1.0),
    #         'normalize_weights': [True, False],
    #         'normalize_correlations': [True, False],
    #         'epsilon': (1e-9, 1e-7)
    #     }
    # },
    # # 基于马氏距离的超椭球体模型 - 聚类优化版
    # 'FHDES_JFB_vector_he_clustering_optimized': {
    #     'class': FHDES_JFB_vector_he_clustering_optimized,
    #     'params': {
    #         # 基础参数
    #         'theta': (0.1, 1.0),
    #         'mu': (0.1, 1.0),
    #         'mis_sample_based': [True, False],
    #         'doContraction': [True, False],
    #         'thetaCheck': [True, False],
    #         'multiCore_process': [True, False],
    #         'shuffle_dataOrder': [True, False],

    #         'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
    #         'bandwidth': (0.1, 3.0),
    #         'density_estimation': ['kde', 'knn_density', 'pca_kde'],
    #         'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'use_pca': [True, False],
    #         'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
    #         # 权重参数 - 独立优化，不再使用线性约束
    #         'density_weight': (0.1, 1.0),
    #         'cluster_distance_weight': (0.1, 1.0),
    #         'basic_weight': (0.1, 1.0),
    #         'enhancement_weight': (0.1, 1.0),
    #         'normalize_weights': [True, False],
    #         'normalize_correlations': [True, False],
    #         'epsilon': (1e-9, 1e-7),
    #         'gamma': (0.01, 50),
    #         'reg_lambda': (1e-6, 1e-4)
    #     }
    # },
    # # 基于超球体模型 - 聚类优化版
    # 'FHDES_JFB_vector_sphere_clustering_optimized': {
    #     'class': FHDES_JFB_vector_sphere_clustering_optimized,
    #     'params': {
    #         # 基础参数
    #         'theta': (0.1, 1.0),
    #         'mu': (0.1, 1.0),
    #         'mis_sample_based': [True, False],
    #         'doContraction': [True, False],
    #         'thetaCheck': [True, False],
    #         'multiCore_process': [True, False],
    #         'shuffle_dataOrder': [True, False],

    #         'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
    #         'bandwidth': (0.1, 3.0),
    #         'density_estimation': ['kde', 'knn_density', 'pca_kde'],
    #         'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'use_pca': [True, False],
    #         'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
    #         # 权重参数 - 独立优化，不再使用线性约束
    #         'density_weight': (0.1, 1.0),
    #         'cluster_distance_weight': (0.1, 1.0),
    #         'basic_weight': (0.1, 1.0),
    #         'enhancement_weight': (0.1, 1.0),
    #         'normalize_weights': [True, False],
    #         'normalize_correlations': [True, False],
    #         'epsilon': (1e-9, 1e-7),
    #         'sigma': (0.01, 1.0)
    #     }
    # },
    # # 基于超矩形模型 - 聚类优化版
    # 'FHDES_JFB_vector_rectangle_clustering_optimized': {
    #     'class': FHDES_JFB_vector_rectangle_clustering_optimized,
    #     'params': {
    #         # # 基础参数
    #         # 'theta': (0.1, 1.0),
    #         # 'mu': (0.1, 1.0),
    #         # 'mis_sample_based': [True, False],
    #         # 'doContraction': [True, False],
    #         # 'thetaCheck': [True, False],
    #         # 'multiCore_process': [True, False],
    #         # 'shuffle_dataOrder': [True, False],
            
    #         'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'bandwidth_selection': ['silverman', 'cv', 'adaptive'],
    #         'bandwidth': (0.1, 3.0),
    #         'density_estimation': ['kde', 'knn_density', 'pca_kde'],
    #         'knn_k': [3, 4, 5, 6, 7, 8, 9, 10],
    #         'use_pca': [True, False],
    #         'pca_components': [3, 4, 5, 6, 7, 8, 9, 10, None],
    #         # 权重参数 - 独立优化，不再使用线性约束
    #         'density_weight': (0.1, 1.0),
    #         'cluster_distance_weight': (0.1, 1.0),
    #         'basic_weight': (0.1, 1.0),
    #         'enhancement_weight': (0.1, 1.0),
    #         'normalize_weights': [True, False],
    #         'normalize_correlations': [True, False],
    #         'epsilon': (1e-9, 1e-7),
    #         'alpha': (0.1, 2.0),
    #         'beta': (0.5, 5.0)
    #     }
    # },

}

def load_datasets_from_folder(folder_path, file_extension, prefix):
    """加载指定文件夹下的数据集"""
    datasets = []
    for filename in os.listdir(folder_path):
        if filename.startswith(prefix) and filename.endswith(file_extension):
            df = pd.read_csv(os.path.join(folder_path, filename))
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            datasets.append((X, y))
    return datasets

def get_fold_data(fold):
    """获取指定折的训练集、测试集和动态选择集数据"""
    # 加载数据集 - 修正prefix参数，移除路径部分
    synthetic_datasets = load_datasets_from_folder(
        folder_path='filtered_data/ACLF',
        file_extension='.csv', prefix=f'ACLF_final_train_fold_{fold}')
    
    # 安全检查：如果没有找到训练集，直接从CSV文件加载
    if not synthetic_datasets:
        # 尝试直接读取训练集文件
        train_set = pd.read_csv(f"filtered_data/ACLF/ACLF_final_train_fold_{fold}.csv")
        X_train = train_set.iloc[:, :-1]
        y_train = train_set.iloc[:, -1]
    else:
        # 这里简单取第一个训练集
        X_train, y_train = synthetic_datasets[0]
    
    # 得到测试集
    test_set = pd.read_csv(f"filtered_data/ACLF/ACLF_final_test_fold_{fold}.csv")
    X_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]
    
    # 得到动态选择集
    desl_set = pd.read_csv(f'filtered_data/ACLF/ACLF_final_desl_fold_{fold}.csv')
    X_desl = desl_set.iloc[:, :-1]
    y_desl = desl_set.iloc[:, -1]
    
    return X_train, y_train, X_desl, y_desl, X_test, y_test

def _conditional_undersampling(X, y, sampling_strategy=0.5, random_state=42):
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
    
    return X_resampled, y_resampled


def _generate_minority_samples(X, y, method='smote', minority_ratio=0.3, random_state=42):
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
            return X_combined, y_combined
    
    except Exception as e:
        print(f"数据增强方法 {method} 出错: {str(e)}")
        return X, y


def generate_classifier_pools(models, X_train, y_train, X_test, y_test, top_k=7, random_state=42, fold=1):
    """
    生成基分类器池，通过条件欠采样和多种数据增强方法缓解类别不平衡问题
    
    参数:
        models: 基分类器字典
        X_train: 训练集特征
        y_train: 训练集标签
        X_test: 测试集特征
        y_test: 测试集标签
        top_k: 选择的top AUC分类器数量
        random_state: 随机种子
        fold: 当前折数
        
    返回:
        优化后的分类器池
    """
    all_models_info = []  # 存储所有模型信息
    
    # 设置全局随机种子
    np.random.seed(random_state)
    
    # 首先处理原始数据集
    print(f"\n处理原始数据集")
    for model_name, model_cls in models.items():
        try:
            model = model_cls(random_state=random_state) if hasattr(model_cls, '__name__') and model_cls.__name__ not in ['SVC', 'KNeighborsClassifier', 'GaussianNB'] else model_cls()
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
            AUC = roc_auc_score(y_test, y_pred_proba)
            
            all_models_info.append({
                'model': model,
                'name': f"{model_name}_original",
                'original_name': model_name,
                'sampling_method': 'original',
                'AUC': AUC
            })
            
            print(f"  模型 {model_name}_original: AUC={AUC:.4f}")
            
        except Exception as e:
            print(f"  训练模型 {model_name}_original 时出错: {str(e)}")
    
    # 定义5种不同的随机种子用于条件欠采样
    undersample_seeds = [42, 43, 44, 45, 46]
    
    # 定义8种数据增强方法
    augmentation_methods = ['smote', 'adasyn', 'borderline_smote', 'svm_smote', 
                           'gaussian_copula', 'ctgan', 'tvae', 'copulagan'
                           ]
    
    # 增强数据保存路径，包含折数信息
    enhance_data_dir = os.path.join("enhance_data", "ACLF", f"fold{fold}")
    os.makedirs(enhance_data_dir, exist_ok=True)
    
    # 对每种条件欠采样数据集进行处理
    for i, undersample_seed in enumerate(undersample_seeds):
        print(f"\n条件欠采样数据集 {i+1}/{len(undersample_seeds)} (随机种子: {undersample_seed})")
        
        # 执行条件欠采样
        undersampled_X, undersampled_y = _conditional_undersampling(
            X_train, y_train, sampling_strategy=0.5, random_state=undersample_seed)
        
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
                augmented_X, augmented_y = _generate_minority_samples(
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
                    model.fit(augmented_X, augmented_y)
                    
                    # 评估模型
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.decision_function(X_test)
                    AUC = roc_auc_score(y_test, y_pred_proba)
                    
                    all_models_info.append({
                        'model': model,
                        'name': unique_name,
                        'original_name': model_name,
                        'sampling_method': f'undersample_{i+1}_{aug_method}',
                        'AUC': AUC
                    })
                    
                    print(f"    模型 {unique_name}: AUC={AUC:.4f}")
                    
                except Exception as e:
                    print(f"    训练模型 {unique_name} 时出错: {str(e)}")
    
    # 按AUC降序排序所有模型
    all_models_info.sort(key=lambda x: (x['AUC'], x['name']), reverse=True)
    
    # 选择top-k个模型
    selected_models_info = all_models_info[:top_k] if len(all_models_info) > top_k else all_models_info
    
    # 提取分类器池
    classifier_pools = [model_info['model'] for model_info in selected_models_info]
    
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
            'AUC': model_info['AUC']
        })
    
    performance_df = pd.DataFrame(performance_data)
    print(tabulate(performance_df, headers="keys", tablefmt="grid", floatfmt=".4f"))
    
    return classifier_pools


def objective(trial, model_name, X_train, y_train, X_desl, y_desl, X_test, y_test, fold=1, classifier_pool=None):
    """Optuna目标函数，用于搜索最优参数"""
    model_info = fh_models[model_name]
    model_class = model_info['class']
    params = {}
    param_info = model_info['params']

    # 先采样所有分类参数
    categorical_params = {}
    for param_name, param_range in param_info.items():
        if isinstance(param_range, list):
            categorical_params[param_name] = trial.suggest_categorical(param_name, param_range)
    
    # 基于分类参数的条件采样
    # 1. 带宽参数采样
    # 为确保分布一致性，无论bandwidth_selection取值如何，都采样bandwidth参数
    if 'bandwidth' in param_info:
        params['bandwidth'] = trial.suggest_float('bandwidth', param_info['bandwidth'][0], param_info['bandwidth'][1], log=True)
    
    # 2. 密度估计相关参数采样
    # 为确保分布一致性，无论density_estimation取值如何，都采样所有相关参数
    # 让算法本身处理参数之间的依赖关系
    if 'knn_k' in param_info:
        params['knn_k'] = trial.suggest_categorical('knn_k', param_info['knn_k'])
    if 'pca_components' in param_info:
        params['pca_components'] = trial.suggest_categorical('pca_components', param_info['pca_components'])
    if 'use_pca' in categorical_params:
        params['use_pca'] = categorical_params['use_pca']
    
    # 采样剩余的数值参数
    for param_name, param_range in param_info.items():
        # 跳过已经采样的参数
        if param_name in categorical_params or param_name in params:
            continue
            
        if isinstance(param_range, tuple):
            if isinstance(param_range[0], int):
                params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
            else:
                # 增加浮点数采样精度
                # 为每个参数设置固定的分布配置，确保一致性
                if param_name in ['gamma', 'reg_lambda', 'sigma', 'beta', 'bandwidth', 'theta']:
                    # 这些参数使用对数空间采样
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], log=True)
                elif param_name in ['mu', 'density_weight', 'basic_weight']:
                    # 对于权重参数，使用均匀采样，步长为0.05
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1], step=0.05)
                elif param_name in ['alpha', 'epsilon']:
                    # 这些参数使用均匀采样
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                else:
                    # 默认使用均匀采样
                    params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
    
    # 合并所有参数
    params.update(categorical_params)
    
    # 确保参数值不会太长，避免数据库存储问题
    for param_name, value in params.items():
        if isinstance(value, float):
            # 限制浮点数精度
            params[param_name] = round(value, 6)
        elif isinstance(value, str) and len(value) > 100:
            # 截断过长的字符串参数
            params[param_name] = value[:100]

    # 使用预生成的分类器池
    params["pool_classifiers"] = classifier_pool
    
    # 合并固定参数
    full_params = {**params, **fixed_params}

    # 初始化模型
    model = model_class(**full_params)
    model.fit(X_desl, y_desl)

    # 预测并计算AUC作为目标值
    y_pred_proba = np.array(model.predict_proba(X_test))[:, 1]
    if np.isnan(y_pred_proba).any():
        y_pred_proba = np.nan_to_num(y_pred_proba, nan=0.0)
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    return auc_score

def objective_mean_auc(trial, model_name, num_folds=5, classifier_pools_cache=None):
    """计算模型在所有折上的平均AUC"""
    total_auc = 0
    for fold in range(1, num_folds + 1):
        X_train, y_train, X_desl, y_desl, X_test, y_test = get_fold_data(fold)
        auc_score = objective(trial, model_name, X_train, y_train, X_desl, y_desl, X_test, y_test, fold=fold, classifier_pool=classifier_pools_cache[fold])
        total_auc += auc_score
    return total_auc / num_folds

def optimize_fh_models(previous_study_dir=None):
    """对所有模糊超盒模型进行参数优化，得到基于五折数据集均值AUC最优的参数配置"""
    all_best_params = {}
    num_folds = 5
    
    # 预生成并缓存分类器池
    print("开始预生成分类器池...")
    classifier_pools_cache = {}
    
    # 为每个折预生成分类器池
    for fold in range(1, num_folds + 1):
        print(f"\n预生成折 {fold} 的分类器池...")
        # 获取折数据
        X_train, y_train, X_desl, y_desl, X_test, y_test = get_fold_data(fold)
        
        # 生成分类器池
        classifier_pool = generate_classifier_pools(
            models=models,
            X_train=X_train, 
            y_train=y_train, 
            X_test=X_test, 
            y_test=y_test,
            top_k=7, 
            random_state=42,
            fold=fold
        )
        
        # 保存到缓存
        classifier_pools_cache[fold] = classifier_pool
    
    print("\n分类器池预生成完成！")
    
    # 使用高级采样器和剪枝器
    sampler = TPESampler(seed=42, multivariate=True, n_startup_trials=10)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    
    # 获取可用CPU核心数，减少并行度以避免数据库锁定问题
    n_jobs = 1  # 强制单进程运行以避免数据库锁定
    
    # 创建结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f'best_params/clustering_optimized/{timestamp}'
    os.makedirs(result_dir, exist_ok=True)
    
    for model_name in fh_models:
        print(f"Optimizing {model_name} across all folds...")
        
        # 创建study，使用高级采样器和剪枝器
        study_name = f"{model_name}_study"
        storage_name = f"sqlite:///{result_dir}/{model_name}_study.db"
        
        study = None
        # 如果指定了先前的研究目录，则尝试从该目录加载
        if previous_study_dir and os.path.exists(previous_study_dir):
            previous_storage_name = f"sqlite:///{previous_study_dir}/{model_name}_study.db"
            try:
                # 尝试从指定的先前目录加载study
                study = optuna.load_study(study_name=study_name, storage=previous_storage_name)
                print(f"Resuming optimization for {model_name} from {previous_study_dir} with {len(study.trials)} existing trials")
            except Exception as e:
                print(f"Could not load previous study for {model_name}: {e}")
        
        if study is None:
            try:
                # 尝试加载已有的study进行续传
                study = optuna.load_study(study_name=study_name, storage=storage_name)
                print(f"Resuming optimization for {model_name} with {len(study.trials)} existing trials")
            except:
                # 创建新的study
                study = optuna.create_study(
                    direction='maximize',
                    sampler=sampler,
                    pruner=pruner,
                    study_name=study_name,
                    storage=storage_name
                )
        
        # 使用单进程优化以避免数据库锁定问题
        study.optimize(
            lambda trial: objective_mean_auc(trial, model_name, num_folds, classifier_pools_cache), 
            n_trials=200,  # 增加试验次数以提高结果精度
            n_jobs=n_jobs,  # 使用单进程
            # timeout=7200,   # 设置超时时间（秒）
            show_progress_bar=True
        )
        
        # 保存最佳参数
        all_best_params[model_name] = study.best_params.copy()
        
        # 保存study对象以便后续分析
        joblib.dump(study, f"{result_dir}/{model_name}_study.pkl")
        
        # 输出优化结果
        print(f"\nBest trial for {model_name}:")
        print(f"  Value (AUC): {study.best_value:.4f}")
        print(f"  Params: {study.best_params}")
        
        # 可视化并保存结果图表
        try:
            # 参数重要性
            param_importances = optuna.visualization.plot_param_importances(study)
            param_importances.write_image(f"{result_dir}/{model_name}_param_importances.png")
            
            # 优化历史
            optimization_history = optuna.visualization.plot_optimization_history(study)
            optimization_history.write_image(f"{result_dir}/{model_name}_optimization_history.png")
        except Exception as e:
            print(f"Warning: Could not generate visualizations: {e}")

    # 保存最优参数到文件
    with open(f'{result_dir}/ACLF_clustering_fh_best_params.json', 'w', encoding='utf-8') as f:
        json.dump(all_best_params, f, indent=4)
    
    # 同时更新主要参数文件
    with open('best_params/clustering_optimized/ACLF_clustering_fh_best_params.json', 'w', encoding='utf-8') as f:
        json.dump(all_best_params, f, indent=4)

    return all_best_params

def extract_best_params_from_studies(studies_folder, output_folder):
    """
    解析指定文件夹下的所有*.pkl study文件，提取每个模型的最优参数，
    并将最优参数保存至单独文件夹下的JSON文件。
    """
    os.makedirs(output_folder, exist_ok=True)
    best_params_all = {}

    for fname in os.listdir(studies_folder):
        if not fname.endswith("_study.pkl"):
            continue
        model_name = fname.replace("_study.pkl", "")
        pkl_path = os.path.join(studies_folder, fname)
        try:
            study = joblib.load(pkl_path)
            best_params = study.best_params.copy()
            best_value = study.best_value
            
            best_params_all[model_name] = best_params

            # # 为每个模型单独保存一份JSON
            # single_output_path = os.path.join(output_folder, f"{model_name}_best_params.json")
            # with open(single_output_path, 'w', encoding='utf-8') as f:
            #     json.dump({"best_params": best_params, "best_value": best_value}, f, indent=4)
            # print(f"已提取 {model_name} 的最优参数至 {single_output_path}")
        except Exception as e:
            print(f"读取 {pkl_path} 失败: {e}")

    # 汇总所有模型的最优参数
    summary_path = os.path.join(output_folder, "ACLF_all_best_params.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(best_params_all, f, indent=4)
    print(f"所有模型最优参数汇总保存至 {summary_path}")



if __name__ == "__main__":
    # 从指定目录加载已有研究，如果存在的话
    # 如果要从先前的研究继续优化，请取消下面一行的注释并设置正确的目录路径
    # previous_study_dir = 'best_params/clustering_optimized/20251207_120000'
    previous_study_dir = None  # 默认不从先前研究加载
    
    # 运行参数优化
    optimize_fh_models(previous_study_dir)

    # # 解析study文件并提取最优参数
    # studies_folder = 'best_params/clustering_optimized/20251207_120000'  # 指定study.pkl所在文件夹
    # output_folder = 'best_params/clustering_optimized/20251207_120000'  # 指定输出文件夹
    # extract_best_params_from_studies(studies_folder, output_folder)
