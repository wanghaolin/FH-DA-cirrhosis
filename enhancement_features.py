import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score, balanced_accuracy_score, recall_score, average_precision_score
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from sdv.single_table import CTGANSynthesizer, CopulaGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.metadata import Metadata
from sdv.sampling import Condition
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate        
# 定义分类器
from functools import partial
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)

class DataEnhancementEvaluator:
    def __init__(self, data_dir, output_dir):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.train_files = None
        self.test_files = None
        self.fold_results = {}
        # 定义目标比例（minor/major）
        # [0.3, 0.5, 0.8] []
        self.target_ratios = [0.3, 0.5, 0.8]
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 定义数据增强方法
        self.enhancement_methods = {
            'original': self._no_enhancement,
            'smote': self._smote_enhancement,
            'adasyn': self._adasyn_enhancement,
            'borderline_smote': self._borderline_smote_enhancement,
            'svm_smote': self._svm_smote_enhancement,
            'kmeans_smote': self._kmeans_smote_enhancement,
            'gaussian_copula': self._gaussian_copula_enhancement,
            # 'ctgan': self._ctgan_enhancement,
            # 'tvae': self._tvae_enhancement,
            # 'copulagan': self._copulagan_enhancement
        }

        self.classifiers = {
            'NB': partial(GaussianNB),
            'KNN': partial(KNeighborsClassifier, weights='distance',  n_neighbors=3, algorithm='ball_tree'),
            'LR': partial(LogisticRegression, class_weight='balanced', penalty='l1', solver='liblinear', C=0.3, random_state=42),
            'LDA': partial(LinearDiscriminantAnalysis, solver='svd', tol=1e-4),
            'QDA': partial(QuadraticDiscriminantAnalysis, reg_param=0.2, store_covariance=True),
            'RF': partial(RandomForestClassifier, class_weight='balanced',
                          max_depth=10, min_samples_leaf=10, max_features='sqrt',
                          n_estimators=300, random_state=42),
            'ET': partial(ExtraTreesClassifier, class_weight='balanced', max_depth=10, bootstrap=True, max_samples=0.6, random_state=42),
            'ADA': partial(AdaBoostClassifier, n_estimators=300, learning_rate=0.05, random_state=42),
            'GBC': partial(GradientBoostingClassifier, max_depth=5, learning_rate=0.1, n_iter_no_change=10, random_state=42),
            'LGBM': partial(LGBMClassifier, class_weight='balanced', boosting_type='dart',
                            num_leaves=31, reg_alpha=0.1, reg_lambda=0.1, force_col_wise=True, verbosity=-1, random_state=42),
            'XGB': partial(XGBClassifier, max_delta_step=1, reg_alpha=0.1, reg_lambda=0.5,
                           eta=0.07, eval_metric='logloss', random_state=42)
        }
    
    def load_data_files(self):
        """加载五折训练和测试数据集文件路径"""
        files = os.listdir(self.data_dir)
        self.train_files = sorted([f for f in files if 'train' in f and f.endswith('.csv')])
        self.test_files = sorted([f for f in files if 'test' in f and f.endswith('.csv')])
        
        print(f"找到 {len(self.train_files)} 个训练文件和 {len(self.test_files)} 个测试文件")
        return self.train_files, self.test_files
    
    def load_dataset(self, fold_idx):
        """加载指定折的训练和测试数据集"""
        if fold_idx >= len(self.train_files) or fold_idx >= len(self.test_files):
            raise ValueError(f"折索引 {fold_idx} 超出范围")
        
        train_path = os.path.join(self.data_dir, self.train_files[fold_idx])
        test_path = os.path.join(self.data_dir, self.test_files[fold_idx])
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # 假设最后一列是目标变量
        X_train = train_df.iloc[:, :-1]
        y_train = train_df.iloc[:, -1]
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        
        # 确保目标变量是数值型
        if not pd.api.types.is_numeric_dtype(y_train):
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        
        return X_train, y_train, X_test, y_test
    
    def _no_enhancement(self, X, y):
        """不进行数据增强"""
        return X, y
    
    def _smote_enhancement(self, X, y, target_ratio=None):
        """使用SMOTE进行数据增强"""
        if target_ratio is None:
            # 默认等比例增强
            smote = SMOTE(random_state=42)
        else:
            # 计算目标数量
            classes, counts = np.unique(y, return_counts=True)
            majority_count = max(counts)
            minority_count = min(counts)
            minority_class = classes[np.argmin(counts)]
            
            # 计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
            
            # 如果不需要生成样本，直接返回原始数据
            if samples_to_generate <= 0:
                return X, y
            
            smote = SMOTE(sampling_strategy={minority_class: target_minority_count}, random_state=42)
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _adasyn_enhancement(self, X, y, target_ratio=None):
        """使用ADASYN进行数据增强"""
        if target_ratio is None:
            # 默认等比例增强
            adasyn = ADASYN(random_state=42)
        else:
            # 计算目标数量
            classes, counts = np.unique(y, return_counts=True)
            majority_count = max(counts)
            minority_count = min(counts)
            minority_class = classes[np.argmin(counts)]
            
            # 计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
            
            # 如果不需要生成样本，直接返回原始数据
            if samples_to_generate <= 0:
                return X, y
            
            # ADASYN不直接支持目标数量，但可以通过ratio参数控制
            # 计算ratio: 目标少数类数量 / 原始多数类数量
            ratio = target_minority_count / majority_count
            adasyn = ADASYN(sampling_strategy=ratio, random_state=42)
        
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _borderline_smote_enhancement(self, X, y, target_ratio=None):
        """使用Borderline SMOTE进行数据增强"""
        if target_ratio is None:
            # 默认等比例增强
            borderline_smote = BorderlineSMOTE(random_state=42, kind='borderline-1')
        else:
            # 计算目标数量
            classes, counts = np.unique(y, return_counts=True)
            majority_count = max(counts)
            minority_count = min(counts)
            minority_class = classes[np.argmin(counts)]
            
            # 计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
            
            # 如果不需要生成样本，直接返回原始数据
            if samples_to_generate <= 0:
                return X, y
            
            borderline_smote = BorderlineSMOTE(sampling_strategy={minority_class: target_minority_count}, 
                                             random_state=42, kind='borderline-1')
        
        X_resampled, y_resampled = borderline_smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _svm_smote_enhancement(self, X, y, target_ratio=None):
        """使用SVM SMOTE进行数据增强"""
        if target_ratio is None:
            # 默认等比例增强
            svm_smote = SVMSMOTE(random_state=42)
        else:
            # 计算目标数量
            classes, counts = np.unique(y, return_counts=True)
            majority_count = max(counts)
            minority_count = min(counts)
            minority_class = classes[np.argmin(counts)]
            
            # 计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
            
            # 如果不需要生成样本，直接返回原始数据
            if samples_to_generate <= 0:
                return X, y
            
            svm_smote = SVMSMOTE(sampling_strategy={minority_class: target_minority_count}, random_state=42)
        
        X_resampled, y_resampled = svm_smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _kmeans_smote_enhancement(self, X, y, target_ratio=None):
        """使用KMeans SMOTE进行数据增强"""
        if target_ratio is None:
            # 默认等比例增强
            kmeans_smote = KMeansSMOTE(random_state=42, cluster_balance_threshold=0.1)
        else:
            # 计算目标数量
            classes, counts = np.unique(y, return_counts=True)
            majority_count = max(counts)
            minority_count = min(counts)
            minority_class = classes[np.argmin(counts)]
            
            # 计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
            
            # 如果不需要生成样本，直接返回原始数据
            if samples_to_generate <= 0:
                return X, y
            
            kmeans_smote = KMeansSMOTE(sampling_strategy={minority_class: target_minority_count}, 
                                     random_state=42, 
                                     cluster_balance_threshold=0.1)
        
        X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def _gaussian_copula_enhancement(self, X, y, target_ratio=None):
        """使用Gaussian Copula进行数据增强，仅条件合成目标变量为1的样本"""
        # 合并特征和标签用于SDV模型
        train_data = pd.DataFrame(X.copy())
        train_data['target'] = y
        metadata = Metadata.detect_from_dataframe(data=train_data)
        # 训练Gaussian Copula模型
        model = GaussianCopulaSynthesizer(metadata)
        model.fit(train_data)
        
        # 计算类别分布
        classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_count = min(counts)
        minority_class = classes[np.argmin(counts)]
        
        if target_ratio is None:
            # 默认增强：生成少数类样本，使其数量与多数类相同
            samples_to_generate = majority_count - minority_count
        else:
            # 根据目标比例计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
        
        # 如果不需要生成样本，直接返回原始数据
        if samples_to_generate <= 0:
            return X, y
        
        # Step 1: Create Your Conditions - 仅对目标变量为1的数据进行条件采样
        # 确保samples_to_generate为正整数
        samples_to_generate = max(1, int(samples_to_generate))
        condition = Condition(
            num_rows=samples_to_generate,
            column_values={'target': 1}
        )
        # Step 2: Sample Synthetic Data - 增加max_tries_per_batch以提高采样成功率
        synthetic_data = model.sample_from_conditions([condition], max_tries_per_batch=500)
        
        # 提取特征（不包括target列）
        X_synthetic = synthetic_data.iloc[:, :-1]
        y_synthetic = synthetic_data['target']
        
        # 合并原始数据和合成的少数类数据
        X_combined = pd.concat([X, X_synthetic], axis=0)
        y_combined = pd.concat([pd.Series(y), y_synthetic], axis=0)
        
        return X_combined, y_combined
    
    def _ctgan_enhancement(self, X, y, target_ratio=None):
        """使用CTGAN进行数据增强，仅条件合成目标变量为1的样本"""
        train_data = pd.DataFrame(X.copy())
        train_data['target'] = y
        
        metadata = Metadata.detect_from_dataframe(data=train_data)
        # 降低模型复杂度以减少内存使用
        model = CTGANSynthesizer(metadata, epochs=300, cuda=False, batch_size=500)
        model.fit(train_data)
        
        # 计算类别分布
        classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_count = min(counts)
        minority_class = classes[np.argmin(counts)]
        
        if target_ratio is None:
            # 默认增强：生成少数类样本，使其数量与多数类相同
            samples_to_generate = majority_count - minority_count
        else:
            # 根据目标比例计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
        
        # 如果不需要生成样本，直接返回原始数据
        if samples_to_generate <= 0:
            return X, y
        
        # Step 1: Create Your Conditions - 仅对目标变量为1的数据进行条件采样
        # 确保samples_to_generate为正整数
        samples_to_generate = max(1, int(samples_to_generate))
        condition = Condition(
            num_rows=samples_to_generate,
            column_values={'target': 1}
        )
        # Step 2: Sample Synthetic Data - 增加max_tries_per_batch以提高采样成功率
        synthetic_data = model.sample_from_conditions([condition], max_tries_per_batch=100)
        
        # 提取特征（不包括target列）
        X_synthetic = synthetic_data.iloc[:, :-1]
        y_synthetic = synthetic_data['target']
        
        # 合并原始数据和合成的少数类数据
        X_combined = pd.concat([X, X_synthetic], axis=0)
        y_combined = pd.concat([pd.Series(y), y_synthetic], axis=0)
        
        return X_combined, y_combined
    
    def _tvae_enhancement(self, X, y, target_ratio=None):
        """使用TVAE进行数据增强，仅条件合成目标变量为1的样本"""
        train_data = pd.DataFrame(X.copy())
        train_data['target'] = y
        metadata = Metadata.detect_from_dataframe(data=train_data)
        # 降低模型复杂度以减少内存使用
        model = TVAESynthesizer(metadata, epochs=300, cuda=False, batch_size=256)
        model.fit(train_data)
        
        # 计算类别分布
        classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_count = min(counts)
        minority_class = classes[np.argmin(counts)]
        
        if target_ratio is None:
            # 默认增强：生成少数类样本，使其数量与多数类相同
            samples_to_generate = majority_count - minority_count
        else:
            # 根据目标比例计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
        
        # 如果不需要生成样本，直接返回原始数据
        if samples_to_generate <= 0:
            return X, y
        
        # Step 1: Create Your Conditions - 仅对目标变量为1的数据进行条件采样
        # 确保samples_to_generate为正整数
        samples_to_generate = max(1, int(samples_to_generate))
        condition = Condition(
            num_rows=samples_to_generate,
            column_values={'target': 1}
        )
        # Step 2: Sample Synthetic Data - 增加max_tries_per_batch以提高采样成功率
        synthetic_data = model.sample_from_conditions([condition], max_tries_per_batch=700)
        
        # 提取特征（不包括target列）
        X_synthetic = synthetic_data.iloc[:, :-1]
        y_synthetic = synthetic_data['target']
        
        # 合并原始数据和合成的少数类数据
        X_combined = pd.concat([X, X_synthetic], axis=0)
        y_combined = pd.concat([pd.Series(y), y_synthetic], axis=0)
        
        return X_combined, y_combined
    
    def _copulagan_enhancement(self, X, y, target_ratio=None):
        """使用CopulaGAN进行数据增强，仅条件合成目标变量为1的样本"""
        train_data = pd.DataFrame(X.copy())
        train_data['target'] = y
        metadata = Metadata.detect_from_dataframe(data=train_data)
        # 降低模型复杂度以减少内存使用
        model = CopulaGANSynthesizer(metadata, epochs=300, cuda=False, batch_size=256)
        model.fit(train_data)
        
        # 计算类别分布
        classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_count = min(counts)
        minority_class = classes[np.argmin(counts)]
        
        if target_ratio is None:
            # 默认增强：生成少数类样本，使其数量与多数类相同
            samples_to_generate = majority_count - minority_count
        else:
            # 根据目标比例计算需要生成的少数类样本数量
            target_minority_count = int(majority_count * target_ratio)
            samples_to_generate = max(0, target_minority_count - minority_count)
        
        # 如果不需要生成样本，直接返回原始数据
        if samples_to_generate <= 0:
            return X, y
        
        # Step 1: Create Your Conditions - 仅对目标变量为1的数据进行条件采样
        # 确保samples_to_generate为正整数
        samples_to_generate = max(1, int(samples_to_generate))
        condition = Condition(
            num_rows=samples_to_generate,
            column_values={'target': 1}
        )
        # Step 2: Sample Synthetic Data
        synthetic_data = model.sample_from_conditions([condition])
        
        # 提取特征（不包括target列）
        X_synthetic = synthetic_data.iloc[:, :-1]
        y_synthetic = synthetic_data['target']
        
        # 合并原始数据和合成的少数类数据
        X_combined = pd.concat([X, X_synthetic], axis=0)
        y_combined = pd.concat([pd.Series(y), y_synthetic], axis=0)
        
        return X_combined, y_combined
    
    def optimize_threshold(self, y_true, y_pred_proba, method='youden'):
        """使用约登系数优化分类阈值"""
        if method == 'youden':
            # 传统约登指数法(ROC曲线)
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            y = tpr - fpr
            optimal_idx = np.argmax(y)
            best_threshold = thresholds[optimal_idx]
            return best_threshold
        else:
            return 0.5
    
    def evaluate_model(self, clf, X_train, y_train, X_test, y_test):
        """评估模型性能"""
        
        # 训练模型
        clf.fit(X_train, y_train)
        
        # 获取预测概率
        try:
            y_pred_proba = clf.predict_proba(X_test)[:, 1]
        except:
            y_pred_proba = clf.decision_function(X_test)
            # 归一化概率
            y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        
        # 使用约登系数优化阈值
        optimal_threshold = self.optimize_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # 计算评估指标
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        mcc = matthews_corrcoef(y_test, y_pred)
        gmean = geometric_mean_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        recall_positive = recall_score(y_test, y_pred, pos_label=1)  # 阳性样本的Recall
        pr_auc = average_precision_score(y_test, y_pred_proba)  # PR-AUC
        
        return {
            'F1': f1,
            'AUC': auc,
            'MCC': mcc,
            'GMean': gmean,
            'Accuracy': acc,
            'BalancedAccuracy': balanced_acc,
            'RecallPositive': recall_positive,
            'PR-AUC': pr_auc,
            'Threshold': optimal_threshold
        }
    
    def run_fold_evaluation(self, fold_idx):
        """运行指定折的数据增强和评估"""
        print(f"\n{'='*60}")
        print(f"处理第 {fold_idx + 1} 折数据")
        print(f"{'='*60}")
        
        # 加载数据集
        X_train, y_train, X_test, y_test = self.load_dataset(fold_idx)
        print(f"原始训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"原始类别分布: {np.bincount(y_train)}")
        
        # 计算原始数据中的类别比例
        classes, counts = np.unique(y_train, return_counts=True)
        majority_count = max(counts)
        minority_count = min(counts)
        original_ratio = minority_count / majority_count
        print(f"原始少数类/多数类比例: {original_ratio:.4f}")
        
        fold_results = {}
        
        # 对每种增强方法进行评估
        # 如果 self.target_ratios 为空，视为 [None]（默认比例），以确保仍会调用增强方法一次
        target_ratios = self.target_ratios if self.target_ratios else [None]

        for method_name, enhancement_func in self.enhancement_methods.items():
            try:
                # 原始方法不使用目标比例
                if method_name == 'original':
                    print(f"\n应用 {method_name} 数据增强...")

                    # 应用数据增强
                    X_enhanced, y_enhanced = enhancement_func(X_train, y_train)
                    print(f"增强后训练集大小: {len(X_enhanced)}, 类别分布: {np.bincount(y_enhanced)}")

                    method_results = {}

                    # 对每种分类器进行评估
                    for clf_name, clf in self.classifiers.items():
                        print(f"  使用 {clf_name} 分类器...")
                        # 创建分类器副本以避免状态问题
                        if isinstance(clf, partial):
                            clf_copy = clf()  # 直接调用partial函数创建实例
                        else:
                            clf_copy = clf.__class__(**clf.get_params())
                        metrics = self.evaluate_model(clf_copy, X_enhanced, y_enhanced, X_test, y_test)
                        method_results[clf_name] = metrics

                        print(f"    F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, MCC: {metrics['MCC']:.4f}, GMean: {metrics['GMean']:.4f}, BalancedAccuracy: {metrics['BalancedAccuracy']:.4f}, RecallPositive: {metrics['RecallPositive']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")

                    fold_results[method_name] = method_results
                else:
                    # 对每种目标比例进行评估
                    for target_ratio in target_ratios:
                        if target_ratio is None:
                            print(f"\n应用 {method_name} 数据增强 (目标比例: 默认)...")
                        else:
                            print(f"\n应用 {method_name} 数据增强 (目标比例: {target_ratio:.1f})...")

                        try:
                            # 应用数据增强
                            X_enhanced, y_enhanced = enhancement_func(X_train, y_train, target_ratio)
                        except Exception as e:
                            print(f"  {method_name} 增强失败: {str(e)}")
                            # 如果增强失败，跳过当前比例的处理
                            continue

                        # 计算增强后的类别比例
                        enhanced_classes, enhanced_counts = np.unique(y_enhanced, return_counts=True)
                        enhanced_majority_count = max(enhanced_counts)
                        enhanced_minority_count = min(enhanced_counts)
                        enhanced_ratio = enhanced_minority_count / enhanced_majority_count

                        print(f"增强后训练集大小: {len(X_enhanced)}, 类别分布: {enhanced_counts}")
                        # 在打印目标比例时如果 target_ratio 为 None，避免格式化错误
                        if target_ratio is None:
                            print(f"增强后少数类/多数类比例: {enhanced_ratio:.4f} (目标: 默认)")
                        else:
                            print(f"增强后少数类/多数类比例: {enhanced_ratio:.4f} (目标: {target_ratio:.1f})")

                        # 为该方法和比例创建结果字典键
                        if target_ratio is None:
                            method_ratio_key = f"{method_name}_default"
                        else:
                            method_ratio_key = f"{method_name}_ratio_{int(target_ratio*100)}"
                        method_results = {}

                        # 对每种分类器进行评估
                        for clf_name, clf in self.classifiers.items():
                            print(f"  使用 {clf_name} 分类器...")
                            # 创建分类器副本以避免状态问题
                            if isinstance(clf, partial):
                                clf_copy = clf()  # 直接调用partial函数创建实例
                            else:
                                clf_copy = clf.__class__(**clf.get_params())
                            metrics = self.evaluate_model(clf_copy, X_enhanced, y_enhanced, X_test, y_test)
                            method_results[clf_name] = metrics

                            print(f"    F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, MCC: {metrics['MCC']:.4f}, GMean: {metrics['GMean']:.4f}, BalancedAccuracy: {metrics['BalancedAccuracy']:.4f}, RecallPositive: {metrics['RecallPositive']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")

                        fold_results[method_ratio_key] = method_results

                        # 保存增强后的数据集
                        # 创建对应比例的文件夹（None -> default）
                        if target_ratio is None:
                            ratio_dir = os.path.join(self.output_dir, 'default')
                        else:
                            ratio_dir = os.path.join(self.output_dir, f"ratio_{int(target_ratio*100)}")
                        os.makedirs(ratio_dir, exist_ok=True)

                        enhanced_df = pd.DataFrame(X_enhanced, columns=X_train.columns)
                        enhanced_df['HE'] = y_enhanced
                        save_path = os.path.join(ratio_dir, f"HE_enhanced_{method_name}_fold_{fold_idx + 1}.csv")
                        enhanced_df.to_csv(save_path, index=False)
                        print(f"  增强数据集已保存至: {save_path}")

            except Exception as e:
                print(f"  {method_name} 增强失败: {str(e)}")
                # 处理不同比例的异常情况
                if method_name != 'original':
                    # 使用与上面一致的 target_ratios 列表
                    for target_ratio in target_ratios:
                        if target_ratio is None:
                            method_ratio_key = f"{method_name}_default"
                        else:
                            method_ratio_key = f"{method_name}_ratio_{int(target_ratio*100)}"
                        fold_results[method_ratio_key] = None
                else:
                    fold_results[method_name] = None

        self.fold_results[fold_idx] = fold_results
        return fold_results
    
    def run_full_evaluation(self):
        """运行完整的五折交叉验证评估"""
        self.load_data_files()
        
        for fold_idx in range(5):
            self.run_fold_evaluation(fold_idx)
        
        # 计算平均结果
        self.calculate_average_results()
        
        # 确定最优增强方法
        self.determine_best_enhancement()
    
    def calculate_average_results(self):
        """计算所有折的平均评估结果"""
        avg_results = {}
        
        for method_name in self.enhancement_methods.keys():
            method_avg = {}
            
            for clf_name in self.classifiers.keys():
                # 收集所有折的指标
                fold_metrics = []
                for fold_idx in range(5):
                    if (fold_idx in self.fold_results and 
                        self.fold_results[fold_idx] is not None and 
                        method_name in self.fold_results[fold_idx] and 
                        self.fold_results[fold_idx][method_name] is not None and
                        clf_name in self.fold_results[fold_idx][method_name]):
                        fold_metrics.append(self.fold_results[fold_idx][method_name][clf_name])
                
                if fold_metrics:
                    # 计算平均值
                    avg_metrics = {}
                    for metric in ['F1', 'AUC', 'MCC', 'GMean', 'Accuracy', 'BalancedAccuracy', 'RecallPositive', 'PR-AUC']:
                        avg_metrics[metric] = np.mean([m[metric] for m in fold_metrics])
                    method_avg[clf_name] = avg_metrics
                else:
                    method_avg[clf_name] = None
            
            avg_results[method_name] = method_avg
        
        # 打印平均结果
        print(f"\n{'='*60}")
        print("所有折的平均评估结果")
        print(f"{'='*60}")
        
        for method_name, method_results in avg_results.items():
            print(f"\n{method_name} 增强方法:")
            if method_results:
                table_data = []
                for clf_name, metrics in method_results.items():
                    if metrics:
                        table_data.append([
                            clf_name,
                            f"{metrics['F1']:.4f}",
                            f"{metrics['AUC']:.4f}",
                            f"{metrics['MCC']:.4f}",
                            f"{metrics['GMean']:.4f}",
                            f"{metrics['BalancedAccuracy']:.4f}",
                            f"{metrics['RecallPositive']:.4f}",
                            f"{metrics['PR-AUC']:.4f}"
                        ])
                
                if table_data:
                    print(tabulate(table_data, headers=['分类器', 'F1', 'AUC', 'MCC', 'GMean', 'BalancedAccuracy', 'RecallPositive', 'PR-AUC'], tablefmt='grid'))
        
        self.avg_results = avg_results
    
    def determine_best_enhancement(self):
        """确定最优的数据增强方法和目标比例"""
        # 该方法功能已被determine_best_enhanced_dataset替代，保留以确保向后兼容
        pass
    
    def evaluate_enhanced_datasets(self):
        """评估原始数据集和增强数据集文件夹中的所有数据集"""
        print(f"\n{'='*60}")
        print("评估原始数据集和增强数据集文件夹中的所有数据集")
        print(f"{'='*60}")
        
        # 先评估原始数据集
        print("\n评估原始数据集")
        self.load_data_files()
        original_results = {}
        
        for fold_idx in range(5):
            print(f"  处理折 {fold_idx + 1}")
            try:
                X_train, y_train, X_test, y_test = self.load_dataset(fold_idx)
                for clf_name, clf in self.classifiers.items():
                    print(f"    使用 {clf_name} 分类器...")
                    if isinstance(clf, partial):
                        clf_copy = clf()
                    else:
                        clf_copy = clf.__class__(**clf.get_params())
                    try:
                        metrics = self.evaluate_model(clf_copy, X_train, y_train, X_test, y_test)
                        if clf_name not in original_results:
                            original_results[clf_name] = []
                        original_results[clf_name].append(metrics)
                        print(f"      F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, MCC: {metrics['MCC']:.4f}, GMean: {metrics['GMean']:.4f}, BalancedAccuracy: {metrics['BalancedAccuracy']:.4f}, RecallPositive: {metrics['RecallPositive']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
                    except Exception as e:
                        print(f"      分类器 {clf_name} 评估失败: {str(e)}")
            except Exception as e:
                print(f"  加载数据失败: {str(e)}")
        
        # 初始化评估结果存储
        eval_results = {}
        
        # 扫描不同比例文件夹中的增强数据集
        # 如果 self.target_ratios 为空，则视为 [None]（默认），确保扫描 default 文件夹
        scan_target_ratios = self.target_ratios if self.target_ratios else [None]
        for target_ratio in scan_target_ratios:
            if target_ratio is None:
                ratio_dir = os.path.join(self.output_dir, 'default')
            else:
                ratio_dir = os.path.join(self.output_dir, f"ratio_{int(target_ratio*100)}")
            
            if os.path.exists(ratio_dir):
                # 打印时若 target_ratio 为 None，显示为 默认
                ratio_label = f"{int(target_ratio*100)}%" if target_ratio is not None else "默认"
                print(f"\n扫描比例 {ratio_label} 的数据集文件夹: {ratio_dir}")
                
                enhanced_files = [f for f in os.listdir(ratio_dir) 
                                if f.startswith('HE_enhanced_') and f.endswith('.csv')]
                
                if not enhanced_files:
                    ratio_label = f"{int(target_ratio*100)}%" if target_ratio is not None else "默认"
                    print(f"  未找到比例 {ratio_label} 的增强数据集文件")
                else:
                    ratio_label = f"{int(target_ratio*100)}%" if target_ratio is not None else "默认"
                    print(f"  找到 {len(enhanced_files)} 个比例 {ratio_label} 的增强数据集文件")
                    
                    # 按增强方法分组
                    method_files = {}
                    for file in enhanced_files:
                        # 提取增强方法和折数信息
                        parts = file.split('_')
                        if len(parts) >= 5 and 'fold' in parts and parts[-1].endswith('.csv'):
                            # 找到fold的位置
                            fold_index = parts.index('fold')
                            if fold_index + 1 < len(parts):
                                method_name = '_'.join(parts[2:fold_index])  # 处理可能有下划线的方法名
                                # 提取折数并转换为0-based索引
                                fold_number = parts[fold_index + 1].replace('.csv', '')
                                if fold_number.isdigit():
                                    fold_idx = int(fold_number) - 1
                                    
                                    # 构建包含比例信息的方法名键
                                    method_ratio_key = f"{method_name}_ratio_{int(target_ratio*100)}"
                                    
                                    if method_ratio_key not in method_files:
                                        method_files[method_ratio_key] = {}
                                    method_files[method_ratio_key][fold_idx] = (file, ratio_dir)
                                    ratio_label = f"{int(target_ratio*100)}%" if target_ratio is not None else "默认"
                                    print(f"  识别到文件: {file}, 方法: {method_name}, 比例: {ratio_label}, 折: {fold_idx + 1}")
                    
                    # 调试信息：打印识别到的增强方法数量
                    print(f"  识别到 {len(method_files)} 种带比例的增强方法组合")
                    
                    # 对每种增强方法进行评估
                    for method_ratio_key, fold_files in method_files.items():
                        # 从键中提取原始方法名
                        original_method_name = method_ratio_key.split('_ratio_')[0]
                        ratio_value = method_ratio_key.split('_ratio_')[1]
                        
                        print(f"\n评估增强方法: {original_method_name} (比例: {ratio_value}%)")
                        method_results = {}
                        
                        # 对每个折进行评估
                        for fold_idx, (file_name, file_dir) in fold_files.items():
                            print(f"  处理折 {fold_idx + 1}: {file_name}")
                            
                            # 加载增强数据集
                            enhanced_data_path = os.path.join(file_dir, file_name)
                            enhanced_df = pd.read_csv(enhanced_data_path)
                            
                            # 分离特征和目标变量
                            X_enhanced = enhanced_df.iloc[:, :-1]
                            y_enhanced = enhanced_df.iloc[:, -1]
                            
                            # 加载对应的测试集
                            try:
                                _, _, X_test, y_test = self.load_dataset(fold_idx)
                                
                                # 对每种分类器进行评估
                                for clf_name, clf in self.classifiers.items():
                                    print(f"    使用 {clf_name} 分类器...")
                                    
                                    # 创建分类器副本
                                    if isinstance(clf, partial):
                                        clf_copy = clf()
                                    else:
                                        clf_copy = clf.__class__(**clf.get_params())
                                    
                                    try:
                                        # 评估模型
                                        metrics = self.evaluate_model(clf_copy, X_enhanced, y_enhanced, X_test, y_test)
                                        
                                        # 存储结果
                                        if clf_name not in method_results:
                                            method_results[clf_name] = []
                                        method_results[clf_name].append(metrics)
                                        
                                        print(f"      F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, MCC: {metrics['MCC']:.4f}, GMean: {metrics['GMean']:.4f}, BalancedAccuracy: {metrics['BalancedAccuracy']:.4f}, RecallPositive: {metrics['RecallPositive']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
                                    except Exception as e:
                                        print(f"      分类器 {clf_name} 评估失败: {str(e)}")
                            except Exception as e:
                                print(f"  加载测试集失败: {str(e)}")
                        
                        eval_results[method_ratio_key] = method_results
            else:
                ratio_label = f"{int(target_ratio*100)}%" if target_ratio is not None else "默认"
                print(f"\n比例 {ratio_label} 的数据集文件夹不存在: {ratio_dir}")
        
        # 检查根目录是否还有旧格式的增强文件
        root_enhanced_files = [f for f in os.listdir(self.output_dir) 
                             if f.startswith('HE_enhanced_') and f.endswith('.csv')]
        
        if root_enhanced_files:
            print(f"\n在根目录找到 {len(root_enhanced_files)} 个旧格式的增强数据集文件")
            
            # 按增强方法分组
            method_files = {}
            for file in root_enhanced_files:
                # 提取增强方法和折数信息
                parts = file.split('_')
                if len(parts) >= 5 and 'fold' in parts and parts[-1].endswith('.csv'):
                    # 找到fold的位置
                    fold_index = parts.index('fold')
                    if fold_index + 1 < len(parts):
                        method_name = '_'.join(parts[2:fold_index])  # 处理可能有下划线的方法名
                        # 提取折数并转换为0-based索引
                        fold_number = parts[fold_index + 1].replace('.csv', '')
                        if fold_number.isdigit():
                            fold_idx = int(fold_number) - 1
                            
                            if method_name not in method_files:
                                method_files[method_name] = {}
                            method_files[method_name][fold_idx] = (file, self.output_dir)
                            print(f"  识别到文件: {file}, 方法: {method_name}, 折: {fold_idx + 1}")
            
            # 对每种增强方法进行评估
            for method_name, fold_files in method_files.items():
                print(f"\n评估增强方法: {method_name} (未指定比例)")
                method_results = {}
                
                # 对每个折进行评估
                for fold_idx, (file_name, file_dir) in fold_files.items():
                    print(f"  处理折 {fold_idx + 1}: {file_name}")
                    
                    # 加载增强数据集
                    enhanced_data_path = os.path.join(file_dir, file_name)
                    enhanced_df = pd.read_csv(enhanced_data_path)
                    
                    # 分离特征和目标变量
                    X_enhanced = enhanced_df.iloc[:, :-1]
                    y_enhanced = enhanced_df.iloc[:, -1]
                    
                    # 加载对应的测试集
                    try:
                        _, _, X_test, y_test = self.load_dataset(fold_idx)
                        
                        # 对每种分类器进行评估
                        for clf_name, clf in self.classifiers.items():
                            print(f"    使用 {clf_name} 分类器...")
                            
                            # 创建分类器副本
                            if isinstance(clf, partial):
                                clf_copy = clf()
                            else:
                                clf_copy = clf.__class__(**clf.get_params())
                            
                            try:
                                # 评估模型
                                metrics = self.evaluate_model(clf_copy, X_enhanced, y_enhanced, X_test, y_test)
                                
                                # 存储结果
                                if clf_name not in method_results:
                                    method_results[clf_name] = []
                                method_results[clf_name].append(metrics)
                                
                                print(f"      F1: {metrics['F1']:.4f}, AUC: {metrics['AUC']:.4f}, MCC: {metrics['MCC']:.4f}, GMean: {metrics['GMean']:.4f}, BalancedAccuracy: {metrics['BalancedAccuracy']:.4f}, RecallPositive: {metrics['RecallPositive']:.4f}, PR-AUC: {metrics['PR-AUC']:.4f}")
                            except Exception as e:
                                print(f"      分类器 {clf_name} 评估失败: {str(e)}")
                    except Exception as e:
                        print(f"  加载测试集失败: {str(e)}")
                
                eval_results[method_name] = method_results
        
        # 计算平均结果
        print(f"\n{'='*60}")
        print("所有数据集评估的平均结果")
        print(f"{'='*60}")
        
        avg_results = {}
        
        # 计算原始数据集平均结果
        original_avg = {}
        if original_results:
            for clf_name, fold_metrics_list in original_results.items():
                if fold_metrics_list:
                    avg_metrics = {}
                    for metric in ['F1', 'AUC', 'MCC', 'GMean', 'Accuracy', 'BalancedAccuracy', 'RecallPositive', 'PR-AUC']:
                        avg_metrics[metric] = np.mean([m[metric] for m in fold_metrics_list])
                    original_avg[clf_name] = avg_metrics
        avg_results['original'] = original_avg
        
        # 打印原始数据集评估结果
        print("\n原始数据集评估结果:")
        self._print_results_table(original_avg)
        
        # 计算增强数据集平均结果
        if eval_results:  # 直接检查eval_results是否为空，而不是检查是否在locals()中
            for method_name, method_results in eval_results.items():
                print(f"\n{method_name} 增强方法:")
                method_avg = {}
                
                for clf_name, fold_metrics_list in method_results.items():
                    if fold_metrics_list:
                        # 计算平均值
                        avg_metrics = {}
                        for metric in ['F1', 'AUC', 'MCC', 'GMean', 'Accuracy', 'BalancedAccuracy', 'RecallPositive', 'PR-AUC']:
                            avg_metrics[metric] = np.mean([m[metric] for m in fold_metrics_list])
                        method_avg[clf_name] = avg_metrics
                
                avg_results[method_name] = method_avg
                
                # 打印增强数据集评估结果
                self._print_results_table(method_avg)

                # 将每种比例每种增强数据集对应的五折均值最终的评估结果保存至相应的csv文件
                try:
                    # 将方法名中的特殊字符替换为下划线，确保文件名合法
                    safe_method_name = method_name.replace('/', '_').replace('\\', '_').replace(':', '_')
                    results_csv_file = os.path.join(self.output_dir, f'{safe_method_name}_evaluation_results.csv')
                    # 将评估结果转换为DataFrame
                    df = pd.DataFrame.from_dict(method_avg, orient='index')
                    # 展开嵌套的指标字典
                    df = pd.concat([df[col].apply(pd.Series) for col in df.columns], axis=1)
                    df.to_csv(results_csv_file)
                    print(f"评估结果已保存至: {results_csv_file}")
                except Exception as e:
                    print(f"保存 {method_name} 的评估结果失败: {str(e)}")
        
        # 保存评估结果
        try:
            results_file = os.path.join(self.output_dir, 'enhanced_datasets_evaluation_results.txt')
            with open(results_file, 'w') as f:
                f.write("原始数据集和增强数据集评估结果\n")
                f.write("=" * 60 + "\n\n")
                
                for method_name, method_results in avg_results.items():
                    f.write(f"{method_name} 数据集:\n")
                    for clf_name, metrics in method_results.items():
                        f.write(f"  {clf_name}:\n")
                        for metric, value in metrics.items():
                            f.write(f"    {metric}: {value:.4f}\n")
                    f.write("\n")
            
            print(f"\n评估结果已保存至: {results_file}")
        except Exception as e:
            print(f"保存评估结果失败: {str(e)}")
        
        # 确定最优数据增强方法
        self.determine_best_enhanced_dataset(avg_results)
        
        return avg_results
    def _print_results_table(self, results):
        """打印评估结果表格"""
        table_data = []
        for clf_name, metrics in results.items():
            table_data.append([
                clf_name,
                f"{metrics['F1']:.4f}",
                f"{metrics['AUC']:.4f}",
                f"{metrics['MCC']:.4f}",
                f"{metrics['GMean']:.4f}",
                f"{metrics['BalancedAccuracy']:.4f}",
                f"{metrics['RecallPositive']:.4f}",
                f"{metrics['PR-AUC']:.4f}"
            ])
        if table_data:
            print(tabulate(table_data, headers=['分类器', 'F1', 'AUC', 'MCC', 'GMean', 'BalancedAccuracy', 'RecallPositive', 'PR-AUC'], tablefmt='grid'))
    
    def determine_best_enhanced_dataset(self, avg_results):
        """确定最优的数据增强方法
        
        Args:
            avg_results: 包含所有数据集评估结果的字典
            
        Returns:
            best_method: 最优数据增强方法名称
            best_score: 最优数据增强方法的综合评分
            best_metrics: 最优数据增强方法的平均性能指标
        """
        print(f"\n{'='*60}")
        print("确定最优数据增强方法")
        print(f"{'='*60}")
        
        # 为每种数据增强方法计算综合评分
        method_scores = {}
        method_overall_metrics = {}
        
        for method_name, method_results in avg_results.items():
            if method_results:
                # 收集所有分类器的指标
                all_metrics = []
                for clf_metrics in method_results.values():
                    if clf_metrics:
                        all_metrics.append(clf_metrics)
                
                if all_metrics:
                    # 计算该方法在所有分类器上的平均指标
                    overall_metrics = {}
                    for metric in ['F1', 'AUC', 'MCC', 'GMean', 'Accuracy', 'BalancedAccuracy', 'RecallPositive', 'PR-AUC']:
                        if metric in all_metrics[0]:
                            overall_metrics[metric] = np.mean([m[metric] for m in all_metrics])
                    
                    # 使用F1、MCC、GMean、BalancedAccuracy的加权平均作为综合评分
                    # 每个指标权重相等，各占25%
                    combined_score = 0.25 * overall_metrics.get('F1', 0) + \
                                    0.25 * overall_metrics.get('MCC', 0) + \
                                    0.25 * overall_metrics.get('GMean', 0) + \
                                    0.25 * overall_metrics.get('BalancedAccuracy', 0)
                    
                    method_scores[method_name] = combined_score
                    method_overall_metrics[method_name] = overall_metrics
        
        # 确定最优方法
        if method_scores:
            best_method = max(method_scores, key=method_scores.get)
            best_score = method_scores[best_method]
            best_metrics = method_overall_metrics[best_method]
            
            # 打印最优方法结果
            print(f"\n最优数据增强方法: {best_method}")
            print(f"综合评分: {best_score:.4f}")
            print(f"平均性能指标:")
            for metric, value in best_metrics.items():
                print(f"  {metric}: {value:.4f}")
            
            # 打印所有方法的综合评分对比
            print(f"\n{'='*60}")
            print("各数据增强方法综合评分对比")
            print(f"{'='*60}")
            
            # 按评分排序
            sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
            
            # 打印排序后的结果
            table_data = []
            for method_name, score in sorted_methods:
                metrics = method_overall_metrics[method_name]
                table_data.append([
                    method_name,
                    f"{score:.4f}",
                    f"{metrics.get('F1', 'N/A'):.4f}",
                    f"{metrics.get('AUC', 'N/A'):.4f}",
                    f"{metrics.get('MCC', 'N/A'):.4f}",
                    f"{metrics.get('GMean', 'N/A'):.4f}",
                    f"{metrics.get('BalancedAccuracy', 'N/A'):.4f}",
                    f"{metrics.get('RecallPositive', 'N/A'):.4f}",
                    f"{metrics.get('PR-AUC', 'N/A'):.4f}"
                ])
            
            print(tabulate(table_data, headers=['数据增强方法', '综合评分', '平均F1', '平均AUC', '平均MCC', '平均GMean', '平均BalancedAccuracy', '平均RecallPositive', '平均PR-AUC'], tablefmt='grid'))
            
            # 保存结果到文件
            try:
                results_file = os.path.join(self.output_dir, 'best_enhanced_dataset_summary.txt')
                with open(results_file, 'w') as f:
                    f.write("最优数据增强方法分析\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write(f"最优数据增强方法: {best_method}\n")
                    f.write(f"综合评分: {best_score:.4f}\n")
                    f.write(f"平均性能指标:\n")
                    for metric, value in best_metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                    
                    f.write("\n各数据增强方法综合评分对比\n")
                    f.write("=" * 60 + "\n")
                    
                    for method_name, score in sorted_methods:
                        f.write(f"{method_name}: {score:.4f}\n")
                        metrics = method_overall_metrics[method_name]
                        for metric, value in metrics.items():
                            f.write(f"  {metric}: {value:.4f}\n")
                        f.write("\n")
                
                print(f"\n最优数据增强方法分析结果已保存至: {results_file}")
            except Exception as e:
                print(f"保存最优数据增强方法分析结果失败: {str(e)}")
            
            return best_method, best_score, best_metrics
        else:
            print("未找到有效的评估结果，无法确定最优数据增强方法")
            return None, None, None


if __name__ == "__main__":
    # 设置数据目录和输出目录
    data_dir = os.path.join("filtered_data", "HE")
    output_dir = os.path.join("filtered_data", "HE", "enhance")
    
    # 创建评估器实例
    evaluator = DataEnhancementEvaluator(data_dir, output_dir)
    
    # 运行评估
    evaluator.run_full_evaluation()
    
    # # 评估增强数据集文件夹中的数据
    # evaluator.evaluate_enhanced_datasets()