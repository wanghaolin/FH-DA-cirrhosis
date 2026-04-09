import numpy as np
import time
import json
import os
import optuna

from sklearn.metrics import roc_auc_score

from my_deslib.des.base import BaseDES
from data_characteristics import (
    calculate_data_imbalance_ratio,
    calculate_data_density,
    estimate_noise_level,
    calculate_feature_correlation,
    calculate_data_dispersion,
    calculate_mutual_information,
    calculate_outlier_ratio,
    calculate_hard_sample_ratio
)

# 导入所有需要的超椭球体策略类
try:
    from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he import FHDES_JFB_vector_he
    from my_deslib.des.hyperellipsoid_versions.des_FHMW_JFB_vector_he import DESFHMW_JFB_vector_he
    from my_deslib.des.hyperellipsoid_versions.fh_des_AllBoxes_vector_he import FHDES_Allboxes_vector_he
    from my_deslib.des.hyperellipsoid_versions.des_FHMW_AllBoxes_vector_he import DESFHMW_allboxes_vector_he
    from my_deslib.des.hyperellipsoid_versions.fh_des_prior_vector_he import FHDES_prior_vector_he
    from my_deslib.des.hyperellipsoid_versions.des_FHMW_prior_vector_he import DESFHMW_prior_vector_he
except ImportError as e:
    print(f"导入超椭球体策略类时出错: {e}")
    # 在实际使用中，需要确保所有这些类都能被正确导入


class AdaptiveFHDesHE(BaseDES):
    """
    自适应的基于马氏距离的模糊超椭球体集成学习器
    能够根据数据集特点动态选择超椭球体构建策略
    """
    
    _estimator_type = "classifier"
    
    # 直接覆盖__sklearn_tags__方法，不调用super()
    def __sklearn_tags__(self):
        return {
            'allow_nan': False,
            'requires_fit': True,
            'requires_y': True,
            'binary_only': False,
            '_skip_test': True
        }
    
    def fit(self, X, y):
        """
        重写fit方法，确保在调用父类fit之前设置classes_属性
        """
        # 设置classes_属性，避免check_is_fitted错误
        self.classes_ = np.unique(y)
        
        # 调用父类的fit方法
        return super().fit(X, y)
    
    def __init__(
        self,
        pool_classifiers=None,
        k=7,
        DFP=False,
        with_IH=False,
        safe_k=None,
        IH_rate=0.30,
        random_state=None,
        knn_classifier='knn',
        DSEL_perc=0.5,
        strategy_selection_method='auto',
        multiCore_process=False,
        dataset_features=None,
        load_rules=False,  # 不再加载规则，直接使用if-else逻辑
        rules_file_path='rules_output/HE/decision_rules_ranges.json',  # 规则文件路径参数
        reg_lambda=1e-4,  # 正则化参数，用于确保协方差矩阵可逆
        gamma=1.0  # 高斯函数参数，用于计算隶属度
    ):
        super().__init__(
            pool_classifiers=pool_classifiers,
            k=k,
            DFP=DFP,
            with_IH=with_IH,
            safe_k=safe_k,
            IH_rate=IH_rate,
            random_state=random_state,
            knn_classifier=knn_classifier,
            DSEL_perc=0.5
        )
        self.strategy_selection_method = strategy_selection_method
        self.multiCore_process = multiCore_process
        self.dataset_features = dataset_features if dataset_features is not None else {}
        self.selected_strategy = None
        self.selected_params = None
        self.available_strategies = {
            "DESFHMW_allboxes_vector_he": DESFHMW_allboxes_vector_he,
            "FHDES_Allboxes_vector_he": FHDES_Allboxes_vector_he,
            "DESFHMW_JFB_vector_he": DESFHMW_JFB_vector_he,
            "FHDES_JFB_vector_he": FHDES_JFB_vector_he,
            "DESFHMW_prior_vector_he": DESFHMW_prior_vector_he,
            "FHDES_prior_vector_he": FHDES_prior_vector_he
        }
        self.strategy_model = None
        self.feature_cols = [
            "imbalance_ratio", "data_density", "noise", "feature_correlation",
            "data_dispersion", "mutual_information", "outlier_ratio", "avg_hard_sample_ratio"
        ]
        self.rules_file_path = rules_file_path  # 存储规则文件路径
        self.reg_lambda = reg_lambda  # 正则化参数
        self.gamma = gamma  # 高斯函数参数

    def predict_params_directly(self, features_vector):
        """
        基于从给定的json文件中提取规则，然后根据规则使用if...else...逻辑得到对应的参数选择
        
        参数:
        features_vector: 特征向量，按feature_cols顺序排列
        
        返回:
        dict: 包含所有预测参数的字典
        """
        try:
            with open(self.rules_file_path, 'r', encoding='utf-8') as f:
                rules = json.load(f)
        except Exception as e:
            print(f"读取规则文件出错: {e}")
            return {}

        # 创建特征字典以便于访问
        feature_dict = dict(zip(self.feature_cols, features_vector))

        # 初始化参数
        params = {}

        # 遍历规则，为每个参数动态解析树分裂阈值
        for param_name, param_rules in rules.items():
            text = param_rules.get('text', '')
            
            # 处理分类参数 (doContraction, mis_sample_based, shuffle_dataOrder, thetacheck)
            if param_name in ['doContraction', 'mis_sample_based', 'shuffle_dataOrder', 'thetacheck']:
                classes = param_rules.get('classes', ['False', 'True'])
                # 动态解析树结构并确定类别
                class_index = self._parse_tree_structure(text, feature_dict, is_classification=True)
                # 确保索引在有效范围内
                class_index = max(0, min(class_index, len(classes) - 1))
                params[param_name] = classes[class_index]
            
            # 处理回归参数 (mu_min, mu_max, theta_min, theta_max)
            elif param_name in ['mu_min', 'mu_max', 'theta_min', 'theta_max']:
                # 动态解析树结构并确定值
                value = self._parse_tree_structure(text, feature_dict, is_classification=False)
                if value is not None:
                    params[param_name] = value

        # 确保mu_min <= mu_max
        if 'mu_min' in params and 'mu_max' in params and params['mu_min'] > params['mu_max']:
            params['mu_min'], params['mu_max'] = params['mu_max'], params['mu_min']

        # 确保theta_min <= theta_max
        if 'theta_min' in params and 'theta_max' in params and params['theta_min'] > params['theta_max']:
            params['theta_min'], params['theta_max'] = params['theta_max'], params['theta_min']

        return {
            'doContraction': params.get('doContraction', False) if isinstance(params.get('doContraction', 'False'), str) else params.get('doContraction', False),
            'mis_sample_based': params.get('mis_sample_based', False) if isinstance(params.get('mis_sample_based', 'False'), str) else params.get('mis_sample_based', False),
            'shuffle_dataOrder': params.get('shuffle_dataOrder', False) if isinstance(params.get('shuffle_dataOrder', 'False'), str) else params.get('shuffle_dataOrder', False),
            'thetacheck': params.get('thetacheck', False) if isinstance(params.get('thetacheck', 'False'), str) else params.get('thetacheck', False),
            'mu_range': [params.get('mu_min'), params.get('mu_max')],
            'theta_range': [params.get('theta_min'), params.get('theta_max')],
            'reg_lambda': self.reg_lambda,  # 添加超椭球体特有参数
            'gamma': self.gamma  # 添加超椭球体特有参数
        }
        
    def _parse_tree_structure(self, tree_text, feature_dict, is_classification=True):
        """
        解析决策树结构文本，根据特征值和树分裂阈值确定最终的类别或回归值
        
        参数:
        tree_text: 决策树结构文本
        feature_dict: 特征值字典
        is_classification: 是否为分类任务
        
        返回:
        int or float: 分类任务返回类别索引，回归任务返回预测值
        """
        # 按行分割树文本
        lines = tree_text.strip().split('\n')
        
        # 递归解析决策树
        return self._parse_tree_node(lines, 0, feature_dict, is_classification)
        
    def _parse_tree_node(self, lines, line_index, feature_dict, is_classification=True):
        """
        递归解析决策树节点
        
        参数:
        lines: 所有行的列表
        line_index: 当前行索引
        feature_dict: 特征值字典
        is_classification: 是否为分类任务
        
        返回:
        int or float: 分类任务返回类别索引，回归任务返回预测值
        """
        if line_index >= len(lines):
            return 0 if is_classification else 0.5
        
        line = lines[line_index].strip()
        
        # 检查是否是叶子节点（分类任务）
        if is_classification and 'class:' in line:
            try:
                class_index = int(line.split('class:')[1].strip())
                return class_index
            except (ValueError, IndexError):
                return 0
        
        # 检查是否是叶子节点（回归任务）
        if not is_classification and 'value:' in line:
            try:
                value_str = line.split('value:')[1].strip()
                if value_str.startswith('[') and value_str.endswith(']'):
                    value = float(value_str[1:-1].split(',')[0].strip())
                    return value
            except (ValueError, IndexError):
                return 0.5
        
        # 尝试解析内部节点
        try:
            # 提取缩进级别
            indent = len(line) - len(line.lstrip('|   '))
            
            # 提取特征名和阈值
            # 格式例如: "|--- feature <= 0.1234" 或 "|--- feature >  0.1234"
            if '<=' in line:
                parts = line.split('<=')
                condition_part = parts[0].strip('| -')
                threshold = float(parts[1].strip())
                operator = '<='
            elif '>' in line:
                parts = line.split('>')
                condition_part = parts[0].strip('| -')
                threshold = float(parts[1].strip())
                operator = '>'
            else:
                # 如果无法解析条件，使用下一个叶子节点的值
                return self._parse_tree_node(lines, line_index + 1, feature_dict, is_classification)
            
            # 获取特征值
            feature_name = condition_part.strip()
            feature_value = feature_dict.get(feature_name, 0)
            
            # 根据特征值和阈值决定走哪个分支
            # 找到左右子节点的起始行
            left_child_index = line_index + 1
            right_child_index = -1
            
            # 查找右子节点
            for i in range(line_index + 1, len(lines)):
                current_indent = len(lines[i]) - len(lines[i].lstrip('|   '))
                if current_indent == indent and i > left_child_index:
                    right_child_index = i
                    break
            
            # 根据条件选择分支
            if (operator == '<=' and feature_value <= threshold) or \
               (operator == '>' and feature_value > threshold):
                # 左分支
                if right_child_index == -1:
                    return self._parse_tree_node(lines, left_child_index, feature_dict, is_classification)
                else:
                    # 找到左分支的结束位置
                    left_end_index = right_child_index
                    while left_end_index > left_child_index:
                        current_indent = len(lines[left_end_index - 1]) - len(lines[left_end_index - 1].lstrip('|   '))
                        if current_indent == indent + 4:
                            left_end_index -= 1
                        else:
                            break
                    # 处理左分支
                    return self._parse_tree_node(lines, left_child_index, feature_dict, is_classification)
            else:
                # 右分支
                if right_child_index != -1:
                    return self._parse_tree_node(lines, right_child_index, feature_dict, is_classification)
                else:
                    # 如果没有右分支，使用默认值
                    return 0 if is_classification else 0.5
            
        except Exception as e:
            # 如果解析出错，使用简化方式：找到最后一个叶子节点
            if is_classification:
                class_lines = [line for line in lines if 'class:' in line]
                if class_lines:
                    try:
                        return int(class_lines[-1].split('class:')[1].strip())
                    except:
                        return 0
                return 0
            else:
                value_lines = [line for line in lines if 'value:' in line]
                if value_lines:
                    try:
                        value_str = value_lines[-1].split('value:')[1].strip()
                        if value_str.startswith('[') and value_str.endswith(']'):
                            return float(value_str[1:-1].split(',')[0].strip())
                    except:
                        return 0.5
                return 0.5
    
    def select_optimal_strategy(self):
        """
        根据数据集特征选择最优的超椭球体策略
        使用直接的if...else逻辑预测参数，不再通过解析规则文本
        """
        # 添加调试信息，显示当前的数据集特征
        print("\n===== 数据集特征分析 ======")
        for key, value in self.dataset_features.items():
            print(f"{key}: {value}")
        print("=========================")

        # 提取特征向量
        features_vector = self.extract_features_vector()
        print(f"特征向量: {features_vector}")
        
        # 直接使用if...else逻辑预测参数
        params = self.predict_params_directly(features_vector)

        # 从params字典中获取参数值
        doContraction = params['doContraction']
        mis_sample_based = params['mis_sample_based']
        mu_range = params['mu_range']
        shuffle_dataOrder = params['shuffle_dataOrder']
        theta_range = params['theta_range']
        thetacheck = params['thetacheck']
        reg_lambda = params['reg_lambda']  # 超椭球体特有参数
        gamma = params['gamma']  # 超椭球体特有参数

        # 打印预测的参数值
        print(f"预测参数：")
        print(f"  doContraction: {doContraction}")
        print(f"  mis_sample_based: {mis_sample_based}")
        print(f"  mu_range: {mu_range}")
        print(f"  shuffle_dataOrder: {shuffle_dataOrder}")
        print(f"  theta_range: {theta_range}")
        print(f"  thetacheck: {thetacheck}")
        print(f"  reg_lambda: {reg_lambda}")
        print(f"  gamma: {gamma}")

        # 存储这些参数到类属性，供后续使用
        self.doContraction = doContraction
        self.mis_sample_based = mis_sample_based
        self.mu_range = mu_range
        self.shuffle_dataOrder = shuffle_dataOrder
        self.theta_range = theta_range
        self.thetacheck = thetacheck
        self.reg_lambda = reg_lambda
        self.gamma = gamma

        # 根据指令和数据特征选择策略
        features = self.dataset_features
        
        # 检查是否为高维稀疏数据（假设特征数量大于30且稀疏性高）
        if 'sparsity' in features and features['sparsity'] > 0.8:
            self.selected_strategy = 'DESFHMW_prior_vector_he'
            print("检测到高维稀疏数据，选择prior变体超椭球体策略")
        # 检查是否为低资源环境（假设系统资源不足）
        elif 'low_resource' in features and features['low_resource']:
            self.selected_strategy = 'FHDES_prior_vector_he'
            print("检测到低资源环境，选择fh_des-prior超椭球体策略")
        # 检查是否为噪声数据集（假设噪声水平高）
        elif 'noise' in features and features['noise'] > 0.2:
            if doContraction and mis_sample_based:
                self.selected_strategy = 'DESFHMW_allboxes_vector_he'
            elif doContraction:
                self.selected_strategy = 'DESFHMW_allboxes_vector_he'
            elif mis_sample_based:
                self.selected_strategy = 'DESFHMW_JFB_vector_he'
            else:
                self.selected_strategy = 'DESFHMW_JFB_vector_he'
            print("检测到噪声数据集，优先选择des_FHMW_*超椭球体系列策略")
        # 检查是否追求效率（假设需要实时处理）
        elif 'real_time' in features and features['real_time']:
            self.selected_strategy = 'FHDES_JFB_vector_he'
            print("检测到追求效率场景，选择FHDES-JFB超椭球体策略")
        else:
            # 默认策略
            self.selected_strategy = 'FHDES_JFB_vector_he'
            print("使用默认超椭球体策略: FHDES_JFB_vector_he")

        print(f"基于参数选择策略: {self.selected_strategy}")

        # 确保selected_strategy在available_strategies中
        if self.selected_strategy not in self.available_strategies:
            print(f"警告: 选择的策略 '{self.selected_strategy}' 不在可用策略列表中，使用默认策略 'FHDES_JFB_vector_he'")
            self.selected_strategy = 'FHDES_JFB_vector_he'

        print(f"选定的策略: {self.selected_strategy}")
        print(
            f"选择的参数: doContraction={doContraction}, mis_sample_based={mis_sample_based}, "
            f"mu_range={mu_range}, shuffle_dataOrder={shuffle_dataOrder}, "
            f"theta_range={theta_range}, thetacheck={thetacheck}, "
            f"reg_lambda={reg_lambda}, gamma={gamma}")

        return self.selected_strategy

    def extract_features_vector(self):
        """
        从dataset_features中提取特征向量，用于决策树规则预测
        """
        features_vector = []
        for feature_name in self.feature_cols:
            # 根据特征名称从dataset_features中提取相应的值
            # 这里需要根据实际的数据特征结构进行调整
            value = self.dataset_features.get(feature_name, 0)
            features_vector.append(value)
        return features_vector

    def fit(
        self, X, y, X_train, y_train, X_test=None, y_test=None,
        use_optimization=True
    ):
        super().fit(X, y)
        self.select_optimal_strategy()
        strategy_cls = self.available_strategies[self.selected_strategy]
        params = {
            'pool_classifiers': self.pool_classifiers,
            'random_state': self.random_state,
            'reg_lambda': self.reg_lambda,  # 添加超椭球体特有参数
            'gamma': self.gamma  # 添加超椭球体特有参数
        }

        # 如果启用优化且提供了测试集，则进行参数优化
        if use_optimization and X_test is not None and y_test is not None:
            print(f"准备为超椭球体策略 {self.selected_strategy} 进行参数优化...")
            best_params = self.optimize_params_with_optuna(
                X_train, y_train, X_test, y_test, self.selected_strategy
            )
            params.update(best_params)
            print(f"使用优化后的参数: {params}")
            self.selected_params = params.copy()

        self.strategy_model = strategy_cls(**params)
        self.strategy_model.fit(X, y)
        print(f"[调试] 已初始化超椭球体策略模型: {type(self.strategy_model).__name__}, 策略: {self.selected_strategy}")
        return self

    def predict(self, X):
        if self.strategy_model is not None:
            return self.strategy_model.predict(X)
        else:
            raise RuntimeError('未选择有效超椭球体策略模型')

    def predict_proba(self, X):
        if self.strategy_model is not None:
            return self.strategy_model.predict_proba(X)
        else:
            raise RuntimeError('未选择有效超椭球体策略模型')

    def get_strategy_info(self):
        strategy_name = self.selected_strategy if self.selected_strategy else "unknown"
        params = self.selected_params if self.selected_params else {}
        return {
            'strategy_name': strategy_name,
            'params': params,
            'dataset_features': self.dataset_features
        }

    def optimize_params_with_optuna(self, X_train, y_train, X_val, y_val, strategy_name):
        """
        使用optuna库为选定的超椭球体策略寻找最优的theta、mu、reg_lambda和gamma参数
        使用预划分的训练集和验证集进行评估

        参数:
            X_train: 训练集特征数据
            y_train: 训练集标签数据
            X_val: 验证集特征数据
            y_val: 验证集标签数据
            strategy_name: 要进行参数搜索的策略名称
        """
        # 获取策略类
        strategy_cls = self.available_strategies.get(strategy_name, None)
        if strategy_cls is None:
            print(f"策略 {strategy_name} 不存在，无法进行参数优化")
            return {}

        print(f"为超椭球体策略 {strategy_name} 进行参数优化...")
        start_time = time.time()

        def objective(trial):
            # 从类属性中获取theta和mu的搜索空间
            theta_min, theta_max = self.theta_range
            mu_min, mu_max = self.mu_range
            
            # 定义参数搜索空间
            theta = trial.suggest_float('theta', theta_min, theta_max, step=0.001)
            mu = trial.suggest_float('mu', mu_min, mu_max, step=0.1)
            reg_lambda = trial.suggest_float('reg_lambda', 1e-6, 1e-2, log=True)  # 超椭球体特有参数
            gamma = trial.suggest_float('gamma', 0.1, 10.0, log=True)  # 超椭球体特有参数
            
            current_params = {
                'theta': theta,
                'mu': mu,
                'reg_lambda': reg_lambda,
                'gamma': gamma,
                'pool_classifiers': self.pool_classifiers,
                'random_state': self.random_state
            }

            try:
                # 创建模型并训练
                model = strategy_cls(**current_params)
                model.fit(X_train, y_train)

                # 预测概率并计算AUC
                y_pred_proba = model.predict_proba(X_val)
                
                # 处理二分类和多分类情况
                if y_pred_proba.shape[1] == 2:  # 二分类
                    auc = roc_auc_score(y_val, y_pred_proba[:, 1])
                else:  # 多分类
                    auc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
                
                return auc
            except Exception as e:
                print(f"参数优化过程中出错: {e}")
                return 0.0

        # 创建optuna研究对象
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)  # 可以根据需要调整试验次数

        # 获取最佳参数
        best_params = study.best_params
        print(f"参数优化完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"最佳参数: {best_params}")
        print(f"最佳AUC: {study.best_value:.4f}")

        return best_params