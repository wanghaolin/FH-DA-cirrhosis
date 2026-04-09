# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class FHDES_JFB_vector_he(BaseDES):

    def __init__(self, pool_classifiers=None,
                 k=7, DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta=0.05,
                 mu=0.991,
                 gamma=1.0,  # 控制高斯函数宽度的参数
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False,
                 reg_lambda=1e-6):  # 正则化参数，防止协方差矩阵奇异
        self.theta = theta
        self.mu = mu
        self.gamma = gamma
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder
        self.reg_lambda = reg_lambda  # 新增：正则化参数

        super(FHDES_JFB_vector_he, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)

    def add_ellipsoids(self, hboxC, hboxCov, bC, bCov):
        """添加新的超椭球体"""
        hboxC = np.concatenate((hboxC, bC))
        # 扩展协方差矩阵列表
        hboxCov.extend(bCov)
        return hboxC, hboxCov

    def compute_mahalanobis_distance(self, x, center, cov_matrix):
        """计算马氏距离：sqrt((X-μ)^T * Σ^(-1) * (X-μ))
        
        增强版：添加额外的数值稳定性检查，确保计算安全可靠
        """
        delta = x - center
        
        # 添加更强大的正则化以确保矩阵可逆和数值稳定性
        cov_matrix_reg = cov_matrix + self.reg_lambda * np.eye(cov_matrix.shape[0])
        
        try:
            # 确保协方差矩阵是对称的
            cov_matrix_reg = (cov_matrix_reg + cov_matrix_reg.T) / 2
            
            # 计算逆矩阵
            inv_cov = np.linalg.inv(cov_matrix_reg)
            
            # 计算马氏距离的平方
            mahalanobis_squared = np.dot(np.dot(delta, inv_cov), delta)
            
            # 检查数值稳定性：确保值是非负的，处理可能的浮点误差
            mahalanobis_squared = max(0.0, mahalanobis_squared)
            
            # 检查是否为NaN或无穷大
            if np.isnan(mahalanobis_squared) or np.isinf(mahalanobis_squared):
                # 如果出现无效值，使用欧几里得距离作为备选
                return np.linalg.norm(delta)
            
            # 计算最终的马氏距离
            mahalanobis_dist = np.sqrt(mahalanobis_squared)
            return mahalanobis_dist
        except (np.linalg.LinAlgError, ValueError, TypeError) as e:
            # 捕获更多可能的异常类型
            # 如果矩阵仍然不可逆或出现其他错误，使用欧几里得距离作为备选
            return np.linalg.norm(delta)

    def is_inside(self, hboxC, hboxCov, ellipsoidInd, x):
        """检查点是否在超椭球体内
           使用马氏距离，阈值为1.0表示在边界内"""
        mahalanobis_dist = self.compute_mahalanobis_distance(x, hboxC[ellipsoidInd], hboxCov[ellipsoidInd])
        return mahalanobis_dist <= 1.0

    def is_expandable(self, hboxC, hboxCov, ellipsoidInd, x):
        """判断是否可以扩展超椭球体以包含新点"""
        mahalanobis_dist = self.compute_mahalanobis_distance(x, hboxC[ellipsoidInd], hboxCov[ellipsoidInd])
        # 检查扩展后的马氏距离是否小于theta
        return mahalanobis_dist < self.theta

    def expand_ellipsoid(self, hboxC, hboxCov, ellipsoidInd, x):
        """扩展超椭球体以包含新点，更新协方差矩阵"""
        # 获取当前超椭球体的中心和协方差矩阵
        current_center = hboxC[ellipsoidInd]
        current_cov = hboxCov[ellipsoidInd]
        
        # 计算新的中心（保持不变）
        # 计算新的协方差矩阵，考虑新点
        delta = x - current_center
        # 使用增量更新协方差矩阵的简化方法
        # 这是一个近似方法，更精确的方法需要重新计算所有点的协方差
        new_cov = current_cov + np.outer(delta, delta) * 0.1  # 小步长更新
        
        # 确保协方差矩阵是对称的
        new_cov = (new_cov + new_cov.T) / 2
        
        # 更新协方差矩阵
        hboxCov[ellipsoidInd] = new_cov

    def membership_boxes(self, hboxC, hboxCov, Xq):
        """使用高斯函数计算隶属度，基于马氏距离"""
        NO_hypeboxes = len(hboxC)
        # n_features = hboxC.shape[1]
        n_queries = Xq.shape[1]
        
        # 初始化隶属度矩阵
        m = np.zeros((NO_hypeboxes, n_queries))
        
        # 对每个超椭球体和每个查询点计算隶属度
        for i in range(NO_hypeboxes):
            center = hboxC[i]
            cov_matrix = hboxCov[i]
            
            for j in range(n_queries):
                xq = Xq[0, j]  # Xq形状: (1, n_queries, n_features)
                mahalanobis_dist = self.compute_mahalanobis_distance(xq, center, cov_matrix)
                
                # 使用高斯函数计算隶属度
                # 马氏距离越小，隶属度越高
                m[i, j] = np.exp(-self.gamma * mahalanobis_dist ** 2)
        
        # 确保隶属度在[0,1]范围内
        m = np.clip(m, 0, 1)
        
        return m

    def will_exceed_samples(self, hboxC, hboxCov, ellipsoidInd, x, con_samples):
        """判断扩展超椭球体是否会包含矛盾样本"""
        # 扩展超椭球体
        temp_center = hboxC[ellipsoidInd].copy()
        temp_cov = hboxCov[ellipsoidInd].copy()
        
        # 临时扩展超椭球体
        delta = x - temp_center
        temp_cov = temp_cov + np.outer(delta, delta) * 0.1
        temp_cov = (temp_cov + temp_cov.T) / 2
        
        # 检查每个矛盾样本是否在扩展后的椭球体内
        for sample in con_samples:
            mahalanobis_dist = self.compute_mahalanobis_distance(sample, temp_center, temp_cov)
            if mahalanobis_dist <= 1.0:
                return True
        return False

    def contract_samplesBased(self, hboxC, hboxCov, ellipsoidInd, con_samples):
        """基于矛盾样本收缩超椭球体"""
        current_center = hboxC[ellipsoidInd]
        current_cov = hboxCov[ellipsoidInd]
        
        # 找出所有在当前椭球体内的矛盾样本
        conflict_distances = []
        for sample in con_samples:
            mahalanobis_dist = self.compute_mahalanobis_distance(sample, current_center, current_cov)
            if mahalanobis_dist <= 1.0:
                conflict_distances.append(mahalanobis_dist)
        
        # 如果有矛盾样本，收缩椭球体
        if conflict_distances:
            # 计算收缩因子，使最小距离大于1.0
            min_conflict_dist = min(conflict_distances)
            if min_conflict_dist < 1.0:
                # 缩放协方差矩阵来收缩椭球体
                scale_factor = 0.9 * (1.0 / min_conflict_dist)  # 使最小距离变为0.9
                hboxCov[ellipsoidInd] = current_cov / (scale_factor ** 2)

    def fit(self, X, y):
        super(FHDES_JFB_vector_he, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bC, bCov] = self.setup_hyperellipsoids(classifier_index)
                # 使用中心和协方差矩阵代替半径
                class_dic = {"clsr": classifier_index, "Center": bC, "Covariance": bCov}
                self.NO_hypeboxes += len(bC)
                self.HBoxes.append(class_dic)
        else:
            # 修改进程数计算方式：不超过CPU核心数且最多4个进程
            no_processes = min(multiprocessing.cpu_count(), 4)
            with multiprocessing.Pool(processes=no_processes) as pool:
                try:
                    list_results = pool.map(self.setup_hyperellipsoids, range(self.n_classifiers_))
                except OSError as e:
                    if "1450" in str(e):
                        # 回退到单进程模式
                        no_processes = 1
                        with multiprocessing.Pool(processes=no_processes) as pool:
                            list_results = pool.map(self.setup_hyperellipsoids, range(self.n_classifiers_))
                for clsr_box in list_results:
                    class_dic = {"clsr": 0, "Center": clsr_box[0], "Covariance": clsr_box[1]}
                    self.NO_hypeboxes += len(clsr_box[0])
                    self.HBoxes.append(class_dic)

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        
        if self.mis_sample_based:
            highest_mems = np.ones([len(query), self.n_classifiers_])
        else:
            highest_mems = np.zeros([len(query), self.n_classifiers_])

        Xq = query.reshape(1, len(query), self.n_features_)

        for clsr in range(len(self.HBoxes)):
            # 使用中心和协方差矩阵
            hboxC = self.HBoxes[clsr]["Center"]
            hboxCov = self.HBoxes[clsr]["Covariance"]

            clsrBoxes_m = self.membership_boxes(hboxC, hboxCov, Xq)
            if len(hboxC) > 1:
                # 选择隶属度最高的两个超椭球体
                bb_indexes = np.argsort(-clsrBoxes_m, axis=0)
                b1 = bb_indexes[0, :]
                b2 = bb_indexes[1, :]
                for i in range(0, len(query)):
                    # 添加索引边界检查
                    if i < clsrBoxes_m.shape[1]:  # 确保i在clsrBoxes_m的第二维范围内
                        if b1[i] < clsrBoxes_m.shape[0] and b2[i] < clsrBoxes_m.shape[0]:  # 确保b1[i]和b2[i]在clsrBoxes_m的第一维范围内
                            if clsrBoxes_m[b1[i], i] == 1:  # 如果查询样本位于或接近椭球体内
                                highest_mems[i, int(clsr)] = 1
                            else:
                                highest_mems[i, int(clsr)] = clsrBoxes_m[b1[i], i] * 0.7 + clsrBoxes_m[b2[i], i] * 0.3
                        else:
                            # 如果索引超出范围，使用默认值
                            highest_mems[i, int(clsr)] = 0.5  # 使用中间值
                    else:
                        # 如果查询索引超出范围，使用默认值
                        highest_mems[i, int(clsr)] = 0.5  # 使用中间值

            else:  # 如果分类器只有一个超椭球体
                for i in range(0, len(query)):
                    highest_mems[i, int(clsr)] = clsrBoxes_m[0, i]

        # 计算竞争力
        if self.mis_sample_based:
            competences_ = np.max(highest_mems) - highest_mems
        else:
            competences_ = highest_mems

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_

    def setup_hyperellipsoids(self, classifier):
        """为每个分类器创建超椭球体"""
        if np.size(classifier) < 0:
            pass

        if self.mis_sample_based:
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]

        # 初始化超椭球体中心和协方差矩阵
        # 中心初始化为-1，表示尚未创建
        hboxC = np.zeros((1, self.n_features_)) - 1
        # 协方差矩阵初始化为空列表
        hboxCov = []

        selected_samples = self.DSEL_data_[samples_ind, :]
        contraction_samples = self.DSEL_data_[Contraction_ind, :]
        
        # 打乱数据顺序
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples, random_state=classifier)

        for ind, X in enumerate(selected_samples):
            # 创建第一个超椭球体
            if hboxC[0, 0] == -1:
                hboxC[0, :] = X  # 中心设置为第一个样本
                # 初始协方差矩阵：小的单位矩阵，确保可逆
                initial_cov = self.reg_lambda * np.eye(self.n_features_)
                hboxCov.append(initial_cov)
                continue

            # 检查点是否在现有超椭球体内
            is_inEllipsoid = False
            for ellipsoidInd in range(len(hboxC)):
                if self.is_inside(hboxC, hboxCov, ellipsoidInd, X):
                    is_inEllipsoid = True
                    break
            if is_inEllipsoid:
                continue

            ######################## 扩展超椭球体 ############################
            # 计算到每个超椭球体中心的欧几里得距离并排序（作为近似）
            distances = np.linalg.norm(X - hboxC, axis=1)
            nearestEllipsoid_ind = np.argmin(distances)
            
            if self.thetaCheck and self.doContraction:
                # 检查是否可以扩展且扩展后收缩不会包含矛盾样本
                if self.is_expandable(hboxC, hboxCov, nearestEllipsoid_ind, X):
                    # 扩展超椭球体
                    self.expand_ellipsoid(hboxC, hboxCov, nearestEllipsoid_ind, X)
                    # 基于矛盾样本收缩超椭球体
                    self.contract_samplesBased(hboxC, hboxCov, nearestEllipsoid_ind, contraction_samples)
                    continue

            elif self.thetaCheck and not self.doContraction:
                # 仅检查是否可以扩展，不收缩
                if self.is_expandable(hboxC, hboxCov, nearestEllipsoid_ind, X):
                    self.expand_ellipsoid(hboxC, hboxCov, nearestEllipsoid_ind, X)
                    continue

            elif not self.thetaCheck and self.doContraction:
                # 不检查theta，但确保不包含矛盾样本
                if not self.will_exceed_samples(hboxC, hboxCov, nearestEllipsoid_ind, X, contraction_samples):
                    self.expand_ellipsoid(hboxC, hboxCov, nearestEllipsoid_ind, X)
                    continue
            
            ######################## 创建新的超椭球体 ############################
            new_center = X.reshape(1, self.n_features_)
            # 新超椭球体初始协方差矩阵
            new_cov = self.reg_lambda * np.eye(self.n_features_)
            hboxC, hboxCov = self.add_ellipsoids(hboxC, hboxCov, bC=new_center, bCov=[new_cov])

        return hboxC, hboxCov

    def select(self, competences):
        
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers