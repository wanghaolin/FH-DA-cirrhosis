# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class FHDES_prior_vector_he(BaseDES):

    def __init__(self, pool_classifiers=None,
                 k=7, DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 knn_classifier='knn',
                 DSEL_perc=0.5,
                 HyperEllipsoids=[],
                 theta=0.05,
                 mu=0.991,
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False,
                 gamma=1.0,
                 reg_lambda=1e-6):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HEllipsoids = []
        self.NO_ellipsoids = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder
        self.gamma = gamma  # 控制高斯函数的宽度
        self.reg_lambda = reg_lambda  # 协方差矩阵正则化参数

        ############### it should be based on Clustering #############################
        super(FHDES_prior_vector_he, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)

    def add_ellipsoids(self, centers, cov_matrices, new_center, new_cov_matrix):
        centers = np.concatenate((centers, new_center.reshape(1, -1)))
        cov_matrices = np.concatenate((cov_matrices, new_cov_matrix.reshape(1, self.n_features_, self.n_features_)))
        return centers, cov_matrices

    def compute_mahalanobis_distance(self, X, center, cov_matrix):
        # 确保协方差矩阵可逆
        cov_matrix += self.reg_lambda * np.eye(self.n_features_)
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # 如果矩阵不可逆，使用伪逆
            inv_cov = np.linalg.pinv(cov_matrix)
        
        diff = X - center
        # 计算马氏距离：(X-μ)^T Σ^{-1} (X-μ)
        distance = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        return distance

    def is_inside(self, centers, cov_matrices, ellipsoid_ind, x):
        # 使用马氏距离判断样本是否在椭球体内
        distance = self.compute_mahalanobis_distance(x.reshape(1, -1), centers[ellipsoid_ind], cov_matrices[ellipsoid_ind])
        return distance <= 1.0  # 假设阈值为1.0，表示在椭球体内

    def membership_ellipsoids(self, centers, cov_matrices, Xq):
        NO_ellipsoids, n_features = centers.shape
        memberships = np.zeros((NO_ellipsoids, Xq.shape[1]))
        
        for i in range(NO_ellipsoids):
            # 计算每个查询样本到椭球体中心的马氏距离
            for j in range(Xq.shape[1]):
                distance = self.compute_mahalanobis_distance(Xq[0, j, :].reshape(1, -1), centers[i], cov_matrices[i])
                # 使用高斯函数计算隶属度
                memberships[i, j] = np.exp(-self.gamma * distance**2)
                # 限制隶属度在0-1之间
                memberships[i, j] = max(0, min(1, memberships[i, j]))
        
        return memberships

    def update_ellipsoid(self, centers, cov_matrices, ellipsoid_ind, x):
        # 更新椭球体的中心和协方差矩阵
        # 获取当前椭球体内的所有样本（这里简化处理）
        current_center = centers[ellipsoid_ind]
        current_cov = cov_matrices[ellipsoid_ind]
        
        # 计算新的中心（简单平均）
        new_center = (current_center + x) / 2
        
        # 计算新的协方差矩阵
        # 这里简化处理，实际应用中可能需要更复杂的更新策略
        diff_center = x - current_center
        new_cov = (current_cov + np.outer(diff_center, diff_center)) / 2
        
        return new_center, new_cov

    def will_exceed_samples(self, centers, cov_matrices, ellipsoid_ind, x, con_samples):
        # 检查扩展椭球体是否会包含冲突样本
        # 简化实现，实际应用中可能需要更复杂的判断
        if len(con_samples) == 0:
            return False
        
        # 尝试更新椭球体
        temp_center, temp_cov = self.update_ellipsoid(centers, cov_matrices, ellipsoid_ind, x)
        
        # 检查冲突样本是否在新的椭球体内
        for con_sample in con_samples:
            distance = self.compute_mahalanobis_distance(con_sample.reshape(1, -1), temp_center, temp_cov)
            if distance <= 1.0:
                return True
        
        return False

    def contract_samplesBased(self, centers, cov_matrices, ellipsoid_ind, con_samples):
        # 基于冲突样本收缩椭球体
        # 简化实现，实际应用中可能需要更复杂的收缩策略
        pass

    def fit(self, X, y):
        super(FHDES_prior_vector_he, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [centers, cov_matrices] = self.setup_hyperellipsoids(classifier_index)
                class_dic =  { "clsr" : classifier_index, "centers" : centers, "cov_matrices" : cov_matrices }
                self.NO_ellipsoids += len(centers)
                self.HEllipsoids.append(class_dic)
        else:
            # (改)修改进程数计算方式：不超过CPU核心数且最多4个进程
            no_processes = min(multiprocessing.cpu_count(), 4)
            with multiprocessing.Pool(processes=no_processes) as pool:
                try:
                    list = pool.map(self.setup_hyperellipsoids, range(self.n_classifiers_))
                except OSError as e:
                    if "1450" in str(e):
                        # 回退到单进程模式
                        no_processes = 1
                        with multiprocessing.Pool(processes=no_processes) as pool:
                            list = pool.map(self.setup_hyperellipsoids, range(self.n_classifiers_))
                for clsr_ellipsoid in list:
                    class_dic = {"clsr": 0, "centers": clsr_ellipsoid[0], "cov_matrices": clsr_ellipsoid[1]}
                    self.NO_ellipsoids += len(clsr_ellipsoid[0])
                    self.HEllipsoids.append(class_dic)

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):

        if self.mis_sample_based:
            highest_mems = np.ones([len(query), self.n_classifiers_])
        else:
            highest_mems = np.zeros([len(query), self.n_classifiers_])

        Xq = query.reshape(1, len(query), self.n_features_)

        for clsr in range(len(self.HEllipsoids)):
            centers = self.HEllipsoids[clsr]["centers"]
            cov_matrices = self.HEllipsoids[clsr]["cov_matrices"]

            clsrEllipsoids_m = self.membership_ellipsoids(centers, cov_matrices, Xq)
            if len(centers) > 1:
                bb_indexes = np.argsort(-clsrEllipsoids_m, axis=0)
                b1 = bb_indexes[0,:]
                b2 = bb_indexes[1,:]
                for i in range(0, len(query)):
                    if clsrEllipsoids_m[b1[i],i] == 1: # if the query sample is located inside or near to the ellipsoid
                        highest_mems[i, int(clsr)] = 1
                    else:
                        highest_mems[i, int(clsr)] = clsrEllipsoids_m[b1[i],i] * 0.7 + clsrEllipsoids_m[b2[i],i] * 0.3

            else:  # In case that we have only one hyperellipsoid for the classifier
                for i in range(0, len(query)):
                    highest_mems[i, int(clsr)] = clsrEllipsoids_m[0, i]

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(highest_mems) - highest_mems
        else:
            competences_ = highest_mems

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_

    def setup_hyperellipsoids(self, classifier):
        if np.size(classifier) < 0:
            pass

        if(self.mis_sample_based):
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]

        # 初始化椭球体中心和协方差矩阵
        centers = np.zeros((0, self.n_features_))
        cov_matrices = np.zeros((0, self.n_features_, self.n_features_))

        selected_samples = self.DSEL_data_[samples_ind, :]
        contraction_samples = self.DSEL_data_[Contraction_ind, :]
        
        # 为收缩样本添加一些扰动以避免协方差矩阵奇异
        if len(contraction_samples) > 0:
            contraction_samples += np.random.normal(0, self.reg_lambda, contraction_samples.shape)

        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples, random_state = classifier)

        for ind, X in enumerate(selected_samples):
            # 第一个椭球体的创建
            if len(centers) == 0:
                centers = np.array([X])
                # 初始协方差矩阵设为单位矩阵乘以一个小系数
                initial_cov = np.eye(self.n_features_) * self.theta
                cov_matrices = np.array([initial_cov])
                continue

            # X是否在某个椭球体内？
            is_inEllipsoid = False
            for ellipsoidInd in range(len(centers)):
                if self.is_inside(centers, cov_matrices, ellipsoidInd, X):
                    is_inEllipsoid = True
                    break
            if is_inEllipsoid:
                # 不需要操作
                continue

            ######################## 扩展 ############################
            # 按距离排序椭球体
            distances = np.array([self.compute_mahalanobis_distance(X.reshape(1, -1), centers[i], cov_matrices[i])[0] for i in range(len(centers))])
            sorted_indexes = np.argsort(distances)[::-1]  # 从最远到最近排序
            expanded = False
            
            for ind_ellipsoid in sorted_indexes:
                # 检查扩展后是否会包含冲突样本
                if not self.will_exceed_samples(centers, cov_matrices, ind_ellipsoid, X, contraction_samples):
                    # 更新椭球体
                    new_center, new_cov = self.update_ellipsoid(centers, cov_matrices, ind_ellipsoid, X)
                    centers[ind_ellipsoid] = new_center
                    cov_matrices[ind_ellipsoid] = new_cov
                    expanded = True
                    break

            ######################## 创建新椭球体 ############################
            if expanded == False:
                # 创建新的椭球体
                new_center = X
                # 初始协方差矩阵设为单位矩阵乘以一个小系数
                new_cov = np.eye(self.n_features_) * self.theta
                centers, cov_matrices = self.add_ellipsoids(centers, cov_matrices, new_center, new_cov)

        return centers, cov_matrices

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers