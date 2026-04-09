# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class DESFHMW_JFB_vector_he(BaseDES):

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
        super(DESFHMW_JFB_vector_he, self).__init__(pool_classifiers=pool_classifiers,
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
        current_center = centers[ellipsoid_ind]
        current_cov = cov_matrices[ellipsoid_ind]
        
        # 计算新的中心（简单平均）
        new_center = (current_center + x) / 2
        
        # 计算新的协方差矩阵
        diff_center = x - current_center
        new_cov = (current_cov + np.outer(diff_center, diff_center)) / 2
        
        return new_center, new_cov

    def will_exceed_samples(self, centers, cov_matrices, ellipsoid_ind, x, con_samples):
        # 检查扩展椭球体是否会包含冲突样本
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

    def contract_ellipsoidBased(self, b_center, b_cov, cobox_center, cobox_cov):
        # 基于其他椭球体收缩当前椭球体
        # 简化实现：如果两个椭球体距离太近，调整它们的协方差矩阵以减少重叠
        # 计算两个椭球体中心之间的距离
        center_dist = np.linalg.norm(b_center - cobox_center)
        
        # 计算两个椭球体在中心连线上的有效半径之和
        # 简化处理：使用协方差矩阵的迹作为有效大小的指标
        b_size = np.sqrt(np.trace(b_cov))
        cobox_size = np.sqrt(np.trace(cobox_cov))
        
        # 如果两个椭球体重叠
        if center_dist < b_size + cobox_size:
            # 按比例缩小两个椭球体
            scale_factor = center_dist / (b_size + cobox_size + 1e-8)  # 防止除零
            b_cov *= scale_factor
            cobox_cov *= scale_factor
        
        return b_center, b_cov, cobox_center, cobox_cov

    def update_ellipsoids(self, h_centers, h_covs, co_centers, co_covs, missClsSample):
        if missClsSample:  # 误分类样本
            n_centers = h_centers
            n_covs = h_covs
            p_centers = co_centers
            p_covs = co_covs
        else:  # 正确分类样本
            p_centers = h_centers
            p_covs = h_covs
            n_centers = co_centers
            n_covs = co_covs
        return n_centers, n_covs, p_centers, p_covs

    def fit(self, X, y):
        super(DESFHMW_JFB_vector_he, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [centers, cov_matrices] = self.FMM_train(classifier_index)
                class_dic =  { "clsr" : classifier_index, "centers" : centers, "cov_matrices" : cov_matrices }
                self.NO_ellipsoids += len(centers)
                self.HEllipsoids.append(class_dic)
        else:
            # (改)修改进程数计算方式：不超过CPU核心数且最多4个进程
            no_processes = min(multiprocessing.cpu_count(), 4)
            with multiprocessing.Pool(processes=no_processes) as pool:
                try:
                    list = pool.map(self.FMM_train, range(self.n_classifiers_))
                except OSError as e:
                    if "1450" in str(e):
                        # 回退到单进程模式
                        no_processes = 1
                        with multiprocessing.Pool(processes=no_processes) as pool:
                            list = pool.map(self.FMM_train, range(self.n_classifiers_))
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

    def FMM_train(self, classifier_ind):
        # 初始化正常分类和误分类样本的椭球体
        n_centers = np.zeros((0, self.n_features_))
        n_covs = np.zeros((0, self.n_features_, self.n_features_))

        p_centers = np.zeros((0, self.n_features_))
        p_covs = np.zeros((0, self.n_features_, self.n_features_))

        X = self.DSEL_data_
        y = self.DSEL_processed_[:, classifier_ind]

        # 为数据添加一些扰动以避免协方差矩阵奇异
        X = X + np.random.normal(0, self.reg_lambda, X.shape)

        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            X, y = shuffle(X, y, random_state=classifier_ind)

        for ind, x in enumerate(X):
            missClassified = y[ind] == False
            ############## Type: Miss or Correct Classified ############
            if missClassified: # 误分类样本
                h_centers = n_centers
                h_covs = n_covs
                co_centers = p_centers
                co_covs = p_covs
            else: # 正确分类样本
                h_centers = p_centers
                h_covs = p_covs
                co_centers = n_centers
                co_covs = n_covs

            #############################################################
            # 创建第一个椭球体
            if len(h_centers) == 0:
                h_centers = np.array([x])
                # 初始协方差矩阵设为单位矩阵乘以一个小系数
                initial_cov = np.eye(self.n_features_) * self.theta
                h_covs = np.array([initial_cov])
                n_centers, n_covs, p_centers, p_covs = self.update_ellipsoids(h_centers, h_covs, co_centers, co_covs, missClassified)
                continue

            # X是否在某个椭球体内？
            is_inEllipsoid = False
            for ellipsoidInd in range(len(h_centers)):
                if self.is_inside(h_centers, h_covs, ellipsoidInd, x):
                    is_inEllipsoid = True
                    break
            if is_inEllipsoid:
                # 不需要操作
                continue

            ######################## 扩展 ############################
            # 按距离排序椭球体
            distances = np.array([self.compute_mahalanobis_distance(x.reshape(1, -1), h_centers[i], h_covs[i])[0] for i in range(len(h_centers))])
            sorted_indexes = np.argsort(distances)[::-1]  # 从最远到最近排序
            expanded = False
            
            for ind_ellipsoid in sorted_indexes:
                # 更新椭球体
                new_center, new_cov = self.update_ellipsoid(h_centers, h_covs, ind_ellipsoid, x)
                h_centers[ind_ellipsoid] = new_center
                h_covs[ind_ellipsoid] = new_cov
                
                # 如果有其他类型的椭球体，进行收缩以避免重叠
                if len(co_centers) > 0:
                    # 找到最近的冲突椭球体
                    co_distances = np.array([self.compute_mahalanobis_distance(h_centers[ind_ellipsoid].reshape(1, -1), co_centers[i], co_covs[i])[0] for i in range(len(co_centers))])
                    nearest_co_ind = np.argmin(co_distances)
                    
                    # 进行收缩
                    b_center, b_cov, cobox_center, cobox_cov = self.contract_ellipsoidBased(
                        h_centers[ind_ellipsoid], h_covs[ind_ellipsoid],
                        co_centers[nearest_co_ind], co_covs[nearest_co_ind]
                    )
                    h_centers[ind_ellipsoid] = b_center
                    h_covs[ind_ellipsoid] = b_cov
                    co_centers[nearest_co_ind] = cobox_center
                    co_covs[nearest_co_ind] = cobox_cov
                
                expanded = True
                break

            ######################## 创建新椭球体 ############################
            if expanded == False:
                # 创建新的椭球体
                new_center = x
                # 初始协方差矩阵设为单位矩阵乘以一个小系数
                new_cov = np.eye(self.n_features_) * self.theta
                h_centers, h_covs = self.add_ellipsoids(h_centers, h_covs, new_center, new_cov)

            # 更新椭球体集合
            n_centers, n_covs, p_centers, p_covs = self.update_ellipsoids(h_centers, h_covs, co_centers, co_covs, missClassified)

        if self.mis_sample_based:
            return n_centers, n_covs
        else:
            return p_centers, p_covs

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers