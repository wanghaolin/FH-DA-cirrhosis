# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
from my_deslib.util.fuzzy_hyperbox import Hyperbox
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class FHDES_JFB_vector_sphere(BaseDES):

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
                 sigma=0.1,  # 高斯函数参数
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma  # 高斯隶属函数的标准差参数
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder

        ############### it should be based on Clustering #############################
        super(FHDES_JFB_vector_sphere, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)


    def add_spheres(self, hboxC, hboxR, bC, bR):
        hboxC = np.concatenate((hboxC, bC))
        hboxR = np.concatenate((hboxR, bR))
        return hboxC, hboxR

    def expand_sphere(self, hboxC, hboxR, sphereInd, x):
        # 计算当前球心到新点的距离
        dist = np.linalg.norm(hboxC[sphereInd] - x)
        # 新半径是原半径和距离的最大值
        hboxR[sphereInd] = max(hboxR[sphereInd], dist)
    
    def is_expandable(self, hboxC, hboxR, sphereInd, x):
        # 计算扩展后的半径
        new_radius = max(hboxR[sphereInd], np.linalg.norm(hboxC[sphereInd] - x))
        # 检查是否超过阈值
        return new_radius < self.theta
    
    def is_inside(self, hboxC, hboxR, sphereInd, x):
        # 检查点是否在球内
        dist = np.linalg.norm(hboxC[sphereInd] - x)
        return dist <= hboxR[sphereInd]
    
    def membership_spheres(self, hboxC, hboxR, Xq):
        """高斯隶属函数计算"""
        NO_hypespheres, n_features = hboxC.shape
        
        # 重塑以支持批量计算
        spheres_center = hboxC.reshape(NO_hypespheres, 1, n_features)
        spheres_radius = hboxR.reshape(NO_hypespheres, 1)
        
        # 计算每个样本到每个球心的欧氏距离
        distances = np.linalg.norm(spheres_center - Xq, axis=2)
        
        # 高斯隶属函数: m = exp(-(distance^2)/(2*sigma^2))
        # 当距离超过半径时，隶属度快速衰减
        m = np.exp(-(distances**2) / (2 * self.sigma**2))
        
        # 对于距离超过半径的点，进一步降低隶属度
        radius_factor = np.maximum(0, 1 - (distances / (spheres_radius + 1e-9)))
        m = m * radius_factor
        
        return m
    
    def will_exceed_samples(self, hboxC, hboxR, sphereInd, x, con_samples):
        """检查扩展球体是否会包含收缩样本"""
        # 计算扩展后的半径
        new_radius = max(hboxR[sphereInd], np.linalg.norm(hboxC[sphereInd] - x))
        
        # 检查收缩样本是否在扩展后的球体内
        con_distances = np.linalg.norm(con_samples - hboxC[sphereInd], axis=1)
        return np.any(con_distances <= new_radius)
    
    def contract_samplesBased(self, hboxC, hboxR, sphereInd, con_samples):
        """基于冲突样本收缩球体"""
        # 计算当前球体包含的冲突样本
        con_distances = np.linalg.norm(con_samples - hboxC[sphereInd], axis=1)
        conflicting_inds = np.where(con_distances <= hboxR[sphereInd])[0]
        
        for ind in conflicting_inds:
            # 对于每个冲突样本，将半径缩小到刚好不包含它
            conflict_dist = con_distances[ind]
            # 稍微减小半径，确保不包含冲突样本
            hboxR[sphereInd] = conflict_dist * 0.95

    def fit(self, X, y):
        super(FHDES_JFB_vector_sphere, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bC, bR] = self.setup_hyperspheres(classifier_index)
                class_dic =  { "clsr" : classifier_index, "Center" : bC, "Radius" : bR }
                self.NO_hypeboxes += len(bC)
                self.HBoxes.append(class_dic)
        else:
            # 修改进程数计算方式：不超过CPU核心数且最多4个进程
            no_processes = min(multiprocessing.cpu_count(), 4)
            with multiprocessing.Pool(processes=no_processes) as pool:
                try:
                    list = pool.map(self.setup_hyperspheres, range(self.n_classifiers_))
                except OSError as e:
                    if "1450" in str(e):
                        # 回退到单进程模式
                        no_processes = 1
                        with multiprocessing.Pool(processes=no_processes) as pool:
                            list = pool.map(self.setup_hyperspheres, range(self.n_classifiers_))
                for i, clsr_sphere in enumerate(list):
                    class_dic = {"clsr": i, "Center": clsr_sphere[0], "Radius": clsr_sphere[1]}
                    self.NO_hypeboxes += len(clsr_sphere[0])
                    self.HBoxes.append(class_dic)

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):

        if self.mis_sample_based:
            highest_mems = np.ones([len(query), self.n_classifiers_])
        else:
            highest_mems = np.zeros([len(query), self.n_classifiers_])

        Xq = query.reshape(1, len(query), self.n_features_)
        
        # 按分类器分组处理超球
        for clsr_idx in range(self.n_classifiers_):
            # 收集当前分类器的所有超球
            clsr_spheres = [sphere for sphere in self.HBoxes if sphere["clsr"] == clsr_idx]
            
            if not clsr_spheres:
                continue  # 如果没有该分类器的超球，跳过
            
            # 对每个查询样本计算最高隶属度
            for i in range(len(query)):
                max_membership = 0
                
                for sphere in clsr_spheres:
                    hboxC = sphere["Center"]
                    hboxR = sphere["Radius"]
                    
                    # 计算当前超球的隶属度
                    clsrSpheres_m = self.membership_spheres(hboxC, hboxR, Xq)
                    
                    # 更新最高隶属度
                    if len(hboxC) > 1:
                        # 使用前两个最高隶属度超球的加权和
                        bb_indexes = np.argsort(-clsrSpheres_m, axis=0)
                        b1 = bb_indexes[0, i]
                        b2 = bb_indexes[1, i]
                        membership = clsrSpheres_m[b1, i] * 0.7 + clsrSpheres_m[b2, i] * 0.3
                    else:
                        # 只有一个超球的情况
                        membership = clsrSpheres_m[0, i]
                    
                    if membership > max_membership:
                        max_membership = membership
                
                # 存储最高隶属度
                highest_mems[i, clsr_idx] = max_membership

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(highest_mems) - highest_mems
        else:
            competences_ = highest_mems

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_

    def setup_hyperspheres(self, classifier):
        if np.size(classifier) < 0:
            pass

        if(self.mis_sample_based):
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]

        # 初始化球心和半径
        hboxC = np.zeros((1, self.n_features_)) - 1
        hboxR = np.zeros((1,)) - 1

        selected_samples = self.DSEL_data_[samples_ind, :]
        contraction_samples = self.DSEL_data_[Contraction_ind, :]
        
        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples, random_state = classifier)

        for ind, X in enumerate(selected_samples):
            # 创建第一个球
            if hboxC[0, 0] == -1:
                hboxC[0, :] = X
                hboxR[0] = 0  # 初始半径为0
                continue

            # 检查X是否在某个球内
            is_inSphere = False
            for sphereInd in range(len(hboxC)):
                if self.is_inside(hboxC, hboxR, sphereInd, X):
                    is_inSphere = True
                    break
            if is_inSphere:
                continue

            ######################## Expand ############################
            # 按距离排序球
            expanded = False
            distances = np.linalg.norm(X - hboxC, axis=1)
            nearestSphere_ind = np.argmin(distances)
            
            if self.thetaCheck and self.doContraction:
                if self.is_expandable(hboxC, hboxR, nearestSphere_ind, X):
                    self.expand_sphere(hboxC, hboxR, nearestSphere_ind, X)
                    self.contract_samplesBased(hboxC, hboxR, nearestSphere_ind, contraction_samples)
                    continue

            elif self.thetaCheck and not self.doContraction:
                if self.is_expandable(hboxC, hboxR, nearestSphere_ind, X):
                    self.expand_sphere(hboxC, hboxR, nearestSphere_ind, X)
                    continue

            elif not self.thetaCheck and self.doContraction:
                if not self.will_exceed_samples(hboxC, hboxR, nearestSphere_ind, X, contraction_samples):
                    self.expand_sphere(hboxC, hboxR, nearestSphere_ind, X)
                    expanded = True
                    continue
            
            ######################## Creation ############################
            # 创建新球
            xt = X.reshape(1, self.n_features_)
            rt = np.array([0])  # 新球初始半径为0
            hboxC, hboxR = self.add_spheres(hboxC, hboxR, bC=xt, bR=rt)

        return hboxC, hboxR

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers