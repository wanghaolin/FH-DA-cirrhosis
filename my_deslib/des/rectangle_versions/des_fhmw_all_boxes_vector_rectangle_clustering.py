# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class DESFHMW_allboxes_vector_rectangle_clustering(BaseDES):

    def __init__(self, pool_classifiers=None,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta=0.05,
                 mu=0.991,
                 alpha=0.5,  # 双斜率函数参数
                 beta=2.0,   # 双斜率函数参数
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False,
                 # 新增聚类和密度估计参数
                 n_clusters=5,
                 bandwidth=1.0,
                 density_weight=0.5,
                 cluster_distance_weight=0.5,
                 # 额外参数支持
                 gamma=0.8,
                 reg_lambda=1e-6):
        # 定义类特定参数
        self.theta = theta
        self.mu = mu
        self.alpha = alpha  # 内部区域斜率
        self.beta = beta    # 外部区域斜率
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder
        
        # 新增聚类和密度估计参数
        self.n_clusters = n_clusters  # K-means聚类数量
        self.bandwidth = bandwidth    # 核密度估计带宽
        self.density_weight = density_weight          # 密度权重
        self.cluster_distance_weight = cluster_distance_weight  # 距离权重
        
        # 额外参数
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        
        # 存储聚类和密度相关信息
        self.cluster_centers_ = None  # 簇中心
        self.cluster_kdes_ = None     # 每个簇的KDE模型
        self.cluster_labels_ = None   # 训练样本的簇标签
        self.cluster_densities_ = None # 每个样本的局部密度
        self.cluster_box_adjustments_ = None # 每个簇的超盒调整因子

        # 调用父类初始化
        super(DESFHMW_allboxes_vector_rectangle_clustering, self).__init__(pool_classifiers=pool_classifiers,
                                    with_IH=with_IH,
                                    safe_k=safe_k,
                                    IH_rate=IH_rate,
                                    mode='hybrid',  # hybrid,weighting
                                    random_state=random_state,
                                    DSEL_perc=DSEL_perc)

    def add_boxes(self, hboxV, hboxW, bV, bW):
        hboxV = np.concatenate((hboxV, bV))
        hboxW = np.concatenate((hboxW, bW))
        return hboxV, hboxW

    def expand_box(self, hboxV, hboxW, boxInd, x):
        hboxV[boxInd] = np.minimum(hboxV[boxInd], x)
        hboxW[boxInd] = np.maximum(hboxW[boxInd], x)
    
    def is_expandable(self, hboxV, hboxW, boxInd, x):
        candV = np.minimum(hboxV[boxInd], x)
        candW = np.maximum(hboxW[boxInd], x)
        return all((candW - candV) < self.theta)
    
    def is_inside(self, hboxV, hboxW, boxInd, x):
        return np.all(hboxV[boxInd] < x) and np.all(hboxW[boxInd] > x)
    
    def membership_boxes(self, hboxV, hboxW, Xq):
        """双斜率隶属函数计算"""
        NO_hypeboxes, n_features = hboxV.shape
        
        # 计算盒的中心和半宽
        hboxC = (hboxV + hboxW) / 2
        halfsize = (hboxW - hboxV) / 2
        
        # 重塑以支持批量计算
        boxes_center = hboxC.reshape(NO_hypeboxes, 1, n_features)
        boxes_halfsize = halfsize.reshape(NO_hypeboxes, 1, n_features)
        
        # 计算每个维度上的相对距离
        rel_dist = np.abs(boxes_center - Xq) / (boxes_halfsize + 1e-9)
        
        # 双斜率隶属函数
        # 内部区域 (rel_dist <= 1): 缓慢下降
        # 外部区域 (rel_dist > 1): 快速下降
        m_internal = 1 - self.alpha * (rel_dist - np.minimum(rel_dist, np.ones_like(rel_dist)))
        m_external = np.exp(-self.beta * (rel_dist - np.ones_like(rel_dist)))
        
        # 组合内部和外部隶属度
        m = np.where(rel_dist <= 1, m_internal, m_external)
        
        # 对每个超盒的所有维度取最小值作为最终隶属度
        m = np.min(m, axis=2)
        
        # 确保隶属度在0-1之间
        m = np.maximum(0, np.minimum(1, m))
        
        return m
    
    def will_exceed_samples(self, hboxV, hboxW, boxInd, x, con_samples):
        candidV = np.minimum(hboxV[boxInd], x)
        candidW = np.maximum(hboxW[boxInd], x)
        V = candidV.reshape(1, self.n_features_)
        W = candidW.reshape(1, self.n_features_)
        return np.any(np.all(V <= con_samples, 1) & np.all(W >= con_samples, 1))
    
    def contract_samplesBased(self, hboxV, hboxW, boxInd, con_samples):
        di = 0
        v = hboxV[boxInd]
        w = hboxW[boxInd]
        mi = con_samples - v
        ma = w - con_samples

        inds1 = np.all(mi > 0, 1)
        inds2 = np.all(ma > 0, 1)
        confilicts_inds = np.where(inds1 & inds2)

        for ind in list(confilicts_inds[0]):
            if np.all(v < con_samples[ind]) and np.all(w > con_samples[ind]):
                mi = con_samples[ind] - v
                ma = w - con_samples[ind]
                if min(mi) < min(ma):
                    d = np.where(mi == min(mi))
                    hboxV[boxInd, d] = con_samples[ind, d]
                else:
                    d = np.where(ma == min(ma))
                    hboxW[boxInd, d] = con_samples[ind, d]

    def setup_hyperboxs(self, classifier):
        if np.size(classifier) < 0:
            pass

        if(self.mis_sample_based):
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]

        hboxV = np.zeros((1, self.n_features_)) - 1
        hboxW = np.zeros((1, self.n_features_)) - 1

        selected_samples = self.DSEL_data_[samples_ind, :]
        contraction_samples = self.DSEL_data_[Contraction_ind, :]
        
        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples, random_state = classifier)

        for ind, X in enumerate(selected_samples):
            # Creation first box
            if hboxV[0, 0] == -1:
                hboxV[0, :] = X
                hboxW[0, :] = X
                continue

            # X is in a box?
            is_inBox = False
            for boxInd in range(len(hboxV)):
                if self.is_inside(hboxV, hboxW, boxInd, X):
                    is_inBox = True
                    break
            if is_inBox:
                continue

            ######################## Expand ############################
            # Sort boxes by the distances
            hboxC = (hboxV + hboxW) / 2
            expanded = False
            box_list = np.linalg.norm(X - hboxC, axis=1)

            nearestBox_ind = np.argmin(box_list)
            if self.thetaCheck and self.doContraction:
                if self.is_expandable(hboxV, hboxW, nearestBox_ind, X):
                    self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                    self.contract_samplesBased(hboxV, hboxW, nearestBox_ind, contraction_samples)
                    continue

            elif self.thetaCheck and not self.doContraction:
                if self.is_expandable(hboxV, hboxW, nearestBox_ind, X):
                    self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                    continue

            elif not self.thetaCheck and self.doContraction:
                if not self.will_exceed_samples(hboxV, hboxW, nearestBox_ind, X, contraction_samples):
                    self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                    expanded = True
                    continue
            ######################## Creation ############################
            xt = X.reshape(1, self.n_features_)
            hboxV, hboxW = self.add_boxes(hboxV, hboxW, bV=xt, bW=xt)

        return hboxV, hboxW

    def _perform_clustering(self):
        """使用K-means对训练数据进行聚类"""
        # 使用DSEL数据进行聚类
        X_dsel = self.DSEL_data_
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.cluster_labels_ = kmeans.fit_predict(X_dsel)
        self.cluster_centers_ = kmeans.cluster_centers_
    
    def _estimate_densities(self):
        """对每个簇执行核密度估计"""
        X_dsel = self.DSEL_data_
        self.cluster_kdes_ = []
        self.cluster_densities_ = np.zeros(len(X_dsel))
        
        # 对每个簇分别进行密度估计
        for cluster_idx in range(self.n_clusters):
            # 获取该簇的样本
            cluster_samples = X_dsel[self.cluster_labels_ == cluster_idx]
            
            if len(cluster_samples) > 0:
                # 为每个簇创建一个KDE模型
                kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
                kde.fit(cluster_samples)
                self.cluster_kdes_.append(kde)
                
                # 计算该簇内样本的密度
                log_density = kde.score_samples(cluster_samples)
                self.cluster_densities_[self.cluster_labels_ == cluster_idx] = np.exp(log_density)
            else:
                # 如果簇为空，添加一个空的KDE模型
                self.cluster_kdes_.append(None)
    
    def _calculate_box_adjustments(self):
        """计算每个簇的超盒调整因子"""
        # 归一化密度值
        density_min, density_max = np.min(self.cluster_densities_), np.max(self.cluster_densities_)
        if density_max - density_min > 0:
            normalized_densities = (self.cluster_densities_ - density_min) / (density_max - density_min)
        else:
            normalized_densities = np.ones_like(self.cluster_densities_)
        
        # 计算每个簇的平均密度
        self.cluster_box_adjustments_ = np.zeros(self.n_clusters)
        for cluster_idx in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_idx
            if np.any(cluster_mask):
                # 密度越高，调整因子越小（使超盒更紧凑）
                avg_density = np.mean(normalized_densities[cluster_mask])
                self.cluster_box_adjustments_[cluster_idx] = 1.0 - (0.5 * avg_density)  # 范围在0.5到1.0之间
            else:
                self.cluster_box_adjustments_[cluster_idx] = 1.0
    
    def _adjust_boxes_by_density(self):
        """基于聚类和密度信息调整超盒形状"""
        X_dsel = self.DSEL_data_
        
        # 对每个分类器的超盒进行调整
        for clsr_idx in range(len(self.HBoxes)):
            box = self.HBoxes[clsr_idx]
            classifier_idx = box["clsr"]
            hboxV = box["Min"]
            hboxW = box["Max"]
            
            # 为每个超盒找到最相关的簇
            for box_idx in range(len(hboxV)):
                # 计算超盒中心
                box_center = (hboxV[box_idx] + hboxW[box_idx]) / 2
                
                # 找到离超盒中心最近的簇
                distances_to_centers = cdist([box_center], self.cluster_centers_)[0]
                nearest_cluster = np.argmin(distances_to_centers)
                
                # 使用该簇的调整因子调整超盒
                adjustment_factor = self.cluster_box_adjustments_[nearest_cluster]
                
                # 调整超盒大小：密度高的区域超盒更小
                box_size = (hboxW[box_idx] - hboxV[box_idx]) * adjustment_factor
                new_center = (hboxV[box_idx] + hboxW[box_idx]) / 2
                hboxV[box_idx] = new_center - box_size / 2
                hboxW[box_idx] = new_center + box_size / 2
            
            # 更新超盒
            box["Min"] = hboxV
            box["Max"] = hboxW
    
    def _calculate_enhanced_competence(self, query, basic_competences):
        """结合聚类和密度信息计算增强的隶属度"""
        Xq = np.array(query).reshape(-1, self.n_features_)
        n_queries = Xq.shape[0]
        
        enhanced_competences = np.zeros((n_queries, self.n_classifiers_))
        
        for i in range(n_queries):
            query_point = Xq[i:i+1]
            
            # 计算查询点到每个簇中心的距离
            distances_to_centers = cdist(query_point, self.cluster_centers_)[0]
            
            # 归一化距离（距离越小，值越大）
            if np.max(distances_to_centers) > 0:
                normalized_distances = 1.0 - (distances_to_centers / np.max(distances_to_centers))
            else:
                normalized_distances = np.ones_like(distances_to_centers)
            
            # 计算查询点在每个簇中的密度
            query_densities = np.zeros(self.n_clusters)
            for cluster_idx in range(self.n_clusters):
                if self.cluster_kdes_[cluster_idx] is not None:
                    query_densities[cluster_idx] = np.exp(self.cluster_kdes_[cluster_idx].score_samples(query_point))
            
            # 归一化密度
            if np.max(query_densities) > 0:
                normalized_densities = query_densities / np.max(query_densities)
            else:
                normalized_densities = np.ones_like(query_densities)
            
            # 结合距离和密度计算查询点的簇隶属度
            cluster_memberships = (self.cluster_distance_weight * normalized_distances + 
                                   self.density_weight * normalized_densities)
            
            # 对每个分类器，计算其超盒与查询点的簇隶属度的相关性
            for clsr_idx in range(self.n_classifiers_):
                # 获取当前分类器的超盒
                clsr_boxes = [box for box in self.HBoxes if box["clsr"] == clsr_idx]
                
                if not clsr_boxes:
                    enhanced_competences[i, clsr_idx] = basic_competences[i, clsr_idx]
                    continue
                
                # 计算分类器的超盒与各簇的相关性
                box_cluster_correlations = np.zeros(self.n_clusters)
                for box in clsr_boxes:
                    hboxV = box["Min"]
                    hboxW = box["Max"]
                    
                    for box_idx in range(len(hboxV)):
                        # 超盒中心
                        box_center = (hboxV[box_idx] + hboxW[box_idx]) / 2
                        
                        # 计算超盒中心到各簇中心的距离
                        box_distances = cdist([box_center], self.cluster_centers_)[0]
                        box_normalized_distances = 1.0 - (box_distances / (np.max(box_distances) + 1e-8))
                        
                        # 累加到簇相关性
                        box_cluster_correlations += box_normalized_distances
                
                # 归一化簇相关性
                if np.sum(box_cluster_correlations) > 0:
                    box_cluster_correlations /= np.sum(box_cluster_correlations)
                
                # 计算分类器对查询点的增强竞争力
                clsr_enhancement = np.sum(cluster_memberships * box_cluster_correlations)
                
                # 结合基本竞争力和增强因子
                enhanced_competences[i, clsr_idx] = (basic_competences[i, clsr_idx] * 0.7 + 
                                                    clsr_enhancement * 0.3)
        
        # 归一化增强竞争力
        scaler = preprocessing.MinMaxScaler()
        for i in range(n_queries):
            enhanced_competences[i] = scaler.fit_transform(enhanced_competences[i].reshape(-1, 1)).flatten()
        
        return enhanced_competences

    def fit(self, X, y):
        super(DESFHMW_allboxes_vector_rectangle_clustering, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bV, bW] = self.setup_hyperboxs(classifier_index)
                class_dic =  { "clsr" : classifier_index, "Min" : bV, "Max" : bW }
                self.NO_hypeboxes += len(bV)
                self.HBoxes.append(class_dic)
        else:
            # 修改进程数计算方式：不超过CPU核心数且最多4个进程
            no_processes = min(multiprocessing.cpu_count(), 4)
            with multiprocessing.Pool(processes=no_processes) as pool:
                try:
                    list = pool.map(self.setup_hyperboxs, range(self.n_classifiers_))
                except OSError as e:
                    if "1450" in str(e):
                        # 回退到单进程模式
                        no_processes = 1
                        with multiprocessing.Pool(processes=no_processes) as pool:
                            list = pool.map(self.setup_hyperboxs, range(self.n_classifiers_))
                for i, clsr_box in enumerate(list):
                    class_dic = {"clsr": i, "Min": clsr_box[0], "Max": clsr_box[1]}
                    self.NO_hypeboxes += len(clsr_box[0])
                    self.HBoxes.append(class_dic)
        
        # 执行聚类和密度估计
        self._perform_clustering()
        self._estimate_densities()
        self._calculate_box_adjustments()
        
        # 基于聚类和密度调整超盒
        self._adjust_boxes_by_density()
        
        return self

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        # 计算基本竞争力（基于隶属度）
        if self.mis_sample_based:
            highest_mems = np.ones([len(query), self.n_classifiers_])
        else:
            highest_mems = np.zeros([len(query), self.n_classifiers_])

        Xq = query.reshape(1, len(query), self.n_features_)

        # 按分类器分组处理超盒
        for clsr_idx in range(self.n_classifiers_):
            # 收集当前分类器的所有超盒
            clsr_boxes = [box for box in self.HBoxes if box["clsr"] == clsr_idx]
            
            if not clsr_boxes:
                continue  # 如果没有该分类器的超盒，跳过
            
            # 对每个查询样本计算最高隶属度
            for i in range(len(query)):
                max_membership = 0
                
                for box in clsr_boxes:
                    hboxV = box["Min"]
                    hboxW = box["Max"]
                    
                    # 计算当前超盒的隶属度
                    clsrBoxes_m = self.membership_boxes(hboxV, hboxW, Xq)
                    
                    # 更新最高隶属度
                    if len(hboxV) > 1:
                        # 使用前两个最高隶属度超盒的加权和
                        bb_indexes = np.argsort(-clsrBoxes_m, axis=0)
                        b1 = bb_indexes[0, i]
                        b2 = bb_indexes[1, i]
                        membership = clsrBoxes_m[b1, i] * 0.7 + clsrBoxes_m[b2, i] * 0.3
                    else:
                        # 只有一个超盒的情况
                        membership = clsrBoxes_m[0, i]
                    
                    if membership > max_membership:
                        max_membership = membership
                
                # 存储最高隶属度
                highest_mems[i, clsr_idx] = max_membership

        #### 计算基本竞争力 ####
        if self.mis_sample_based:
            basic_competences = np.max(highest_mems) - highest_mems
        else:
            basic_competences = highest_mems

        # 归一化基本竞争力
        scaler = preprocessing.MinMaxScaler()
        basic_competences = scaler.fit_transform(basic_competences)
        
        # 如果没有进行聚类（可能是因为没有数据），直接返回基本隶属度
        if self.cluster_centers_ is None:
            return basic_competences
        
        # 计算基于聚类和密度的增强隶属度
        enhanced_competences = self._calculate_enhanced_competence(query, basic_competences)
        
        return enhanced_competences