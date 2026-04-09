# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.des_FHMW_JFB_vector import DESFHMW_JFB_vector
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing
import multiprocessing
from sklearn.utils import shuffle

class DESFHMW_JFB_vector_clustering(DESFHMW_JFB_vector):
    
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
                 mis_sample_based=True,
                 doContraction=True,
                 thetaCheck=True,
                 multiCore_process=False,
                 shuffle_dataOrder=False,
                 # 新增聚类和密度估计参数
                 n_clusters=5,
                 bandwidth=1.0,
                 density_weight=0.5,
                 cluster_distance_weight=0.5):
        
        # 调用父类初始化
        super(DESFHMW_JFB_vector_clustering, self).__init__(
            pool_classifiers=pool_classifiers,
            k=k,
            DFP=DFP,
            with_IH=with_IH,
            safe_k=safe_k,
            IH_rate=IH_rate,
            random_state=random_state,
            knn_classifier=knn_classifier,
            DSEL_perc=DSEL_perc,
            HyperBoxes=HyperBoxes,
            theta=theta,
            mu=mu,
            mis_sample_based=mis_sample_based,
            doContraction=doContraction,
            thetaCheck=thetaCheck,
            multiCore_process=multiCore_process,
            shuffle_dataOrder=shuffle_dataOrder
        )
        
        # 新增参数
        self.n_clusters = n_clusters  # K-means聚类数量
        self.bandwidth = bandwidth    # 核密度估计带宽
        self.density_weight = density_weight          # 密度权重
        self.cluster_distance_weight = cluster_distance_weight  # 距离权重
        
        # 存储聚类和密度相关信息
        self.cluster_centers_ = None  # 簇中心
        self.cluster_kdes_ = None     # 每个簇的KDE模型
        self.cluster_labels_ = None   # 训练样本的簇标签
        self.cluster_densities_ = None # 每个样本的局部密度
        self.cluster_box_adjustments_ = None # 每个簇的超盒调整因子
    
    def fit(self, X, y):
        # 调用父类fit方法
        super(DESFHMW_JFB_vector_clustering, self).fit(X, y)
        
        # 执行聚类和密度估计
        self._perform_clustering()
        self._estimate_densities()
        self._calculate_box_adjustments()
        
        # 基于聚类和密度调整超盒
        self._adjust_boxes_by_density()
        
        return self
    
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
            
            # 确定该分类器的样本
            if self.mis_sample_based:
                samples_ind = ~self.DSEL_processed_[:, classifier_idx]
            else:
                samples_ind = self.DSEL_processed_[:, classifier_idx]
            
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
    
    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        # 调用父类的estimate_competence获取基本隶属度
        basic_competences = super(DESFHMW_JFB_vector_clustering, self).estimate_competence(
            query, neighbors, distances, predictions
        )
        
        # 如果没有进行聚类（可能是因为没有数据），直接返回基本隶属度
        if self.cluster_centers_ is None:
            return basic_competences
        
        # 计算基于聚类和密度的增强隶属度
        enhanced_competences = self._calculate_enhanced_competence(query, basic_competences)
        
        return enhanced_competences
    
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
    
    def FMM_train(self, classifier_ind):
        # 调用父类的FMM_train方法构建基础超盒
        hboxV, hboxW = super(DESFHMW_JFB_vector_clustering, self).FMM_train(classifier_ind)
        
        # 注意：超盒的调整在_adjust_boxes_by_density方法中进行，这里保持父类的实现不变
        
        return hboxV, hboxW