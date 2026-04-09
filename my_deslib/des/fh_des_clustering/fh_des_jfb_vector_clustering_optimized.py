# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.fh_des_JFB_vector import FHDES_JFB_vector
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing
import multiprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

class FHDES_JFB_vector_clustering_optimized(FHDES_JFB_vector):
    
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
                 # 优化版聚类和密度估计参数
                 n_clusters=5,
                 bandwidth_selection='silverman',  # 'silverman', 'cv', 'adaptive', 固定值
                 bandwidth=1.0,
                 density_estimation='kde',  # 'kde', 'knn_density', 'pca_kde'
                 knn_k=5,
                 pca_components=None,  # PCA降维的组件数，None表示自动选择
                 use_pca=False,  # 是否使用PCA降维
                 density_weight=0.5,
                 cluster_distance_weight=0.5,
                 basic_weight=0.7,  # 基本竞争力权重
                 enhancement_weight=0.3,  # 增强竞争力权重
                 normalize_weights=True,  # 是否归一化权重
                 epsilon=1e-8,  # 数值稳定性参数
                 normalize_correlations=True):  # 归一化超盒相关系数，防止数量偏差
        
        # 调用父类初始化
        super(FHDES_JFB_vector_clustering_optimized, self).__init__(
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
        self.bandwidth_selection = bandwidth_selection  # 带宽选择方法
        self.bandwidth = bandwidth  # 核密度估计带宽
        self.density_estimation = density_estimation  # 密度估计方法
        self.knn_k = knn_k  # kNN-density的k值
        self.pca_components = pca_components  # PCA降维的组件数
        self.use_pca = use_pca  # 是否使用PCA降维
        self.density_weight = density_weight  # 密度权重
        self.cluster_distance_weight = cluster_distance_weight  # 距离权重
        self.basic_weight = basic_weight  # 基本竞争力权重
        self.enhancement_weight = enhancement_weight  # 增强竞争力权重
        self.normalize_weights = normalize_weights  # 是否归一化权重
        self.epsilon = epsilon  # 数值稳定性参数
        self.normalize_correlations = normalize_correlations  # 归一化超盒相关系数，防止数量偏差
        
        # 确保权重规范化
        total_weight = density_weight + cluster_distance_weight
        if total_weight > 0:
            self.density_weight = density_weight / total_weight
            self.cluster_distance_weight = cluster_distance_weight / total_weight
        
        total_mix_weight = basic_weight + enhancement_weight
        if total_mix_weight > 0:
            self.basic_weight = basic_weight / total_mix_weight
            self.enhancement_weight = enhancement_weight / total_mix_weight
        
        # 存储聚类和密度相关信息
        self.cluster_centers_ = None  # 簇中心
        self.cluster_kdes_ = None     # 每个簇的KDE模型
        self.cluster_labels_ = None   # 训练样本的簇标签
        self.cluster_densities_ = None # 每个样本的局部密度
        self.cluster_box_adjustments_ = None # 每个簇的超盒调整因子
        self.pca_ = None  # PCA模型
        self.cluster_bandwidths_ = []  # 每个簇的自适应带宽
    
    def fit(self, X, y):
        # 调用父类fit方法
        super(FHDES_JFB_vector_clustering_optimized, self).fit(X, y)
        self.n_features_ = X.shape[1]
        
        # 如果需要PCA降维，先降维
        if self.use_pca or (self.density_estimation == 'pca_kde' and (self.pca_components is not None or self.n_features_ > 10)):
            self._apply_pca()
        
        # 执行聚类和密度估计
        self._perform_clustering()
        self._estimate_densities()
        self._calculate_box_adjustments()
        
        # 基于聚类和密度调整超盒
        self._adjust_boxes_by_density()
        
        return self
    
    def _apply_pca(self):
        """应用PCA降维以解决维数灾难问题"""
        X_dsel = self.DSEL_data_
        n_samples, n_features = X_dsel.shape
        
        # 确定PCA组件数
        if self.pca_components is None:
            # 自动选择，最多10个组件或样本数-1
            self.pca_components = min(10, n_samples - 1)
        else:
            # 确保组件数有效
            self.pca_components = min(self.pca_components, n_samples - 1, n_features)
        
        # 应用PCA
        self.pca_ = PCA(n_components=self.pca_components, random_state=self.random_state)
        self.DSEL_data_pca_ = self.pca_.fit_transform(X_dsel)
    
    def _calculate_silverman_bandwidth(self, data):
        """使用Silverman法则计算带宽"""
        n = len(data)
        if n == 0:
            return 1.0
        
        # 获取有效特征数
        if hasattr(self, 'DSEL_data_pca_'):
            d = self.DSEL_data_pca_.shape[1]
        else:
            d = data.shape[1]
        
        # 计算每个特征的标准差
        std_dev = np.std(data, axis=0)
        mean_std = np.mean(std_dev)
        
        # Silverman法则: h = (4/(d+2))^(1/(d+4)) * n^(-1/(d+4)) * σ
        if mean_std > 0:
            bandwidth = (4 / (d + 2)) ** (1 / (d + 4)) * n ** (-1 / (d + 4)) * mean_std
        else:
            bandwidth = 1.0
        
        return bandwidth
    
    def _perform_clustering(self):
        """使用K-means对训练数据进行聚类"""
        # 选择要使用的数据（原始或PCA降维后）
        if hasattr(self, 'DSEL_data_pca_'):
            X_cluster = self.DSEL_data_pca_
        else:
            X_cluster = self.DSEL_data_
        
        # 确保聚类数不超过样本数
        n_samples = X_cluster.shape[0]
        effective_clusters = min(self.n_clusters, max(1, n_samples))
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=effective_clusters, random_state=self.random_state)
        self.cluster_labels_ = kmeans.fit_predict(X_cluster)
        self.cluster_centers_ = kmeans.cluster_centers_
    
    def _estimate_densities(self):
        """对每个簇执行密度估计，支持多种密度估计方法"""
        # 选择要使用的数据
        if hasattr(self, 'DSEL_data_pca_'):
            X_dsel = self.DSEL_data_pca_
        else:
            X_dsel = self.DSEL_data_
        
        self.cluster_kdes_ = []
        self.cluster_densities_ = np.zeros(len(X_dsel))
        self.cluster_bandwidths_ = []
        
        # 对每个簇分别进行密度估计
        for cluster_idx in range(self.n_clusters):
            # 获取该簇的样本
            cluster_samples = X_dsel[self.cluster_labels_ == cluster_idx]
            
            if len(cluster_samples) > 0:
                if self.density_estimation == 'knn_density':
                    # 使用kNN-density估计
                    self._estimate_knn_density(cluster_idx, cluster_samples)
                else:
                    # 使用KDE估计
                    self._estimate_kde_density(cluster_idx, cluster_samples)
            else:
                # 如果簇为空，添加None作为占位
                self.cluster_kdes_.append(None)
                self.cluster_bandwidths_.append(1.0)
    
    def _estimate_kde_density(self, cluster_idx, cluster_samples):
        """使用KDE估计密度，支持多种带宽选择方法"""
        # 确定带宽
        if self.bandwidth_selection == 'silverman':
            bandwidth = self._calculate_silverman_bandwidth(cluster_samples)
        elif self.bandwidth_selection == 'cv':
            # 使用交叉验证选择带宽
            bandwidths = np.logspace(-1, 1, 20)
            # 确保cv至少为2，且不超过样本数
            cv_value = max(2, min(5, len(cluster_samples)))
            # 如果样本数少于2，则不使用交叉验证，直接使用默认带宽
            if len(cluster_samples) < 2:
                bandwidth = 1.0
            else:
                grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                    {'bandwidth': bandwidths},
                                    cv=cv_value)
                grid.fit(cluster_samples)
                bandwidth = grid.best_params_['bandwidth']
        elif self.bandwidth_selection == 'adaptive':
            # 自适应带宽：基于簇内样本数和方差
            n_samples = len(cluster_samples)
            variance = np.mean(np.var(cluster_samples, axis=0))
            if variance > 0:
                bandwidth = self.bandwidth * np.sqrt(variance) * n_samples ** (-0.1)
            else:
                # 当方差为0时，使用默认带宽
                bandwidth = self.bandwidth
        else:
            # 使用固定带宽
            bandwidth = self.bandwidth
        
        # 创建KDE模型
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(cluster_samples)
        self.cluster_kdes_.append(kde)
        self.cluster_bandwidths_.append(bandwidth)
        
        # 计算该簇内样本的密度
        log_density = kde.score_samples(cluster_samples)
        self.cluster_densities_[self.cluster_labels_ == cluster_idx] = np.exp(log_density)
    
    def _estimate_knn_density(self, cluster_idx, cluster_samples):
        """使用kNN-density估计密度"""
        n_samples = len(cluster_samples)
        effective_k = max(1, min(self.knn_k, n_samples - 1))
        
        # 使用kNN计算密度
        nn = NearestNeighbors(n_neighbors=effective_k)
        nn.fit(cluster_samples)
        distances, _ = nn.kneighbors(cluster_samples)
        
        # 密度与最近邻距离的倒数成正比
        # 计算k个最近邻的平均距离
        avg_distances = np.mean(distances, axis=1)
        
        # 添加小常数防止除零
        densities = 1.0 / (avg_distances + 1e-8)
        
        # 存储密度值
        self.cluster_densities_[self.cluster_labels_ == cluster_idx] = densities
        
        # 存储nn模型用于后续查询点密度估计
        self.cluster_kdes_.append(nn)
        self.cluster_bandwidths_.append(effective_k)
    
    def _calculate_box_adjustments(self):
        """计算每个簇的超盒调整因子"""
        # 添加小常数确保数值稳定性
        density_min = np.min(self.cluster_densities_)
        density_max = np.max(self.cluster_densities_)
        
        # 归一化密度值
        if density_max - density_min > 1e-8:
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
                # 如果使用了PCA，需要将超盒中心也投影到PCA空间
                if hasattr(self, 'pca_') and self.pca_ is not None:
                    try:
                        box_center_pca = self.pca_.transform(box_center.reshape(1, -1))[0]
                        distances_to_centers = cdist([box_center_pca], self.cluster_centers_)[0]
                    except ValueError:
                        # 如果转换失败，说明超盒和DSEL_data_的特征数量不匹配
                        # 这种情况下，我们应该使用原始空间中的聚类中心
                        # 重新在原始DSEL数据上计算聚类中心（如果还没有的话）
                        if not hasattr(self, 'cluster_centers_original_'):
                            # 在原始数据上计算每个簇的中心
                            X_dsel = self.DSEL_data_
                            n_clusters = len(self.cluster_centers_)
                            self.cluster_centers_original_ = np.zeros((n_clusters, X_dsel.shape[1]))
                            for cluster_idx in range(n_clusters):
                                cluster_samples = X_dsel[self.cluster_labels_ == cluster_idx]
                                if len(cluster_samples) > 0:
                                    self.cluster_centers_original_[cluster_idx] = np.mean(cluster_samples, axis=0)
                                else:
                                    # 如果簇中没有样本，使用一个默认值
                                    self.cluster_centers_original_[cluster_idx] = np.zeros(X_dsel.shape[1])
                        
                        # 使用原始空间的聚类中心计算距离
                        distances_to_centers = cdist([box_center], self.cluster_centers_original_)[0]
                else:
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
        basic_competences = super(FHDES_JFB_vector_clustering_optimized, self).estimate_competence(
            query, neighbors, distances, predictions
        )
        
        # 如果没有进行聚类（可能是因为没有数据），直接返回基本隶属度
        if self.cluster_centers_ is None:
            return basic_competences
        
        # 计算基于聚类和密度的增强隶属度
        enhanced_competences = self._calculate_enhanced_competence(query, basic_competences)
        
        return enhanced_competences
    
    def _calculate_enhanced_competence(self, query, basic_competences):
        """结合聚类和密度信息计算增强的隶属度，改进了超盒数量偏差问题"""
        Xq = np.array(query).reshape(-1, self.n_features_)
        n_queries = Xq.shape[0]
        
        enhanced_competences = np.zeros((n_queries, self.n_classifiers_))
        
        for i in range(n_queries):
            query_point = Xq[i:i+1]
            
            # 如果使用了PCA，需要将查询点也投影到PCA空间
            if self.pca_ is not None:
                query_point_pca = self.pca_.transform(query_point)
                distances_to_centers = cdist(query_point_pca, self.cluster_centers_)[0]
            else:
                distances_to_centers = cdist(query_point, self.cluster_centers_)[0]
            
            # 归一化距离（距离越小，值越大）
            max_distance = np.max(distances_to_centers)
            if max_distance > 1e-8:
                normalized_distances = 1.0 - (distances_to_centers / max_distance)
            else:
                normalized_distances = np.ones_like(distances_to_centers)
            
            # 计算查询点在每个簇中的密度
            query_densities = np.zeros(self.n_clusters)
            for cluster_idx in range(self.n_clusters):
                if self.cluster_kdes_[cluster_idx] is not None:
                    if self.density_estimation == 'knn_density':
                        # 使用kNN-density计算查询点密度
                        nn_model = self.cluster_kdes_[cluster_idx]
                        effective_k = self.cluster_bandwidths_[cluster_idx]
                        
                        # 确保查询点在正确的空间中
                        if self.pca_ is not None:
                            query_point_transformed = self.pca_.transform(query_point)
                        else:
                            query_point_transformed = query_point
                        
                        dists, _ = nn_model.kneighbors(query_point_transformed, n_neighbors=effective_k)
                        avg_dist = np.mean(dists)
                        query_densities[cluster_idx] = 1.0 / (avg_dist + 1e-8)
                    else:
                        # 使用KDE计算查询点密度
                        kde = self.cluster_kdes_[cluster_idx]
                        
                        # 确保查询点在正确的空间中
                        if self.pca_ is not None:
                            query_point_transformed = self.pca_.transform(query_point)
                        else:
                            query_point_transformed = query_point
                        
                        query_densities[cluster_idx] = np.exp(kde.score_samples(query_point_transformed))
            
            # 归一化密度
            max_density = np.max(query_densities)
            if max_density > 1e-8:
                normalized_densities = query_densities / max_density
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
                        # 确保box_center是NumPy数组格式
                        box_center_array = box_center.values if hasattr(box_center, 'values') else box_center
                        if self.pca_ is not None:
                            box_center_pca = self.pca_.transform(box_center_array.reshape(1, -1))[0]
                            box_distances = cdist([box_center_pca], self.cluster_centers_)[0]
                        else:
                            box_distances = cdist([box_center_array], self.cluster_centers_)[0]
                        
                        # 归一化距离，添加小常数防止除零
                        max_box_dist = np.max(box_distances)
                        if max_box_dist > 1e-8:
                            box_normalized_distances = 1.0 - (box_distances / (max_box_dist + 1e-8))
                        else:
                            box_normalized_distances = np.ones_like(box_distances)
                        
                        # 累加到簇相关性
                        box_cluster_correlations += box_normalized_distances
                
                # 归一化簇相关性，除以超盒数量以解决超盒数量偏差问题
                n_boxes = len(clsr_boxes) * sum(len(box["Min"]) for box in clsr_boxes)
                if n_boxes > 0:
                    box_cluster_correlations = box_cluster_correlations / (n_boxes + 1e-8)
                
                # 计算分类器对查询点的增强竞争力
                clsr_enhancement = np.sum(cluster_memberships * box_cluster_correlations)
                
                # 结合基本竞争力和增强因子
                enhanced_competences[i, clsr_idx] = (basic_competences[i, clsr_idx] * self.basic_weight + 
                                                    clsr_enhancement * self.enhancement_weight)
        
        # 归一化增强竞争力
        scaler = preprocessing.MinMaxScaler()
        for i in range(n_queries):
            # 添加小常数确保数值稳定性
            if np.max(enhanced_competences[i]) > 1e-8:
                enhanced_competences[i] = scaler.fit_transform(enhanced_competences[i].reshape(-1, 1)).flatten()
            else:
                enhanced_competences[i] = np.ones_like(enhanced_competences[i])
        
        return enhanced_competences
    
    def setup_hyperboxs(self, classifier):
        # 调用父类的setup_hyperboxs方法构建基础超盒
        hboxV, hboxW = super(FHDES_JFB_vector_clustering_optimized, self).setup_hyperboxs(classifier)
        
        return hboxV, hboxW