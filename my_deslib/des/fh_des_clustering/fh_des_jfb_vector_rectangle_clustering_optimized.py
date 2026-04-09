# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.rectangle_versions.fh_des_JFB_vector_rectangle import FHDES_JFB_vector_rectangle
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing
import multiprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

class FHDES_JFB_vector_rectangle_clustering_optimized(FHDES_JFB_vector_rectangle):
    
    def __init__(self, pool_classifiers=None,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
                 DSEL_perc=0.5,
                 HyperBoxes=[],
                 theta=0.05,
                 mu=0.991,
                 alpha=0.2,
                 beta=0.2,
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
        # 确保knn_k至少为1
        self.knn_k = max(1, knn_k)
        
        # 调用父类初始化
        super(FHDES_JFB_vector_rectangle_clustering_optimized, self).__init__(
            pool_classifiers=pool_classifiers,
            with_IH=with_IH,
            safe_k=safe_k,
            IH_rate=IH_rate,
            random_state=random_state,
            DSEL_perc=DSEL_perc,
            theta=theta,
            mu=mu,
            alpha=alpha,
            beta=beta,
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
        self.normalize_correlations = normalize_correlations  # 归一化超盒相关系数
        
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
        self.pca_ = None  # PCA模型
        self.cluster_bandwidths_ = []  # 每个簇的自适应带宽
    
    def fit(self, X, y):
        # 调用父类fit方法
        super(FHDES_JFB_vector_rectangle_clustering_optimized, self).fit(X, y)
        self.n_features_ = X.shape[1]
        self.X_ = X
        self.y_ = y
        
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
        
        # 同时在原始空间计算簇中心，用于后续与超盒中心进行比较
        # 注意：cluster_labels_是基于聚类空间的，所以我们需要将原始数据映射到聚类空间
        if hasattr(self, 'DSEL_data_pca_'):
            # 在原始空间计算每个簇的中心
            self.cluster_centers_original_ = np.zeros((effective_clusters, self.DSEL_data_.shape[1]))
            for cluster_idx in range(effective_clusters):
                cluster_samples_original = self.DSEL_data_[self.cluster_labels_ == cluster_idx]
                if len(cluster_samples_original) > 0:
                    self.cluster_centers_original_[cluster_idx] = np.mean(cluster_samples_original, axis=0)
                else:
                    # 如果簇中没有样本，使用一个默认值
                    self.cluster_centers_original_[cluster_idx] = np.zeros(self.DSEL_data_.shape[1])
        else:
            # 如果没有使用PCA，直接使用原始空间的簇中心
            self.cluster_centers_original_ = self.cluster_centers_
    
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
                # 计算标准差用于Silverman法则
                std_dev = np.std(cluster_samples, axis=0)
                mean_std = np.mean(std_dev)
                
                # 选择带宽
                if self.bandwidth_selection == 'silverman':
                    cluster_bandwidth = self._calculate_silverman_bandwidth(cluster_samples)
                elif self.bandwidth_selection == 'cv':
                    # 使用交叉验证选择带宽
                    if len(cluster_samples) >= 2:
                        bandwidths = np.logspace(-2, 1, 20)
                        cv_folds = min(5, len(cluster_samples))
                        grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                           {'bandwidth': bandwidths},
                                           cv=cv_folds)
                        grid.fit(cluster_samples)
                        cluster_bandwidth = grid.best_params_['bandwidth']
                    else:
                        # 样本数量不足，使用Silverman规则
                        cluster_bandwidth = self._calculate_silverman_bandwidth(cluster_samples)
                elif self.bandwidth_selection == 'adaptive':
                    # 自适应带宽
                    cluster_bandwidth = self.bandwidth * (1 + cluster_idx / self.n_clusters)
                else:
                    # 固定带宽
                    cluster_bandwidth = self.bandwidth
                
                self.cluster_bandwidths_.append(cluster_bandwidth)
                
                # 密度估计
                if self.density_estimation in ['kde', 'pca_kde']:
                    kde = KernelDensity(bandwidth=cluster_bandwidth, kernel='gaussian')
                    kde.fit(cluster_samples)
                    self.cluster_kdes_.append(kde)
                    
                    # 计算样本密度
                    log_dens = kde.score_samples(cluster_samples)
                    dens = np.exp(log_dens)
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = dens
                elif self.density_estimation == 'knn_density':
                    # 使用kNN估计密度，确保k值不超过样本数量
                    actual_k = min(self.knn_k, len(cluster_samples))
                    knn = NearestNeighbors(n_neighbors=actual_k)
                    knn.fit(cluster_samples)
                    distances, _ = knn.kneighbors(cluster_samples)
                    # 密度与最近邻距离成反比
                    dens = 1.0 / (distances[:, -1] + self.epsilon)
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = dens
            else:
                self.cluster_bandwidths_.append(self.bandwidth)
                self.cluster_kdes_.append(None)
    
    def _calculate_box_adjustments(self):
        """计算每个簇的超盒调整因子"""
        # 归一化密度
        if np.max(self.cluster_densities_) - np.min(self.cluster_densities_) > self.epsilon:
            norm_densities = (self.cluster_densities_ - np.min(self.cluster_densities_)) / \
                           (np.max(self.cluster_densities_) - np.min(self.cluster_densities_))
        else:
            norm_densities = np.ones_like(self.cluster_densities_)
        
        # 计算每个簇的平均密度和距离
        self.cluster_box_adjustments_ = np.zeros(self.n_clusters)
        
        for cluster_idx in range(self.n_clusters):
            cluster_mask = self.cluster_labels_ == cluster_idx
            if np.any(cluster_mask):
                avg_density = np.mean(norm_densities[cluster_mask])
                
                # 计算簇中心到所有其他簇中心的距离
                if hasattr(self, 'DSEL_data_pca_'):
                    cluster_center = self.cluster_centers_[cluster_idx]
                else:
                    # 使用原始数据计算簇中心
                    cluster_samples = self.DSEL_data_[cluster_mask]
                    cluster_center = np.mean(cluster_samples, axis=0)
                
                # 计算与其他簇的平均距离
                if self.n_clusters > 1:
                    other_centers = np.delete(self.cluster_centers_, cluster_idx, axis=0)
                    avg_distance = np.mean(cdist(cluster_center.reshape(1, -1), other_centers))
                    # 归一化距离
                    if avg_distance > self.epsilon:
                        norm_distance = avg_distance / np.max(cdist(self.cluster_centers_, self.cluster_centers_))
                    else:
                        norm_distance = 1.0
                else:
                    norm_distance = 1.0
                
                # 综合密度和距离计算调整因子
                # 高密度区域：缩小超盒；低密度区域：扩大超盒
                # 距离其他簇远的区域：扩大超盒；距离其他簇近的区域：缩小超盒
                self.cluster_box_adjustments_[cluster_idx] = \
                    self.density_weight * (1 - avg_density) + \
                    self.cluster_distance_weight * norm_distance
            else:
                self.cluster_box_adjustments_[cluster_idx] = 1.0  # 默认不调整
    
    def _adjust_boxes_by_density(self):
        """基于聚类和密度调整矩形超盒大小"""
        # 检查是否有有效的簇中心
        if not hasattr(self, 'cluster_centers_original_') and not hasattr(self, 'cluster_centers_'):
            return  # 没有聚类中心信息，不进行调整
            
        # 获取原始空间的簇中心
        if hasattr(self, 'cluster_centers_original_'):
            cluster_centers = self.cluster_centers_original_
        else:
            cluster_centers = self.cluster_centers_
        
        # 遍历每个分类器的超盒
        for box in self.HBoxes:
            hboxV = box["Min"]
            hboxW = box["Max"]
            
            # 检查hboxV和hboxW的形状
            if len(hboxV.shape) != 2:
                print(f"警告：超盒形状异常，跳过超盒调整")
                continue
                
            n_boxes, n_features = hboxV.shape
            
            # 确保超盒的特征维度与簇中心的特征维度匹配
            if n_features != cluster_centers.shape[1]:
                print(f"警告：超盒特征维度({n_features}维)与簇中心特征维度({cluster_centers.shape[1]}维)不匹配，跳过超盒调整")
                continue
            
            # 遍历每个超盒
            adjusted_hboxV = []
            adjusted_hboxW = []
            for i in range(n_boxes):
                # 计算单个超盒的中心
                boxV = hboxV[i, :]
                boxW = hboxW[i, :]
                boxC = (boxV + boxW) / 2
                
                try:
                    # 计算距离最近的簇
                    cluster_distances = cdist(boxC.reshape(1, -1), cluster_centers)
                    nearest_cluster = np.argmin(cluster_distances)
                    adjustment_factor = self.cluster_box_adjustments_[nearest_cluster]
                    
                    # 计算原始超盒的半宽
                    half_size = (boxW - boxV) / 2
                    
                    # 应用调整，确保每个维度的大小不小于最小值
                    new_half_size = np.maximum(half_size * adjustment_factor, 1e-8)
                    
                    # 更新超盒的Min和Max
                    new_boxV = boxC - new_half_size
                    new_boxW = boxC + new_half_size
                    
                    adjusted_hboxV.append(new_boxV)
                    adjusted_hboxW.append(new_boxW)
                except Exception as e:
                    print(f"警告：调整超盒时出错，使用原始超盒: {e}")
                    adjusted_hboxV.append(boxV)
                    adjusted_hboxW.append(boxW)
            
            # 更新该分类器的所有超盒
            box["Min"] = np.array(adjusted_hboxV)
            box["Max"] = np.array(adjusted_hboxW)
    
    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        """基于聚类和密度的竞争力估计"""
        # 调用父类的基本竞争力估计
        basic_competences = super(FHDES_JFB_vector_rectangle_clustering_optimized, self).estimate_competence(query, neighbors, distances, predictions)
        
        # 如果没有启用增强功能，直接返回基本竞争力
        if self.enhancement_weight <= 0:
            return basic_competences
        
        # 计算增强竞争力
        enhanced_competences = self._calculate_enhanced_competence(query)
        
        # 合并基本竞争力和增强竞争力
        competences_ = self.basic_weight * basic_competences + self.enhancement_weight * enhanced_competences
        
        # 归一化
        if self.normalize_weights:
            competences_ = preprocessing.MinMaxScaler().fit_transform(competences_)
        
        return competences_
    
    def _calculate_enhanced_competence(self, query):
        """基于聚类和密度计算增强竞争力"""
        n_queries = query.shape[0] if len(query.shape) > 1 else 1
        competences = np.zeros((n_queries, self.n_classifiers_))
        
        # 将查询转换为合适的形状
        if n_queries == 1:
            Xq = query.reshape(1, 1, self.n_features_)
        else:
            Xq = query.reshape(1, n_queries, self.n_features_)
        
        # 对每个分类器计算增强竞争力
        for clsr_idx in range(self.n_classifiers_):
            # 收集当前分类器的所有超盒
            clsr_boxes = [box for box in self.HBoxes if box["clsr"] == clsr_idx]
            
            if not clsr_boxes:
                continue  # 如果没有该分类器的超盒，跳过
            
            # 收集所有超盒的Min和Max
            all_hboxV = np.vstack([box["Min"] for box in clsr_boxes])
            all_hboxW = np.vstack([box["Max"] for box in clsr_boxes])
            
            if len(all_hboxV) == 0:
                continue
            
            # 计算隶属度
            clsrBoxes_m = self.membership_boxes(all_hboxV, all_hboxW, Xq)
            
            # 计算增强竞争力
            if len(all_hboxV) > 1:
                # 使用前两个最高隶属度超盒的加权和
                bb_indexes = np.argsort(-clsrBoxes_m, axis=0)
                b1 = bb_indexes[0, :]
                b2 = bb_indexes[1, :]
                
                for i in range(n_queries):
                    membership = clsrBoxes_m[b1[i], i] * 0.7 + clsrBoxes_m[b2[i], i] * 0.3
                    competences[i, clsr_idx] = membership
            else:
                # 只有一个超盒的情况
                competences[:, clsr_idx] = clsrBoxes_m[0, :]
        
        return competences