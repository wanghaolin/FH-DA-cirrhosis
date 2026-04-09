# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.des_FHMW_JFB_vector import DESFHMW_JFB_vector
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity, NearestNeighbors
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import multiprocessing
from sklearn.utils import shuffle

class DESFHMW_JFB_vector_clustering_optimized(DESFHMW_JFB_vector):
    
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
                 bandwidth_selection='fixed',  # 'fixed', 'silverman', 'cv', 'adaptive'
                 density_estimation='kde',  # 'kde', 'knn_density', 'pca_kde'
                 density_weight=0.5,
                 cluster_distance_weight=0.5,
                 basic_weight=0.7,  # 基本竞争力权重
                 enhancement_weight=0.3,  # 增强竞争力权重
                 knn_k=5):  # kNN密度估计的k值
        
        # 调用父类初始化
        super(DESFHMW_JFB_vector_clustering_optimized, self).__init__(
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
        self.bandwidth_selection = bandwidth_selection  # 带宽选择方法
        self.density_estimation = density_estimation  # 密度估计方法
        self.density_weight = density_weight  # 密度权重
        self.cluster_distance_weight = cluster_distance_weight  # 距离权重
        self.basic_weight = basic_weight  # 基本竞争力权重
        self.enhancement_weight = enhancement_weight  # 增强竞争力权重
        self.knn_k = knn_k  # kNN密度估计的k值
        
        # 规范化权重，确保和为1
        total_weight = density_weight + cluster_distance_weight
        if total_weight > 0:
            self.density_weight = density_weight / total_weight
            self.cluster_distance_weight = cluster_distance_weight / total_weight
        else:
            self.density_weight = 0.5
            self.cluster_distance_weight = 0.5
        
        # 规范化基本权重和增强权重
        total_basic_enhance = basic_weight + enhancement_weight
        if total_basic_enhance > 0:
            self.basic_weight = basic_weight / total_basic_enhance
            self.enhancement_weight = enhancement_weight / total_basic_enhance
        else:
            self.basic_weight = 0.7
            self.enhancement_weight = 0.3
        
        # 存储聚类和密度相关信息
        self.cluster_centers_ = None  # 簇中心
        self.cluster_kdes_ = None     # 每个簇的KDE模型
        self.cluster_labels_ = None   # 训练样本的簇标签
        self.cluster_densities_ = None # 每个样本的局部密度
        self.cluster_box_adjustments_ = None # 每个簇的超盒调整因子
    
    def fit(self, X, y):
        # 调用父类fit方法
        super(DESFHMW_JFB_vector_clustering_optimized, self).fit(X, y)
        
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
        """对每个簇执行密度估计"""
        X_dsel = self.DSEL_data_
        self.cluster_kdes_ = []
        self.cluster_densities_ = np.zeros(len(X_dsel))
        self.cluster_pcas_ = []  # 存储PCA模型（如果使用）
        
        # 对每个簇分别进行密度估计
        for cluster_idx in range(self.n_clusters):
            # 获取该簇的样本
            cluster_samples = X_dsel[self.cluster_labels_ == cluster_idx]
            
            if len(cluster_samples) > 0:
                # 处理高维数据 - 降维或使用kNN密度
                if self.density_estimation == 'pca_kde' and X_dsel.shape[1] > 10:
                    # 使用PCA降维
                    pca = PCA(n_components=min(10, X_dsel.shape[1]-1), random_state=self.random_state)
                    pca.fit(cluster_samples)
                    self.cluster_pcas_.append(pca)
                    transformed_samples = pca.transform(cluster_samples)
                    
                    # 选择带宽
                    bandwidth = self._select_bandwidth(transformed_samples, cluster_idx)
                    
                    # 创建KDE模型
                    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
                    kde.fit(transformed_samples)
                    self.cluster_kdes_.append(kde)
                    
                    # 计算密度
                    log_density = kde.score_samples(transformed_samples)
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = np.exp(log_density)
                elif self.density_estimation == 'knn_density':
                    # 使用kNN密度估计
                    self.cluster_kdes_.append(None)  # KDE模型设为None
                    self.cluster_pcas_.append(None)  # PCA模型设为None
                    
                    # 计算kNN密度
                    nbrs = NearestNeighbors(n_neighbors=self.knn_k, algorithm='auto')
                    nbrs.fit(cluster_samples)
                    distances, _ = nbrs.kneighbors(cluster_samples)
                    
                    # 计算基于k近邻距离的密度（距离倒数）
                    # 添加小常数避免除零
                    knn_density = 1.0 / (distances[:, -1] + 1e-8)
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = knn_density
                else:
                    # 标准KDE
                    self.cluster_pcas_.append(None)
                    
                    # 选择带宽
                    bandwidth = self._select_bandwidth(cluster_samples, cluster_idx)
                    
                    # 创建KDE模型
                    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
                    kde.fit(cluster_samples)
                    self.cluster_kdes_.append(kde)
                    
                    # 计算密度
                    log_density = kde.score_samples(cluster_samples)
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = np.exp(log_density)
            else:
                # 如果簇为空，添加空模型
                self.cluster_kdes_.append(None)
                self.cluster_pcas_.append(None)
    
    def _select_bandwidth(self, samples, cluster_idx):
        """根据选择的方法确定带宽"""
        if self.bandwidth_selection == 'fixed':
            return self.bandwidth
        elif self.bandwidth_selection == 'silverman':
            # Silverman法则
            n, d = samples.shape
            if n > 0 and d > 0:
                # 计算样本标准差
                std = np.std(samples, axis=0).mean()
                # Silverman公式
                return std * (n * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
            return self.bandwidth
        elif self.bandwidth_selection == 'cv':
            # 交叉验证选择带宽
            try:
                # 确保样本数至少为2，cv至少为2
                if len(samples) < 2:
                    return self.bandwidth
                param_grid = {'bandwidth': np.logspace(-1, 1, 10)}
                cv_value = max(2, min(5, len(samples)))
                grid = GridSearchCV(KernelDensity(), param_grid, cv=cv_value)
                grid.fit(samples)
                return grid.best_params_['bandwidth']
            except:
                # 如果交叉验证失败，回退到fixed带宽
                return self.bandwidth
        elif self.bandwidth_selection == 'adaptive':
            # 基于簇大小的自适应带宽
            n = len(samples)
            # 簇样本越多，带宽可以越大
            return self.bandwidth * (1 + np.log(n + 1) / 10)
        else:
            return self.bandwidth
    
    def _calculate_box_adjustments(self):
        """计算每个簇的超盒调整因子"""
        # 归一化密度值
        density_min, density_max = np.min(self.cluster_densities_), np.max(self.cluster_densities_)
        if density_max - density_min > 1e-8:  # 添加小常数避免除零
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
        basic_competences = super(DESFHMW_JFB_vector_clustering_optimized, self).estimate_competence(
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
            max_distance = np.max(distances_to_centers)
            if max_distance > 1e-8:  # 添加小常数避免除零
                normalized_distances = 1.0 - (distances_to_centers / max_distance)
            else:
                normalized_distances = np.ones_like(distances_to_centers)
            
            # 计算查询点在每个簇中的密度
            query_densities = np.zeros(self.n_clusters)
            for cluster_idx in range(self.n_clusters):
                if self.cluster_kdes_[cluster_idx] is not None:
                    # 检查是否需要使用PCA转换
                    if len(self.cluster_pcas_) > cluster_idx and self.cluster_pcas_[cluster_idx] is not None:
                        # 使用PCA转换查询点
                        transformed_query = self.cluster_pcas_[cluster_idx].transform(query_point)
                        query_densities[cluster_idx] = np.exp(self.cluster_kdes_[cluster_idx].score_samples(transformed_query))
                    else:
                        query_densities[cluster_idx] = np.exp(self.cluster_kdes_[cluster_idx].score_samples(query_point))
                elif self.density_estimation == 'knn_density':
                    # 对于kNN密度估计，我们不计算查询点的密度
                    # 而是使用簇中心距离作为替代
                    pass
            
            # 归一化密度
            max_density = np.max(query_densities)
            if max_density > 1e-8:  # 添加小常数避免除零
                normalized_densities = query_densities / max_density
            else:
                normalized_densities = np.ones_like(query_densities)
            
            # 结合距离和密度计算查询点的簇隶属度
            # 使用规范化的权重
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
                total_boxes = 0
                
                for box in clsr_boxes:
                    hboxV = box["Min"]
                    hboxW = box["Max"]
                    
                    for box_idx in range(len(hboxV)):
                        # 超盒中心
                        box_center = (hboxV[box_idx] + hboxW[box_idx]) / 2
                        
                        # 计算超盒中心到各簇中心的距离
                        box_distances = cdist([box_center], self.cluster_centers_)[0]
                        max_box_distance = np.max(box_distances)
                        if max_box_distance > 1e-8:  # 添加小常数避免除零
                            box_normalized_distances = 1.0 - (box_distances / max_box_distance)
                        else:
                            box_normalized_distances = np.ones_like(box_distances)
                        
                        # 累加到簇相关性
                        box_cluster_correlations += box_normalized_distances
                        total_boxes += 1
                
                # 除以超盒总数，防止超盒数量偏置
                if total_boxes > 0:
                    box_cluster_correlations /= total_boxes
                
                # 归一化簇相关性
                correlation_sum = np.sum(box_cluster_correlations)
                if correlation_sum > 1e-8:  # 添加小常数避免除零
                    box_cluster_correlations /= correlation_sum
                
                # 计算分类器对查询点的增强竞争力
                clsr_enhancement = np.sum(cluster_memberships * box_cluster_correlations)
                
                # 结合基本竞争力和增强因子，使用规范化的权重
                enhanced_competences[i, clsr_idx] = (basic_competences[i, clsr_idx] * self.basic_weight + 
                                                    clsr_enhancement * self.enhancement_weight)
        
        # 归一化增强竞争力
        scaler = preprocessing.MinMaxScaler()
        for i in range(n_queries):
            enhanced_competences[i] = scaler.fit_transform(enhanced_competences[i].reshape(-1, 1)).flatten()
        
        return enhanced_competences
    
    def FMM_train(self, classifier_ind):
        # 调用父类的FMM_train方法构建基础超盒
        hboxV, hboxW = super(DESFHMW_JFB_vector_clustering_optimized, self).FMM_train(classifier_ind)
        
        # 注意：超盒的调整在_adjust_boxes_by_density方法中进行，这里保持父类的实现不变
        
        return hboxV, hboxW