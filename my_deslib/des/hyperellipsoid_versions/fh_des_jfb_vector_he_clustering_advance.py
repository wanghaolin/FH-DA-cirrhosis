# coding=utf-8

import numpy as np
from my_deslib.des.hyperellipsoid_versions.fh_des_JFB_vector_he_clustering import FHDES_JFB_vector_he_clustering
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing

# 全局数值稳定常量
EPS = 1e-8

class FHDES_JFB_vector_he_clustering_advance(FHDES_JFB_vector_he_clustering):
    """
    基于 FHDES_JFB_vector_he_clustering 的扩展版：
    - 在聚类 + 局部密度的基础上，增加多项局部稳定性指标（局部准确率、置信度、重叠、hardness、协方差 condition、bootstrap instability）
    - 将密度因子与稳定性因子结合生成 FinalAdj（簇级最终调整因子）
    - 考虑分类器在簇的错分率，生成 classifier_adjustments_，并用其缩放 HBoxes（中心不变）
    - 保持父类接口兼容：fit / estimate_competence / setup_hyperboxs
    """

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
                 gamma=1.0,
                 mis_sample_based=True,
                 doContraction=True,
                 thetaCheck=True,
                 multiCore_process=False,
                 shuffle_dataOrder=False,
                 # 原有聚类和密度估计参数
                 n_clusters=5,
                 bandwidth=1.0,
                 density_weight=0.5,
                 cluster_distance_weight=0.5,
                 reg_lambda=1e-6,
                 # 新增稳定性/控制参数
                 metric_weights=None,
                 overlap_sample_n=1000,
                 bootstrap_B=20,
                 hard_k=5,
                 min_samples_for_cov=10,
                 stab_lambda=0.5,
                 alpha_density=0.5,
                 min_adjustment_factor=0.5,
                 beta_mis=0.5):
        # 调用父类初始化
        super(FHDES_JFB_vector_he_clustering_advance, self).__init__(
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
            gamma=gamma,
            mis_sample_based=mis_sample_based,
            doContraction=doContraction,
            thetaCheck=thetaCheck,
            multiCore_process=multiCore_process,
            shuffle_dataOrder=shuffle_dataOrder,
            n_clusters=n_clusters,
            bandwidth=bandwidth,
            density_weight=density_weight,
            cluster_distance_weight=cluster_distance_weight,
            reg_lambda=reg_lambda
        )

        # 新增参数与默认值
        self.metric_weights = metric_weights if metric_weights is not None else {
            "acc": 0.4, "conf": 0.15, "overlap": 0.2, "hard": 0.15, "cond": 0.1
        }
        self.overlap_sample_n = overlap_sample_n
        self.bootstrap_B = bootstrap_B
        self.hard_k = hard_k
        self.min_samples_for_cov = min_samples_for_cov
        self.stab_lambda = stab_lambda
        self.alpha_density = alpha_density
        self.min_adjustment_factor = min_adjustment_factor
        self.beta_mis = beta_mis

        # 存储聚类和密度相关信息（会在 fit 后填充）
        self.cluster_stability_score_ = None
        self.box_cluster_correlations_cache = None
        self.classifier_adjustments_ = None
        self.cluster_local_acc_ = None
        self.cluster_local_conf_ = None
        self.cluster_hardness_ = None
        self.cluster_condnum_ = None
        self.cluster_overlap_ = None

    # --------------------------
    # fit（入口）
    # --------------------------
    def fit(self, X, y):
        # 先调用父类 fit（父类应构建 DSEL_data_, DSEL_processed_, HBoxes 等）
        super(FHDES_JFB_vector_he_clustering_advance, self).fit(X, y)

        # 确保聚类和密度估计已完成
        if self.cluster_centers_ is None or self.cluster_densities_ is None:
            self._perform_clustering()
            self._estimate_densities()

        # 计算 FinalAdj（密度 + 稳定性）并应用到 HBoxes
        self._calculate_box_adjustments()
        self._adjust_boxes_by_density_and_stability()
        
        # 预计算盒与簇的相关性并应用分类器级调整
        self._precompute_box_cluster_correlations_and_apply()

        return self

    # --------------------------
    # 补丁：计算局部稳定性指标与合成 FinalAdj
    # --------------------------
    def _compute_local_metrics(self):
        """
        计算簇级和簇-分类器级局部指标：
         - self.cluster_local_acc_ (n_clusters, n_classifiers)
         - self.cluster_local_conf_ (n_clusters, n_classifiers) （若没有概率信息则为0）
         - self.cluster_hardness_ (n_clusters)
         - self.cluster_condnum_ (n_clusters)
         - self.cluster_overlap_ (n_clusters)
        这些供后续合成稳定性指标使用。
        """
        X = np.asarray(self.DSEL_data_) if self.DSEL_data_ is not None else np.zeros((0,))
        labels = np.asarray(self.cluster_labels_) if self.cluster_labels_ is not None else np.array([], dtype=int)
        n_clusters = 0 if self.cluster_centers_ is None else int(self.cluster_centers_.shape[0])
        n_cls = int(getattr(self, "n_classifiers_", getattr(self, "n_classifiers", 0)))

        # init
        self.cluster_local_acc_ = np.zeros((n_clusters, n_cls))
        self.cluster_local_conf_ = np.zeros((n_clusters, n_cls))
        self.cluster_hardness_ = np.zeros(n_clusters)
        self.cluster_condnum_ = np.zeros(n_clusters)
        self.cluster_overlap_ = np.zeros(n_clusters)

        # prepare DSEL labels/probs if provided by parent
        DSEL_y = getattr(self, "DSEL_y_", None)  # optional
        DSEL_probs = getattr(self, "DSEL_probs_", None)  # optional: shape (n_samples, n_classifiers)

        # per-cluster indices
        cluster_indices = [np.where(labels == c)[0] for c in range(n_clusters)]

        for c in range(n_clusters):
            idxs = cluster_indices[c]
            if idxs.size == 0:
                # default values if no samples
                self.cluster_local_acc_[c, :] = 0.0
                self.cluster_local_conf_[c, :] = 0.0
                self.cluster_hardness_[c] = 0.5
                self.cluster_condnum_[c] = 1.0
                self.cluster_overlap_[c] = 0.0
                continue

            # local accuracy: from self.DSEL_processed_ (True means correct)
            if hasattr(self, "DSEL_processed_") and self.DSEL_processed_ is not None:
                cluster_processed = self.DSEL_processed_[idxs, :]  # shape (n_in_cluster, n_classifiers)
                accs = np.sum(cluster_processed, axis=0) / (idxs.size + EPS)
                self.cluster_local_acc_[c, :] = accs
            else:
                self.cluster_local_acc_[c, :] = 0.0

            # local confidence: if DSEL_probs provided and aligned
            if DSEL_probs is not None:
                try:
                    probs = np.asarray(DSEL_probs)[idxs, :]
                    self.cluster_local_conf_[c, :] = np.mean(probs, axis=0)
                except Exception:
                    self.cluster_local_conf_[c, :] = 0.0
            else:
                self.cluster_local_conf_[c, :] = 0.0

            # hardness: kNN-based (1 - fraction same label among k neighbors)
            if DSEL_y is not None and idxs.size > 1:
                y_local = np.asarray(DSEL_y)[idxs]
                k = min(self.hard_k, max(1, idxs.size - 1))
                try:
                    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X[idxs])
                    dists, neigh = nbrs.kneighbors(X[idxs])
                    same_frac = []
                    for i_row, neigh_idx in enumerate(neigh):
                        neigh_idx_no_self = neigh_idx[1:]
                        same_frac.append(np.mean(y_local[neigh_idx_no_self] == y_local[i_row]))
                    same_frac = np.array(same_frac)
                    hard = 1.0 - np.mean(same_frac)
                except Exception:
                    hard = 0.5
            else:
                hard = 0.5
            self.cluster_hardness_[c] = hard

            # covariance condition number
            try:
                cov = np.cov(X[idxs].T) if X.ndim == 2 and idxs.size > 1 else np.array([[1.0]])
                cond = np.linalg.cond(cov) if cov.size else 1.0
            except Exception:
                cond = 1.0
            # clamp cond to a reasonable large number to avoid inf
            if not np.isfinite(cond) or cond <= 0:
                cond = 1.0
            self.cluster_condnum_[c] = cond

            # overlap estimate (sample-based)
            try:
                self.cluster_overlap_[c] = self._estimate_overlap_for_cluster(cluster_idx=c, idxs=idxs, n_samples=self.overlap_sample_n)
            except Exception:
                self.cluster_overlap_[c] = 0.0

        return

    def _estimate_overlap_for_cluster(self, cluster_idx, idxs=None, n_samples=500):
        """
        采样式估计簇局部的超椭球体重叠度。
        返回重叠比例（0..1），越大表示重叠越严重。
        """
        X = np.asarray(self.DSEL_data_) if self.DSEL_data_ is not None else np.zeros((0,))
        if idxs is None:
            if self.cluster_labels_ is None:
                return 0.0
            idxs = np.where(self.cluster_labels_ == cluster_idx)[0]
        if idxs.size == 0:
            return 0.0

        pts = X[idxs]
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        margin = 0.1 * (maxs - mins + EPS)
        low = mins - margin
        high = maxs + margin

        # cap sampling budget
        n_samples = int(min(n_samples, 2000))
        rng = np.random.RandomState(0)
        samples = rng.uniform(low=low, high=high, size=(n_samples, X.shape[1]))

        # 收集候选椭球体
        ellipsoids = []
        for box in self.HBoxes:
            if "Center" in box and "Cov" in box:
                ellipsoids.append((box["Center"], box["Cov"]))
            elif "Center" in box and "Covariance" in box:
                ellipsoids.append((box["Center"], box["Covariance"]))

        if len(ellipsoids) == 0:
            return 0.0

        # 找出离簇中心最近的椭球体
        cluster_center = np.mean(pts, axis=0)
        dists = []
        for center, _ in ellipsoids:
            dists.append(np.linalg.norm(center - cluster_center))
        dists = np.array(dists)
        K_box = min(len(ellipsoids), 200)
        nearest_idx = np.argsort(dists)[:K_box]
        nearest_ellipsoids = [ellipsoids[i] for i in nearest_idx]

        # 检查点是否在椭球体内
        overlap_count = 0
        for sample in samples:
            inside_count = 0
            for center, cov in nearest_ellipsoids:
                # 计算马氏距离的平方
                diff = sample - center
                try:
                    inv_cov = np.linalg.inv(cov)
                    mahalanobis_sq = np.dot(np.dot(diff, inv_cov), diff)
                    # 如果马氏距离平方小于等于1，则在椭球体内
                    if mahalanobis_sq <= 1.0:
                        inside_count += 1
                except np.linalg.LinAlgError:
                    continue
            
            if inside_count >= 2:
                overlap_count += 1

        overlap_frac = float(overlap_count) / n_samples
        return overlap_frac

    def _bootstrap_center_stability(self, cluster_idx, idxs, B=20):
        """
        基于簇点的 bootstrap 估计簇中心稳定性（取 0..1 范围，不稳定越接近 1）。
        """
        X = np.asarray(self.DSEL_data_) if self.DSEL_data_ is not None else np.zeros((0,))
        if idxs.size <= 1:
            return 0.0
        pts = X[idxs]
        rng = np.random.RandomState(0)
        centers = []
        for b in range(B):
            sel = rng.choice(np.arange(len(pts)), size=len(pts), replace=True)
            centers.append(np.mean(pts[sel], axis=0))
        centers = np.asarray(centers)
        var_centers = np.mean(np.var(centers, axis=0))
        total_var = np.mean(np.var(pts, axis=0)) + EPS
        instability = var_centers / (total_var + EPS)
        return float(np.clip(instability, 0.0, 1.0))

    def _calculate_box_adjustments(self):
        """
        新版：在已有的 cluster_densities_ 基础上，结合新增稳定性指标，计算簇级 FinalAdj（self.cluster_box_adjustments_）。
        """
        # prepare cluster indices
        if self.cluster_centers_ is None:
            self.cluster_box_adjustments_ = np.array([])
            return

        n_clusters = int(self.cluster_centers_.shape[0])
        labels = np.asarray(self.cluster_labels_)
        cluster_indices = [np.where(labels == c)[0] for c in range(n_clusters)]

        # compute per-cluster average normalized density rho_bar
        rho_bar = np.zeros(n_clusters)
        if self.cluster_densities_ is None or len(self.cluster_densities_) == 0:
            rho_bar = np.zeros(n_clusters)
        else:
            for c in range(n_clusters):
                idxs = cluster_indices[c]
                if idxs.size == 0:
                    rho_bar[c] = 0.0
                else:
                    rho_bar[c] = float(np.mean(self.cluster_densities_[idxs]))

        # compute local metrics used for stability
        self._compute_local_metrics()

        # prepare metrics arrays
        acc_mean = np.mean(self.cluster_local_acc_, axis=1) if getattr(self, "cluster_local_acc_", None) is not None else np.zeros(n_clusters)
        conf_mean = np.mean(self.cluster_local_conf_, axis=1) if getattr(self, "cluster_local_conf_", None) is not None else np.zeros(n_clusters)
        overlap = getattr(self, "cluster_overlap_", np.zeros(n_clusters))
        hard = getattr(self, "cluster_hardness_", np.zeros(n_clusters))
        condnum = getattr(self, "cluster_condnum_", np.zeros(n_clusters))

        # bootstrap instability per cluster
        instability = np.zeros(n_clusters)
        for c in range(n_clusters):
            idxs = cluster_indices[c]
            instability[c] = self._bootstrap_center_stability(c, idxs, B=self.bootstrap_B) if idxs.size > 0 else 0.0

        # min-max normalization helper
        def minmax(arr):
            mn = np.min(arr); mx = np.max(arr)
            if mx - mn < EPS:
                return np.ones_like(arr) * 0.5
            return (arr - mn) / (mx - mn + EPS)

        acc_n = minmax(acc_mean)
        conf_n = minmax(conf_mean)
        overlap_n = minmax(overlap)
        hard_n = minmax(hard)
        cond_n = minmax(condnum)
        instability_n = minmax(instability)

        w = self.metric_weights
        # Compose stability score: positive (acc/conf) minus negative (overlap/hard/cond) and adjusted by instability
        Stab = (w.get("acc", 0.4) * acc_n +
                w.get("conf", 0.15) * conf_n -
                w.get("overlap", 0.2) * overlap_n -
                w.get("hard", 0.15) * hard_n -
                w.get("cond", 0.1) * cond_n)
        Stab = Stab - 0.2 * instability_n
        Stab = minmax(Stab)
        self.cluster_stability_score_ = Stab  # higher => more稳定

        # combine density and stability to FinalAdj
        alpha = getattr(self, "alpha_density", 0.5)
        lamb = getattr(self, "stab_lambda", 0.5)
        min_adj = getattr(self, "min_adjustment_factor", 0.5)

        FinalAdj = np.clip(1.0 - alpha * rho_bar * (1.0 - lamb * Stab), min_adj, 1.0)
        self.cluster_box_adjustments_ = FinalAdj
        return

    # --------------------------
    # 预计算相关性并应用 classifier-level adjustments
    # --------------------------
    def _precompute_box_cluster_correlations_and_apply(self):
        """
        1) 计算 box -> cluster 相关性矩阵 self.box_cluster_correlations_cache (n_classifiers, n_clusters)
        2) 基于 self.cluster_box_adjustments_ 与每簇的 mis_frac 计算 classifier_adjustments_
        3) 将 classifier_adjustments_ 应用到 self.HBoxes（按类缩放）
        """
        if self.cluster_centers_ is None or len(self.cluster_centers_) == 0:
            self.box_cluster_correlations_cache = np.zeros((getattr(self, "n_classifiers_", 0), 0))
            return

        centers = np.asarray(self.cluster_centers_)
        n_clusters = centers.shape[0]
        n_cls = int(getattr(self, "n_classifiers_", getattr(self, "n_classifiers", 0)))
        correlations = np.zeros((n_cls, n_clusters))

        # gather sub-box centers grouped by classifier
        for cl in range(n_cls):
            sub_centers = []
            for box in self.HBoxes:
                if box["clsr"] != cl:
                    continue
                if "Center" in box:
                    center = box["Center"]
                    if isinstance(center, np.ndarray) and center.ndim == 1:
                        sub_centers.append(center)
            
            if len(sub_centers) == 0:
                correlations[cl, :] = 0.0
                continue
            
            sub_centers = np.asarray(sub_centers)
            d = cdist(sub_centers, centers)
            max_per_row = np.max(d, axis=1, keepdims=True) + EPS
            sim = 1.0 - (d / max_per_row)
            agg = sim.mean(axis=0)
            correlations[cl, :] = agg

        self.box_cluster_correlations_cache = correlations

        # compute classifier-level adjustments based on cluster adjustments and correlations
        if hasattr(self, "cluster_box_adjustments_"):
            adj = np.zeros(n_cls)
            beta = getattr(self, "beta_mis", 0.5)
            
            for cl in range(n_cls):
                # 计算该分类器在各簇中的错分率
                if hasattr(self, "cluster_local_acc_"):
                    mis_frac = 1.0 - np.mean(self.cluster_local_acc_[cl, :]) if self.cluster_local_acc_.shape[0] > cl else 0.5
                else:
                    mis_frac = 0.5
                
                # 基于相关性加权的簇调整因子
                weighted_adj = np.sum(correlations[cl, :] * self.cluster_box_adjustments_) / (np.sum(correlations[cl, :]) + EPS)
                
                # 结合错分率调整（错分率越高，调整因子越大）
                cl_adj = weighted_adj * (1.0 + beta * mis_frac)
                adj[cl] = np.clip(cl_adj, getattr(self, "min_adjustment_factor", 0.5), 2.0)
            
            self.classifier_adjustments_ = adj
            
            # 应用分类器级调整到 HBoxes
            for box in self.HBoxes:
                cl = box["clsr"]
                if cl < len(adj):
                    factor = adj[cl]
                    # 缩放协方差矩阵（中心不变）
                    if "Cov" in box:
                        box["Cov"] = box["Cov"] * (factor ** 2)
                    if "Covariance" in box:
                        box["Covariance"] = box["Covariance"] * (factor ** 2)

    def _adjust_boxes_by_density_and_stability(self):
        """
        基于聚类、密度和稳定性信息调整超椭球体参数
        """
        if self.cluster_centers_ is None or self.cluster_box_adjustments_ is None:
            return

        X_dsel = self.DSEL_data_
        
        # 对每个分类器的超椭球体进行调整
        for clsr_idx in range(len(self.HBoxes)):
            box = self.HBoxes[clsr_idx]
            classifier_idx = box["clsr"]
            
            # 获取椭球体参数
            if "Center" in box and ("Cov" in box or "Covariance" in box):
                center = box["Center"]
                cov = box.get("Cov", box.get("Covariance"))
                
                # 找到离椭球体中心最近的簇
                distances_to_centers = cdist([center], self.cluster_centers_)[0]
                nearest_cluster = np.argmin(distances_to_centers)
                
                # 使用该簇的调整因子调整椭球体大小（缩放协方差矩阵）
                adjustment_factor = self.cluster_box_adjustments_[nearest_cluster]
                
                # 缩放协方差矩阵
                scaled_cov = cov * (adjustment_factor ** 2)
                
                # 更新椭球体参数
                if "Cov" in box:
                    box["Cov"] = scaled_cov
                if "Covariance" in box:
                    box["Covariance"] = scaled_cov

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        # 调用父类的estimate_competence获取基本隶属度
        basic_competences = super(FHDES_JFB_vector_he_clustering_advance, self).estimate_competence(
            query, neighbors, distances, predictions
        )
        
        # 如果没有进行聚类（可能是因为没有数据），直接返回基本隶属度
        if self.cluster_centers_ is None or self.box_cluster_correlations_cache is None:
            return basic_competences
        
        # 计算基于聚类、密度和稳定性的增强隶属度
        enhanced_competences = self._calculate_enhanced_competence(query, basic_competences)
        
        return enhanced_competences
    
    def _calculate_enhanced_competence(self, query, basic_competences):
        """
        结合聚类、密度和稳定性信息计算增强的隶属度
        """
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
            
            # 结合距离、密度和稳定性计算查询点的簇隶属度
            cluster_memberships = (self.cluster_distance_weight * normalized_distances + 
                                   self.density_weight * normalized_densities)
            
            # 如果有稳定性分数，将其整合到簇隶属度中
            if self.cluster_stability_score_ is not None:
                stability_factor = 0.7 + 0.3 * self.cluster_stability_score_
                cluster_memberships *= stability_factor
            
            # 对每个分类器，计算其椭球体与查询点的簇隶属度的相关性
            for clsr_idx in range(self.n_classifiers_):
                # 获取当前分类器的簇相关性
                if clsr_idx < self.box_cluster_correlations_cache.shape[0]:
                    box_cluster_correlations = self.box_cluster_correlations_cache[clsr_idx]
                else:
                    box_cluster_correlations = np.zeros(self.n_clusters)
                
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
    
    def setup_hyperboxs(self, classifier):
        # 调用父类的setup_hyperboxs方法构建基础超盒
        return super(FHDES_JFB_vector_he_clustering_advance, self).setup_hyperboxs(classifier)