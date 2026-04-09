# coding=utf-8

import numpy as np
from my_deslib.des.fh_des_JFB_vector import FHDES_JFB_vector
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity, NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist
import sklearn.preprocessing as preprocessing

# 全局数值稳定常量
EPS = 1e-8


class FHDES_JFB_vector_clustering_advance(FHDES_JFB_vector):
    """
    基于原始 FHDES_JFB_vector 的扩展版：
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
                 mis_sample_based=True,
                 doContraction=True,
                 thetaCheck=True,
                 multiCore_process=False,
                 shuffle_dataOrder=False,
                 # 新增聚类和密度估计参数（保留原有）
                 n_clusters=5,
                 bandwidth=1.0,
                 density_weight=0.5,
                 cluster_distance_weight=0.5,
                 # 新增稳定性/控制参数（补丁用）
                 metric_weights=None,      # 词典：权重
                 overlap_sample_n=1000,
                 bootstrap_B=20,
                 hard_k=5,
                 min_samples_for_cov=10,
                 stab_lambda=0.5,
                 alpha_density=0.5,
                 min_adjustment_factor=0.5,
                 beta_mis=0.5):
        # 调用父类初始化
        super(FHDES_JFB_vector_clustering_advance, self).__init__(
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

        # 原有参数
        self.n_clusters = n_clusters  # 聚类数量
        self.bandwidth = bandwidth    # KDE 带宽
        self.density_weight = density_weight
        self.cluster_distance_weight = cluster_distance_weight

        # 补丁新增参数与默认值
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
        self.cluster_centers_ = None
        self.cluster_kdes_ = None
        self.cluster_labels_ = None
        self.cluster_densities_ = None
        self.cluster_box_adjustments_ = None  # 原始密度驱动的簇因子（被 FinalAdj 覆盖）
        # 补丁新增成员：
        self.cluster_stability_score_ = None
        self.box_cluster_correlations_cache = None
        self.classifier_adjustments_ = None

    # --------------------------
    # fit（入口）
    # --------------------------
    def fit(self, X, y):
        # 先调用父类 fit（父类应构建 DSEL_data_, DSEL_processed_, HBoxes 等）
        super(FHDES_JFB_vector_clustering_advance, self).fit(X, y)

        # 执行聚类和密度估计
        self._perform_clustering()
        self._estimate_densities()

        # 计算 FinalAdj（密度 + 稳定性）并应用到 HBoxes
        self._calculate_box_adjustments()
        self._adjust_boxes_by_density()

        return self

    # --------------------------
    # 聚类与密度（保留原逻辑，可按需扩展）
    # --------------------------
    def _perform_clustering(self):
        """使用 KMeans 对 DSEL_data_ 进行聚类（保持原逻辑）。"""
        X_dsel = self.DSEL_data_
        # 如果 DSEL_data_ 为空则跳过
        if X_dsel is None or len(X_dsel) == 0:
            self.cluster_labels_ = np.array([], dtype=int)
            self.cluster_centers_ = np.array([])
            return

        k = max(1, min(int(self.n_clusters), X_dsel.shape[0]))
        kmeans = KMeans(n_clusters=k, random_state=self.random_state)
        self.cluster_labels_ = kmeans.fit_predict(X_dsel)
        self.cluster_centers_ = kmeans.cluster_centers_

    def _estimate_densities(self):
        """对每个簇执行核密度估计（原实现），并生成 self.cluster_densities_（per-sample）。"""
        X_dsel = self.DSEL_data_
        n = 0 if X_dsel is None else X_dsel.shape[0]
        self.cluster_kdes_ = []
        self.cluster_densities_ = np.zeros(n)

        if X_dsel is None or n == 0:
            return

        # 对每个簇做 KDE（若簇为空则 None）
        # 集中使用 self.n_clusters (如果聚类后生成的簇数少于该值，取 cluster_centers_ 的实际数目)
        try:
            n_clusters_actual = self.cluster_centers_.shape[0]
        except Exception:
            n_clusters_actual = int(self.n_clusters)

        for cluster_idx in range(n_clusters_actual):
            cluster_samples = X_dsel[self.cluster_labels_ == cluster_idx]
            if cluster_samples.shape[0] > 0:
                try:
                    kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
                    kde.fit(cluster_samples)
                    self.cluster_kdes_.append(kde)
                    log_density = kde.score_samples(cluster_samples)
                    # 取 exp(log_density)，然后 normalize 会在后续步骤做
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = np.exp(log_density - np.max(log_density))
                except Exception:
                    # KDE 失败时 fallback to ones
                    self.cluster_kdes_.append(None)
                    self.cluster_densities_[self.cluster_labels_ == cluster_idx] = 1.0
            else:
                self.cluster_kdes_.append(None)

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
        采样式估计簇局部的超盒重叠度（使用 axis-aligned boxes 判断）。
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

        # helper: test points inside boxes
        def point_in_box(samples_arr, Min, Max):
            Min = np.asarray(Min)
            Max = np.asarray(Max)
            if Min.ndim == 1:
                Min = Min.reshape(1, -1)
                Max = Max.reshape(1, -1)
            inside = np.all((samples_arr[:, None, :] >= Min[None, :, :] - EPS) & (samples_arr[:, None, :] <= Max[None, :, :] + EPS), axis=2)
            return inside  # (n_samples, n_boxes)

        # collect candidate boxes (centers) but limit to nearest boxes to the cluster center
        box_centers = []
        box_mins = []
        box_maxs = []
        for box in self.HBoxes:
            Min = np.asarray(box["Min"])
            Max = np.asarray(box["Max"])
            if Min.ndim == 1:
                c = (Min + Max) / 2.0
                box_centers.append(c)
                box_mins.append(Min)
                box_maxs.append(Max)
            else:
                sc = (Min + Max) / 2.0
                for r in range(sc.shape[0]):
                    box_centers.append(sc[r])
                    box_mins.append(Min[r])
                    box_maxs.append(Max[r])

        if len(box_centers) == 0:
            return 0.0

        box_centers = np.asarray(box_centers)
        cluster_center = np.mean(pts, axis=0)
        dists = cdist(box_centers, cluster_center.reshape(1, -1)).flatten()
        K_box = min(len(box_centers), 200)
        nearest_idx = np.argsort(dists)[:K_box]
        mins_keep = [box_mins[i] for i in nearest_idx]
        maxs_keep = [box_maxs[i] for i in nearest_idx]

        inside = point_in_box(samples, np.asarray(mins_keep), np.asarray(maxs_keep))
        counts = np.sum(inside, axis=1)
        overlap_frac = float(np.mean(counts >= 2))
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

        # gather sub-box centers grouped by classifier (like earlier)
        for cl in range(n_cls):
            sub_centers = []
            for box in self.HBoxes:
                if box["clsr"] != cl:
                    continue
                Min = np.asarray(box["Min"])
                Max = np.asarray(box["Max"])
                if Min.ndim == 1:
                    Min = Min.reshape(1, -1)
                    Max = Max.reshape(1, -1)
                sc = (Min + Max) / 2.0
                for r in range(sc.shape[0]):
                    sub_centers.append(sc[r])
            if len(sub_centers) == 0:
                correlations[cl, :] = 0.0
                continue
            sub_centers = np.asarray(sub_centers)
            d = cdist(sub_centers, centers)
            max_per_row = np.max(d, axis=1, keepdims=True) + EPS
            sim = 1.0 - (d / max_per_row)
            agg = sim.mean(axis=0)
            if agg.sum() > EPS:
                agg = agg / (agg.sum() + EPS)
            correlations[cl, :] = agg
        self.box_cluster_correlations_cache = correlations

        # mis_frac per classifier per cluster
        labels = self.cluster_labels_
        mis_matrix = (~self.DSEL_processed_) if getattr(self, "mis_sample_based", True) else self.DSEL_processed_
        mis_frac = np.zeros((n_cls, n_clusters))
        for c in range(n_clusters):
            idxs = np.where(labels == c)[0]
            if idxs.size == 0:
                mis_frac[:, c] = 0.0
                continue
            mis_frac[:, c] = np.sum(mis_matrix[idxs, :], axis=0) / (idxs.size + EPS)

        # compute classifier-level adjustments
        beta = getattr(self, "beta_mis", 0.5)
        FinalAdj = self.cluster_box_adjustments_
        cls_adj = np.zeros(n_cls)
        for cl in range(n_cls):
            corr = correlations[cl, :]
            if corr.sum() <= EPS:
                cls_adj[cl] = 1.0
                continue
            weighted = corr * (FinalAdj * (1.0 - beta * mis_frac[cl, :]))
            val = np.sum(weighted)
            cls_adj[cl] = float(np.clip(val, self.min_adjustment_factor, 1.0))
        self.classifier_adjustments_ = cls_adj

        # apply cls_adj to HBoxes (scale boxes sizes, keep centers)
        for box in self.HBoxes:
            cl = box["clsr"]
            adj = cls_adj[cl]
            Min = np.asarray(box["Min"])
            Max = np.asarray(box["Max"])
            if Min.ndim == 1:
                Min = Min.reshape(1, -1)
                Max = Max.reshape(1, -1)
            centers = (Min + Max) / 2.0
            sizes = (Max - Min) * adj
            new_min = centers - sizes / 2.0
            new_max = centers + sizes / 2.0
            box["Min"] = new_min
            box["Max"] = new_max

        return

    def _adjust_boxes_by_density(self):
        """
        在计算好 cluster_box_adjustments_ 之后，预计算相关性并应用 classifier-level adjustments。
        旧版调整逻辑已被替换为 _precompute_box_cluster_correlations_and_apply。
        """
        # ensure cluster_box_adjustments_ 存在
        if getattr(self, "cluster_box_adjustments_", None) is None:
            self._calculate_box_adjustments()
        self._precompute_box_cluster_correlations_and_apply()
        return

    # --------------------------
    # 预测时增强 competences（保留原逻辑）
    # --------------------------
    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):
        basic_competences = super(FHDES_JFB_vector_clustering_advance, self).estimate_competence(
            query, neighbors, distances, predictions
        )

        # 如果没有进行聚类（可能是因为没有数据），直接返回基本隶属度
        if self.cluster_centers_ is None or getattr(self, "box_cluster_correlations_cache", None) is None:
            return basic_competences

        # 计算基于聚类和密度的增强隶属度（向量化版本）
        enhanced_competences = self._calculate_enhanced_competence(query, basic_competences)

        return enhanced_competences

    def _calculate_enhanced_competence(self, query, basic_competences):
        """
        向量化实现：计算 cluster_memberships（使用已计算的 cluster_kdes_ 或距离代理），
        enhancement = cluster_memberships.dot( box_cluster_correlations_cache.T )
        final = basic * alpha + enhancement * (1-alpha) ; each query 行归一化。
        """
        Xq = np.array(query).reshape(-1, self.n_features_)
        nq = Xq.shape[0]
        n_clusters = 0 if self.cluster_centers_ is None else int(self.cluster_centers_.shape[0])
        # compute cluster_memberships (nq, n_clusters)
        # If KDE available, use it; else approximate via negative distance
        cluster_memberships = np.zeros((nq, n_clusters))
        if n_clusters == 0:
            return basic_competences

        # compute log-likelihood per cluster for queries if KDE exists
        log_liks = np.full((nq, n_clusters), -1e12)
        for c in range(n_clusters):
            kde = self.cluster_kdes_[c] if (self.cluster_kdes_ is not None and c < len(self.cluster_kdes_)) else None
            if kde is not None:
                try:
                    logp = kde.score_samples(Xq)
                    log_liks[:, c] = logp - np.max(logp)  # numeric shift
                except Exception:
                    # fallback to negative distance
                    center = self.cluster_centers_[c].reshape(1, -1)
                    d = cdist(Xq, center).flatten()
                    scale = np.mean(d) + EPS
                    log_liks[:, c] = -d / (scale + EPS)
            else:
                center = self.cluster_centers_[c].reshape(1, -1)
                d = cdist(Xq, center).flatten()
                scale = np.mean(d) + EPS
                log_liks[:, c] = -d / (scale + EPS)

        # combine with cluster prior (empirical)
        priors = np.zeros(n_clusters)
        for c in range(n_clusters):
            priors[c] = np.sum(self.cluster_labels_ == c) + EPS
        priors = priors / priors.sum()
        log_post = log_liks + np.log(priors + EPS)
        max_row = np.max(log_post, axis=1, keepdims=True)
        expv = np.exp(log_post - max_row)
        cluster_memberships = expv / (np.sum(expv, axis=1, keepdims=True) + EPS)  # (nq, n_clusters)

        # compute enhancement via cached correlations
        corr_t = self.box_cluster_correlations_cache.T if self.box_cluster_correlations_cache is not None else np.zeros((n_clusters, int(getattr(self, "n_classifiers_", 0))))
        enhancement = cluster_memberships.dot(corr_t)  # (nq, n_classifiers)

        # combine with basic
        alpha = getattr(self, "enhancement_mix_basic", 0.7) if hasattr(self, "enhancement_mix_basic") else 0.7
        enhanced = basic_competences * alpha + enhancement * (1.0 - alpha)

        # per-query row-wise MinMax normalization
        scaler = MinMaxScaler()
        for i in range(enhanced.shape[0]):
            try:
                enhanced[i] = scaler.fit_transform(enhanced[i].reshape(-1, 1)).flatten()
            except Exception:
                pass
        return enhanced

    # --------------------------
    # 保持与父类一致的 hyperbox setup
    # --------------------------
    def setup_hyperboxs(self, classifier):
        hboxV, hboxW = super(FHDES_JFB_vector_clustering_advance, self).setup_hyperboxs(classifier)
        return hboxV, hboxW
