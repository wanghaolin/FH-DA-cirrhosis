# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class FHDES_AllBoxes_vector_rectangle(BaseDES):

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
                 shuffle_dataOrder = False):
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

        ############### it should be based on Clustering #############################
        super(FHDES_AllBoxes_vector_rectangle, self).__init__(pool_classifiers=pool_classifiers,
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

    def fit(self, X, y):
        super(FHDES_AllBoxes_vector_rectangle, self).fit(X, y)
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

    def estimate_competence(self, query, neighbors=None, distances=None, predictions=None):

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

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(highest_mems) - highest_mems
        else:
            competences_ = highest_mems

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_