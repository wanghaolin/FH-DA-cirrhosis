# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.des_FHMW_AllBoxes_vector import DESFHMW_allboxes_vector
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from my_deslib.des.base import BaseDES
from sklearn.utils import shuffle

class DESFHMW_allboxes_vector_sphere(BaseDES):

    def __init__(self, pool_classifiers=None,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30,
                 random_state=None,
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
        # 先定义类特定的参数
        self.theta = theta
        self.mu = mu
        self.sigma = sigma  # 高斯隶属函数的标准差参数
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []  # 确保HBoxes被初始化为空列表
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder

        # 调用BaseDES父类初始化，使用正确的参数
        super(DESFHMW_allboxes_vector_sphere, self).__init__(pool_classifiers=pool_classifiers,
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
    
    def contract_boxBased(self, bC, bR, coboxC, coboxR):
        """基于冲突球体收缩当前球体"""
        # 对于每个冲突球体，检查是否有重叠
        for i in range(len(coboxC)):
            # 计算两个球心之间的距离
            center_dist = np.linalg.norm(bC - coboxC[i])
            # 检查是否重叠
            if center_dist < (bR + coboxR[i]):
                # 重叠情况下，调整半径以避免重叠
                # 将当前球体的半径减小到刚好不与冲突球体重叠
                bR = max(0, center_dist - coboxR[i] - 1e-6)
        return bR

    def update_boxes(self, hboxC, hboxR, coboxC, coboxR, missClsSample):
        if missClsSample:  # miss classified sample
            nboxC = hboxC
            nboxR = hboxR
            pboxC = coboxC
            pboxR = coboxR

        else:
            pboxC = hboxC
            pboxR = hboxR
            nboxC = coboxC
            nboxR = coboxR
        return nboxC, nboxR, pboxC, pboxR

    def FMM_train(self, classifier_ind):
        #        print(np.size(samples_ind))
        nboxC = np.zeros((1, self.n_features_)) - 1  # np.array()
        nboxR = np.zeros((1,)) - 1

        pboxC = np.zeros((1, self.n_features_)) - 1  # np.array()
        pboxR = np.zeros((1,)) - 1

        X = self.DSEL_data_
        y = self.DSEL_processed_[:,classifier_ind]

        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            X, y = shuffle(X, y, random_state=classifier_ind)

        for ind, x in enumerate(X):
            missClassified = y[ind]==False
            ############## Type: Miss or Correct Classified ############
            if missClassified: # miss classified sample
                hboxC = nboxC
                hboxR = nboxR
                coboxC = pboxC
                coboxR = pboxR

            else:
                hboxC = pboxC
                hboxR = pboxR
                coboxC = nboxC
                coboxR = nboxR

            #############################################################
            # Creation first box
            if hboxC[0,0] == -1:
                hboxC[0, :] = x
                hboxR[0] = 0
                nboxC, nboxR, pboxC, pboxR = self.update_boxes(hboxC, hboxR, coboxC, coboxR, missClassified)
                continue

            # X is in a box?
            is_inSphere = False
            for sphereInd in range(len(hboxC)):
                if self.is_inside(hboxC, hboxR, sphereInd, x):
                    is_inSphere = True
                    break
            if is_inSphere:
                # nop
                continue
            ######################## Expand ############################
            # Finding nearest box
            distances = np.linalg.norm(x - hboxC, axis=1)
            expanded = False
            sorted_indexes = np.argsort(distances)[::-1]
            for ind_sphere in sorted_indexes:
                if self.is_expandable(hboxC, hboxR, ind_sphere, x):
                    self.expand_sphere(hboxC, hboxR, ind_sphere, x)
                    # 收缩以避免与冲突球体重叠
                    if len(coboxC) > 0 and coboxC[0,0] != -1:
                        hboxR[ind_sphere] = self.contract_boxBased(hboxC[ind_sphere], hboxR[ind_sphere], coboxC, coboxR)
                    expanded = True
                    break

            ######################## Creation ############################
            if expanded ==False:
                xt = x.reshape(1,self.n_features_)
                rt = np.array([0])
                hboxC, hboxR = self.add_spheres(hboxC, hboxR, bC=xt, bR=rt)

            nboxC, nboxR, pboxC, pboxR = self.update_boxes(hboxC, hboxR, coboxC, coboxR, missClassified)

        if self.mis_sample_based:
            return nboxC, nboxR
        else:
            return pboxC, pboxR

    def fit(self, X, y):
        super(DESFHMW_allboxes_vector_sphere, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bC, bR] = self.FMM_train(classifier_index)
                class_dic =  { "clsr" : classifier_index, "Center" : bC, "Radius" : bR }
                self.NO_hypeboxes += len(bC)
                self.HBoxes.append(class_dic)
        else:
            # 修改进程数计算方式：不超过CPU核心数且最多4个进程
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

        for clsr in range(len(self.HBoxes)):
            # c_range = range( indices[k], indices[k] + count[k])
            hboxC = self.HBoxes[clsr]["Center"]
            hboxR = self.HBoxes[clsr]["Radius"]

            clsrSpheres_m = self.membership_spheres(hboxC, hboxR, Xq)
            if len(hboxC) > 1:
                #bb_indexes = np.argpartition(-clsrSpheres_m, kth=2, axis=0)[:2]
                bb_indexes = np.argsort(-clsrSpheres_m, axis=0)
                b1 = bb_indexes[0,:]
                b2 = bb_indexes[1,:]
                for i in range(0,len(query)):
                    if clsrSpheres_m[b1[i],i]==1 : # if the query sample is located inside or near to the box
                        highest_mems[i,int(clsr)] = 1
                    else:
                        highest_mems[i,int(clsr)] = clsrSpheres_m[b1[i],i] *0.7 + clsrSpheres_m[b2[i],i]*0.3

            else:  # In case that we have only one hyperbox for the classifier
                for i in range(0, len(query)):
                    highest_mems[i, int(clsr)] = clsrSpheres_m[0, i]

        #### was mistake ####
        if self.mis_sample_based:
            competences_ = np.max(highest_mems) - highest_mems
            # competences_ = np.sqrt(self.n_features_)  - competences_
        else:
            competences_ = highest_mems

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_