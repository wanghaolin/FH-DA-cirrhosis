# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
from my_deslib.des.base import BaseDES
from my_deslib.util.fuzzy_hyperbox import Hyperbox
import sklearn.preprocessing as preprocessing
from my_deslib.util.instance_hardness import *
import multiprocessing
from sklearn.utils import shuffle

class FHDES_JFB_vector(BaseDES):

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
                 mis_sample_based = True,
                 doContraction = True,
                 thetaCheck = True,
                 multiCore_process = False,
                 shuffle_dataOrder = False):
        self.theta = theta
        self.mu = mu
        self.mis_sample_based = mis_sample_based
        self.HBoxes = []
        self.NO_hypeboxes = 0
        self.doContraction = doContraction
        self.thetaCheck = thetaCheck
        self.multiCore_process = multiCore_process
        self.shuffle_dataOrder = shuffle_dataOrder

        ############### it should be based on Clustering #############################
        super(FHDES_JFB_vector, self).__init__(pool_classifiers=pool_classifiers,
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
        hboxV[boxInd] = np.minimum(hboxV[boxInd],x)
        hboxW[boxInd] = np.maximum(hboxW[boxInd], x)
    def is_expandable(self,hboxV, hboxW, boxInd, x):
        candV = np.minimum(hboxV[boxInd], x)
        candW = np.maximum(hboxW[boxInd], x)
        return all((candW-candV) < self.theta)
    def is_inside(self,hboxV, hboxW, boxInd,x):
        return np.all(hboxV[boxInd] < x) and np.all(hboxW[boxInd] > x)
    def membership_boxes(self, hboxV, hboxW, Xq):
        NO_hypeboxes, n_features = hboxV.shape
        hboxC = np.add(hboxV, hboxW) / 2
        boxes_W = hboxW.reshape(NO_hypeboxes, 1, n_features)
        boxes_V = hboxV.reshape(NO_hypeboxes, 1, n_features)
        boxes_center = hboxC.reshape(NO_hypeboxes, 1, n_features)
        halfsize = ((boxes_W - boxes_V) / 2).reshape(NO_hypeboxes, 1, n_features)
        d = np.abs(boxes_center - Xq) - halfsize
        d[d < 0] = 0
        dd = np.linalg.norm(d, axis=2)
        dd = dd / np.sqrt(self.n_features_)
        m = 1 - dd  # m: membership
        m =  np.power(m,6)
        return m
    def will_exceed_samples(self, hboxV, hboxW, boxInd,x, con_samples):
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


    def fit(self, X, y):
        super(FHDES_JFB_vector, self).fit(X, y)
        if self.mu > 1 or self.mu <= 0:
            raise Exception("The value of Mu must be between 0 and 1.")
        if self.theta > 1 or self.theta <= 0:
            raise Exception("The value of Theta must be between 0 and 1.")

        if self.multiCore_process == False:
            for classifier_index in range(self.n_classifiers_):
                [bV,bW] = self.setup_hyperboxs(classifier_index)
                class_dic =  { "clsr" : classifier_index, "Min" : bV, "Max" : bW }
                self.NO_hypeboxes += len(bV)
                self.HBoxes.append(class_dic)
        else:
            # (改)修改进程数计算方式：不超过CPU核心数且最多4个进程
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

        Xq = query.reshape(1,len(query),self.n_features_)
        
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
            # competences_ = np.sqrt(self.n_features_)  - competences_
        else:
            competences_ = highest_mems

        scaler = preprocessing.MinMaxScaler()
        competences_ = scaler.fit_transform(competences_)

        return competences_

    def setup_hyperboxs(self, classifier ):
        if np.size(classifier) < 0:
            pass

        if(self.mis_sample_based):
            samples_ind = ~self.DSEL_processed_[:, classifier]
            Contraction_ind = self.DSEL_processed_[:, classifier]
        else:
            samples_ind = self.DSEL_processed_[:, classifier]
            Contraction_ind = ~self.DSEL_processed_[:, classifier]

        hboxV = np.zeros((1,self.n_features_)) - 1#np.array()
        hboxW =np.zeros((1,self.n_features_)) - 1

        selected_samples = self.DSEL_data_[samples_ind, :]

        contraction_samples = self.DSEL_data_[Contraction_ind,:]
        ############################################################# Shuffle
        if self.shuffle_dataOrder:
            selected_samples = shuffle(selected_samples,random_state = classifier)

        for ind, X in enumerate(selected_samples):
            # Creation first box
            if hboxV[0,0] == -1:
                hboxV[0, :] = X
                hboxW[0, :] = X
                continue

            # X is in a box?
            is_inBox = False
            for boxInd in range(len(hboxV)):
                if self.is_inside(hboxV, hboxW, boxInd,X):
                    is_inBox = True
                    break
            if is_inBox:
                # nop
                continue

            ######################## Expand ############################
            # Sort boxes by the distances
            hboxC = (hboxV + hboxW) / 2
            expanded = False
            box_list = np.linalg.norm(X-hboxC,axis=1)

            nearestBox_ind = np.argmin(box_list)
            if self.thetaCheck and self.doContraction:
                if self.is_expandable(hboxV, hboxW, nearestBox_ind, X):
                    self.expand_box(hboxV,hboxW,nearestBox_ind,X)
                    self.contract_samplesBased(hboxV,hboxW,nearestBox_ind,contraction_samples)
                    # expanded = True
                    continue


            elif self.thetaCheck and not self.doContraction:
                if self.is_expandable(hboxV, hboxW, nearestBox_ind, X):
                    self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                    # expanded = True
                    continue

            elif not self.thetaCheck and self.doContraction:
                if not self.will_exceed_samples(hboxV, hboxW, nearestBox_ind, X, contraction_samples):
                    self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                    expanded = True
                    continue
            ######################## Creation ############################
            xt = X.reshape(1,self.n_features_)
            hboxV, hboxW = self.add_boxes(hboxV, hboxW, bV=xt, bW=xt)



        return hboxV, hboxW

    def select(self, competences):

        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        max_value = np.max(competences, axis=1)
        selected_classifiers = (
                competences >= self.mu * max_value.reshape(competences.shape[0], -1))

        return selected_classifiers

    def visualize_hyperboxes(self, classifier_idx=None, feature_indices=None, ax=None, title=None, show=True):
        """
        可视化指定分类器的超盒和相关样本
        
        参数:
        classifier_idx: int或None, 要可视化的分类器索引，如果为None则可视化所有分类器
        feature_indices: tuple(int, int)或None, 要可视化的两个特征索引，如果为None则使用前两个特征
        ax: matplotlib轴对象，用于绘制，如果为None则创建新的图
        title: str, 图表标题
        show: bool, 是否显示图表，默认为True
        
        返回:
        figure: matplotlib图表对象
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        
        # 检查是否已经拟合数据
        if not hasattr(self, 'HBoxes') or len(self.HBoxes) == 0:
            print("警告: 模型尚未拟合数据或没有生成超盒")
            return None
        
        # 检查是否有DSEL数据
        if not hasattr(self, 'DSEL_data_'):
            print("警告: 没有找到DSEL_data_属性")
            # 只绘制超盒
            if ax is None:
                fig, ax = plt.subplots(figsize=(10, 8))
            
            # 设置默认特征索引
            if feature_indices is None:
                feature_indices = (0, 1)
            
            # 获取要可视化的分类器超盒
            if classifier_idx is not None:
                boxes_to_visualize = [box for box in self.HBoxes if box["clsr"] == classifier_idx]
            else:
                boxes_to_visualize = self.HBoxes
            
            # 可视化超盒
            for box in boxes_to_visualize:
                hboxV = box["Min"]
                hboxW = box["Max"]
                
                # 可视化当前分类器的超盒
                for i in range(len(hboxV)):
                    # 绘制超盒
                    width = hboxW[i, feature_indices[1]] - hboxV[i, feature_indices[1]]
                    height = hboxW[i, feature_indices[0]] - hboxV[i, feature_indices[0]]
                    
                    rect = Rectangle((hboxV[i, feature_indices[1]], hboxV[i, feature_indices[0]]), 
                                    width, height, fill=False, edgecolor='green', 
                                    linewidth=1.5, alpha=0.7)
                    ax.add_patch(rect)
            
            # 设置图表属性
            ax.set_xlabel(f'特征 {feature_indices[1]}')
            ax.set_ylabel(f'特征 {feature_indices[0]}')
            
            if title is None:
                if classifier_idx is not None:
                    title = f'分类器 {classifier_idx} 的超盒可视化'
                else:
                    title = '所有分类器的超盒可视化'
            
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            if show:
                plt.show()
            
            return ax.figure
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        # 设置默认特征索引
        if feature_indices is None:
            feature_indices = (0, 1)
        
        # 获取要可视化的分类器超盒
        if classifier_idx is not None:
            boxes_to_visualize = [box for box in self.HBoxes if box["clsr"] == classifier_idx]
        else:
            boxes_to_visualize = self.HBoxes
        
        # 可视化超盒
        for box in boxes_to_visualize:
            clsr = box["clsr"]
            hboxV = box["Min"]
            hboxW = box["Max"]
            
            # 获取当前分类器的样本
            if self.mis_sample_based:
                # 对于错误样本模式，显示正确分类的样本（用于收缩）
                correct_inds = self.DSEL_processed_[:, clsr]
                correct_samples = self.DSEL_data_[correct_inds, :]
            else:
                # 对于正确样本模式，显示错误分类的样本（用于收缩）
                incorrect_inds = ~self.DSEL_processed_[:, clsr]
                incorrect_samples = self.DSEL_data_[incorrect_inds, :]
                
            # 可视化当前分类器的超盒
            for i in range(len(hboxV)):
                # 绘制超盒
                width = hboxW[i, feature_indices[1]] - hboxV[i, feature_indices[1]]
                height = hboxW[i, feature_indices[0]] - hboxV[i, feature_indices[0]]
                
                rect = Rectangle((hboxV[i, feature_indices[1]], hboxV[i, feature_indices[0]]), 
                                width, height, fill=False, edgecolor='green', 
                                linewidth=1.5, alpha=0.7)
                ax.add_patch(rect)
            
            # 绘制相关样本
            if self.mis_sample_based:
                if len(correct_samples) > 0:
                    ax.scatter(correct_samples[:, feature_indices[1]], 
                              correct_samples[:, feature_indices[0]], 
                              c='red', marker='o', s=30, alpha=0.6, label='错误样本')
            else:
                if len(incorrect_samples) > 0:
                    ax.scatter(incorrect_samples[:, feature_indices[1]], 
                              incorrect_samples[:, feature_indices[0]], 
                              c='red', marker='o', s=30, alpha=0.6, label='错误样本')
            
            # 绘制用于构建超盒的样本
            if self.mis_sample_based:
                # 错误样本模式下，超盒基于正确分类的样本
                box_inds = ~self.DSEL_processed_[:, clsr]
            else:
                # 正确样本模式下，超盒基于错误分类的样本
                box_inds = self.DSEL_processed_[:, clsr]
                
            box_samples = self.DSEL_data_[box_inds, :]
            if len(box_samples) > 0:
                ax.scatter(box_samples[:, feature_indices[1]], 
                          box_samples[:, feature_indices[0]], 
                          c='green', marker='o', s=30, alpha=0.6, label='正确样本')
        
        # 设置图表属性
        ax.set_xlabel(f'特征 {feature_indices[1]}')
        ax.set_ylabel(f'特征 {feature_indices[0]}')
        
        if title is None:
            if classifier_idx is not None:
                title = f'分类器 {classifier_idx} 的超盒可视化'
            else:
                title = '所有分类器的超盒可视化'
        
        ax.set_title(title)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 添加图例
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        
        if show:
            plt.show()
            
        return ax.figure
    
    def visualize_hyperbox_building(self, classifier_idx, feature_indices=None, steps=10, save_path=None, show=False):
        """
        可视化超盒的构建过程
        
        参数:
        classifier_idx: int, 要可视化的分类器索引
        feature_indices: tuple(int, int)或None, 要可视化的两个特征索引，如果为None则使用前两个特征
        steps: int, 要展示的构建步骤数
        save_path: str或None, 如果提供，则保存图片的路径
        show: bool, 是否在每一步显示图表，默认为False
        
        返回:
        figures: 图表列表
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        import os
        
        # 检查是否已经拟合数据
        if not hasattr(self, 'DSEL_data_') or not hasattr(self, 'DSEL_processed_'):
            print("警告: 模型尚未拟合数据，无法可视化构建过程")
            return []
        
        # 设置默认特征索引
        if feature_indices is None:
            feature_indices = (0, 1)
        
        # 获取用于构建超盒的样本
        if self.mis_sample_based:
            samples_ind = ~self.DSEL_processed_[:, classifier_idx]
            Contraction_ind = self.DSEL_processed_[:, classifier_idx]
        else:
            samples_ind = self.DSEL_processed_[:, classifier_idx]
            Contraction_ind = ~self.DSEL_processed_[:, classifier_idx]
        
        selected_samples = self.DSEL_data_[samples_ind, :]
        contraction_samples = self.DSEL_data_[Contraction_ind, :]
        
        # 复制setup_hyperboxs方法的逻辑，但添加可视化步骤
        hboxV = np.zeros((1, self.n_features_)) - 1
        hboxW = np.zeros((1, self.n_features_)) - 1
        
        # 计算需要展示的步骤
        total_samples = len(selected_samples)
        if total_samples == 0:
            print(f"警告: 分类器 {classifier_idx} 没有用于构建超盒的样本")
            return []
        
        if steps > total_samples:
            steps = total_samples
        step_interval = max(1, total_samples // steps)
        
        # 创建用于保存所有图的列表
        figures = []
        
        for ind, X in enumerate(selected_samples):
            # 第一步：创建第一个超盒
            if hboxV[0, 0] == -1:
                hboxV[0, :] = X
                hboxW[0, :] = X
            else:
                # 检查样本是否已经在超盒内
                is_inBox = False
                for boxInd in range(len(hboxV)):
                    if self.is_inside(hboxV, hboxW, boxInd, X):
                        is_inBox = True
                        break
                
                if not is_inBox:
                    # 尝试扩展最近的超盒
                    hboxC = (hboxV + hboxW) / 2
                    box_list = np.linalg.norm(X - hboxC, axis=1)
                    nearestBox_ind = np.argmin(box_list)
                    
                    expanded = False
                    if self.thetaCheck and self.doContraction:
                        if self.is_expandable(hboxV, hboxW, nearestBox_ind, X):
                            self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                            self.contract_samplesBased(hboxV, hboxW, nearestBox_ind, contraction_samples)
                            expanded = True
                    elif self.thetaCheck and not self.doContraction:
                        if self.is_expandable(hboxV, hboxW, nearestBox_ind, X):
                            self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                            expanded = True
                    elif not self.thetaCheck and self.doContraction:
                        if not self.will_exceed_samples(hboxV, hboxW, nearestBox_ind, X, contraction_samples):
                            self.expand_box(hboxV, hboxW, nearestBox_ind, X)
                            expanded = True
                    
                    # 如果没有扩展现有超盒，则创建新超盒
                    if not expanded:
                        xt = X.reshape(1, self.n_features_)
                        hboxV, hboxW = self.add_boxes(hboxV, hboxW, bV=xt, bW=xt)
            
            # 可视化当前步骤
            if ind % step_interval == 0 or ind == total_samples - 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # 绘制当前所有超盒
                for i in range(len(hboxV)):
                    width = hboxW[i, feature_indices[1]] - hboxV[i, feature_indices[1]]
                    height = hboxW[i, feature_indices[0]] - hboxV[i, feature_indices[0]]
                    
                    rect = Rectangle((hboxV[i, feature_indices[1]], hboxV[i, feature_indices[0]]), 
                                    width, height, fill=False, edgecolor='green', 
                                    linewidth=1.5, alpha=0.7)
                    ax.add_patch(rect)
                
                # 绘制已处理的样本
                ax.scatter(selected_samples[:ind+1, feature_indices[1]], 
                          selected_samples[:ind+1, feature_indices[0]], 
                          c='green', marker='o', s=30, alpha=0.6, label='已处理样本')
                
                # 绘制未处理的样本
                if ind < total_samples - 1:
                    ax.scatter(selected_samples[ind+1:, feature_indices[1]], 
                              selected_samples[ind+1:, feature_indices[0]], 
                              c='gray', marker='o', s=30, alpha=0.3, label='未处理样本')
                
                # 绘制收缩样本
                if len(contraction_samples) > 0:
                    ax.scatter(contraction_samples[:, feature_indices[1]], 
                              contraction_samples[:, feature_indices[0]], 
                              c='red', marker='o', s=30, alpha=0.6, label='收缩样本')
                
                # 设置图表属性
                ax.set_xlabel(f'特征 {feature_indices[1]}')
                ax.set_ylabel(f'特征 {feature_indices[0]}')
                ax.set_title(f'分类器 {classifier_idx} 超盒构建过程 - 步骤 {ind+1}/{total_samples}')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                plt.tight_layout()
                
                if show:
                    plt.show()
                
                figures.append((fig, f'step_{ind+1}'))
        
        # 保存所有图片
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            for fig, name in figures:
                fig.savefig(os.path.join(save_path, f'{name}.png'))
                plt.close(fig)
        
        return figures

#
