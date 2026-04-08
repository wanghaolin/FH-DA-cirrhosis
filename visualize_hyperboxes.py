# coding=utf-8
"""
可视化四种FH_DES超盒的Python脚本
支持超矩形、超椭球体、超球体和双斜率超矩形的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
import matplotlib.cm as cm

# Set default font
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Fix negative sign display issue

class HyperboxVisualizer:
    """
    超盒可视化类，支持四种不同形状的超盒可视化
    """
    
    def __init__(self, figsize=(12, 10)):
        """
        初始化可视化器
        
        参数:
        figsize: tuple - 图形大小
        """
        self.figsize = figsize
        
    def draw_hyperrectangle(self, ax, center, width, height, color='blue', alpha=0.3, label=None):
        """
        绘制超矩形（投影到二维）
        
        参数:
        ax: matplotlib轴对象
        center: tuple - 中心坐标 (x, y)
        width: float - 宽度
        height: float - 高度
        color: str - 颜色
        alpha: float - 透明度
        label: str - 标签
        """
        x, y = center
        rect = Rectangle((x - width/2, y - height/2), width, height, 
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=2, label=label)
        ax.add_patch(rect)
        
    def draw_hypersphere(self, ax, center, radius, color='green', alpha=0.3, label=None):
        """
        绘制超球体（投影到二维为圆形）
        
        参数:
        ax: matplotlib轴对象
        center: tuple - 中心坐标 (x, y)
        radius: float - 半径
        color: str - 颜色
        alpha: float - 透明度
        label: str - 标签
        """
        circle = plt.Circle(center, radius, 
                           facecolor=color, alpha=alpha, edgecolor=color, linewidth=2, label=label)
        ax.add_patch(circle)
        
    def draw_hyperellipsoid(self, ax, center, cov_matrix, color='red', alpha=0.3, label=None, scale=1.0):
        """
        绘制超椭球体（投影到二维）
        
        参数:
        ax: matplotlib轴对象
        center: tuple - 中心坐标 (x, y)
        cov_matrix: 2x2 numpy数组 - 协方差矩阵
        color: str - 颜色
        alpha: float - 透明度
        label: str - 标签
        scale: float - 缩放因子（控制椭球大小）
        """
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 排序特征值和特征向量
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        # 计算半轴长度
        width, height = 2 * scale * np.sqrt(eigenvalues)
        
        # 计算旋转角度
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        
        # 绘制椭球
        ellipse = Ellipse(center, width, height, angle=angle, 
                         facecolor=color, alpha=alpha, edgecolor=color, linewidth=2, label=label)
        ax.add_patch(ellipse)
    
    def draw_double_slope_rectangle(self, ax, center, width, height, alpha_param=0.5, beta_param=2.0, 
                                   color='purple', alpha=0.3, label=None):
        """
        绘制双斜率超矩形（投影到二维）
        
        参数:
        ax: matplotlib轴对象
        center: tuple - 中心坐标 (x, y)
        width: float - 宽度
        height: float - 高度
        alpha_param: float - 内部区域斜率参数
        beta_param: float - 外部区域斜率参数
        color: str - 颜色
        alpha: float - 透明度
        label: str - 标签
        """
        # 绘制内部矩形
        self.draw_hyperrectangle(ax, center, width, height, color, alpha, label)
        
        # 绘制外部影响区域（虚线表示）
        x, y = center
        outer_width = width * (1 + 1/beta_param)
        outer_height = height * (1 + 1/beta_param)
        rect = Rectangle((x - outer_width/2, y - outer_height/2), outer_width, outer_height, 
                         facecolor='none', edgecolor=color, linewidth=1, linestyle='--', alpha=0.5)
        ax.add_patch(rect)
    
    def visualize_all_hyperboxes(self, examples=5):
        """
        可视化所有四种超盒类型
        
        参数:
        examples: int - 每种超盒的示例数量
        """
        # 创建随机数据
        np.random.seed(42)  # 设置随机种子以确保可重复性
        
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('四种FH_DES超盒可视化', fontsize=16)
        
        # 莫兰迪色系颜色映射
        colors = [
            '#B2BEDD',  # 莫兰迪蓝
            '#D4CBB7',  # 莫兰迪棕
            '#C1D0C4',  # 莫兰迪绿
            '#DCC0CE',  # 莫兰迪粉
            '#D6D6D0'   # 莫兰迪灰
        ]
        
        # 1. 超矩形可视化
        ax = axs[0, 0]
        ax.set_title('Rectangular hyperbox')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        
        # 为超矩形分配不重叠的区域
        for i in range(examples):
            # 使用网格布局确保不重叠
            row = i // 2
            col = i % 2
            center = np.array([3 + col * 6, 3 + row * 6])
            width = 2.5
            height = 2.5
            self.draw_hyperrectangle(ax, center, width, height, color=colors[i], alpha=0.4,
                                    label=f'矩形 {i+1}' if i == 0 else None)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 2. 超球体可视化
        ax = axs[0, 1]
        ax.set_title('Spherical hyperbox')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        
        # 为超球体分配不重叠的区域
        for i in range(examples):
            # 使用网格布局确保不重叠
            row = i // 2
            col = i % 2
            center = np.array([3 + col * 6, 3 + row * 6])
            radius = 1.2
            self.draw_hypersphere(ax, center, radius, color=colors[i], alpha=0.4,
                                 label=f'球体 {i+1}' if i == 0 else None)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 3. 超椭球体可视化
        ax = axs[1, 0]
        ax.set_title('Ellipsoidal hyperbox')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        
        # 为超椭球体分配不重叠的区域
        for i in range(examples):
            # 使用网格布局确保不重叠
            row = i // 2
            col = i % 2
            center = np.array([3 + col * 6, 3 + row * 6])
            # 创建随机协方差矩阵
            cov = np.array([[0.8, 0.3], [0.3, 0.5]]) * 2
            self.draw_hyperellipsoid(ax, center, cov, color=colors[i], alpha=0.4,
                                    label=f'椭球体 {i+1}' if i == 0 else None, scale=1.5)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 4. 双斜率超矩形可视化
        ax = axs[1, 1]
        ax.set_title('Bi-slope rectangular hyperbox')
        ax.set_xlabel('特征1')
        ax.set_ylabel('特征2')
        
        # 为双斜率超矩形分配不重叠的区域
        for i in range(examples):
            # 使用网格布局确保不重叠
            row = i // 2
            col = i % 2
            center = np.array([3 + col * 6, 3 + row * 6])
            width = 2.5
            height = 2.5
            alpha_param = 0.5
            beta_param = 2.0
            self.draw_double_slope_rectangle(ax, center, width, height, alpha_param, beta_param,
                                            color=colors[i], alpha=0.4,
                                            label=f'双斜率矩形 {i+1}' if i == 0 else None)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axs
    
    def visualize_overlap(self):
        """
        可视化四种超盒类型的重叠及两个样本点的隶属度对比
        """
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), gridspec_kw={'width_ratios': [1, 0.4]})
        fig.suptitle('四种FH_DES超盒重叠及样本隶属度对比可视化', fontsize=18)
        
        # 统一的中心点位置
        center = (5, 5)
        
        # 模拟两个新样本点
        sample1 = (3, 4)  # 第一个样本点
        sample2 = (6.5, 7)  # 第二个样本点
        samples = [sample1, sample2]
        sample_labels = ['样本1', '样本2']
        sample_colors = ['#E8B79C', '#A8C2CB']  # 莫兰迪橙和莫兰迪蓝灰
        
        # 超盒参数
        rect_width, rect_height = 4, 4
        sphere_radius = 2
        cov_matrix = np.array([[3, 1.5], [1.5, 2]])
        dslope_width, dslope_height = 4, 4
        
        # 更新超盒颜色为更明显的莫兰迪色组合
        hyperbox_colors = ['#8EA3D1', '#8DBF9F', '#E89898', '#D4B996']  # 更深更明显的莫兰迪色
        hyperbox_types = ['超矩形', '超球体', '超椭球体', '双斜率超矩形']
        
        # 绘制四种超盒（使用半透明但更明显的颜色以显示重叠效果）
        self.draw_hyperrectangle(ax1, center, width=rect_width, height=rect_height, color=hyperbox_colors[0], alpha=0.35, label='超矩形')
        self.draw_hypersphere(ax1, center, radius=sphere_radius, color=hyperbox_colors[1], alpha=0.35, label='超球体')
        self.draw_hyperellipsoid(ax1, center, cov_matrix, color=hyperbox_colors[2], alpha=0.35, label='超椭球体', scale=1.5)
        self.draw_double_slope_rectangle(ax1, center, width=dslope_width, height=dslope_height, color=hyperbox_colors[3], alpha=0.35, label='双斜率超矩形')
        
        # 用渐变色表示总体隶属度衰减（所有超盒的隶属度最大值）
        x_range = np.linspace(1, 9, 50)
        y_range = np.linspace(1, 9, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # 计算每个网格点的最大隶属度（来自四种超盒）
        memberships = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                
                # 计算四种超盒的隶属度
                m_rect = self.hyperrectangle_membership(point, np.array(center), rect_width, rect_height)
                m_sphere = self.hypersphere_membership(point, np.array(center), sphere_radius, sigma=0.5)
                m_ellipsoid = self.hyperellipsoid_membership(point, np.array(center), cov_matrix)
                m_dslope = self.double_slope_membership(point, np.array(center), dslope_width, dslope_height)
                
                # 取最大值作为总体隶属度
                memberships[i, j] = max(m_rect, m_sphere, m_ellipsoid, m_dslope)
        
        # 绘制渐变色热图
        im = ax1.imshow(memberships, extent=[1, 9, 1, 9], origin='lower', cmap='YlGnBu', alpha=0.4)
        
        # 添加颜色条
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='最大隶属度')
        
        # 添加中心点标记
        ax1.plot(center[0], center[1], 'o', color='black', markersize=10, markerfacecolor='white', linewidth=2, label='中心点')
        ax1.text(center[0]+0.2, center[1]+0.2, '中心点', fontsize=11, color='black', bbox=dict(facecolor='white', alpha=0.8))
        
        # 计算并存储两个样本点的隶属度
        sample_memberships = []
        
        for i, (sample, label, color) in enumerate(zip(samples, sample_labels, sample_colors)):
            # 使用更明显的样本点标记
            ax1.plot(sample[0], sample[1], 'o', color=color, markersize=12, markerfacecolor='white', linewidth=3, label=label)
            ax1.plot(sample[0], sample[1], 'x', color=color, markersize=10, mew=3)
            
            # 计算并显示样本点到中心点的距离
            distance = np.sqrt((sample[0] - center[0])**2 + (sample[1] - center[1])**2)
            ax1.text(sample[0] + 0.2, sample[1] + 0.2, f'd={distance:.2f}', 
                    fontsize=11, color=color, bbox=dict(facecolor='white', alpha=0.8))
            
            # 计算并显示样本点到四种超盒的隶属度
            point = np.array(sample)
            m_rect = self.hyperrectangle_membership(point, np.array(center), rect_width, rect_height)
            m_sphere = self.hypersphere_membership(point, np.array(center), sphere_radius, sigma=0.5)
            m_ellipsoid = self.hyperellipsoid_membership(point, np.array(center), cov_matrix)
            m_dslope = self.double_slope_membership(point, np.array(center), dslope_width, dslope_height)
            
            memberships_sample = [m_rect, m_sphere, m_ellipsoid, m_dslope]
            sample_memberships.append(memberships_sample)
            
            # 绘制从中心点指向样本点的箭头，显示样本到中心点的距离方向
            ax1.annotate('', xy=(sample[0], sample[1]), 
                       xytext=(center[0], center[1]), 
                       arrowprops=dict(arrowstyle='->', color=color, linewidth=2, alpha=0.8))
        
        # 在右侧子图中绘制柱状图展示隶属度对比
        ax2.set_title('样本隶属度对比', fontsize=14)
        ax2.set_xlabel('超盒类型')
        ax2.set_ylabel('隶属度值')
        
        bar_width = 0.35
        x = np.arange(len(hyperbox_types))
        
        # 绘制两个样本点的隶属度柱状图
        bars1 = ax2.bar(x - bar_width/2, sample_memberships[0], bar_width, label='样本1', color=sample_colors[0], alpha=0.8)
        bars2 = ax2.bar(x + bar_width/2, sample_memberships[1], bar_width, label='样本2', color=sample_colors[1], alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.3f}', 
                        ha='center', va='bottom', fontsize=9)
        
        # 设置x轴标签和图例
        ax2.set_xticks(x)
        ax2.set_xticklabels(hyperbox_types, rotation=30, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.05)
        
        # 主图设置
        ax1.legend(loc='upper right', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_xlabel('特征1', fontsize=12)
        ax1.set_ylabel('特征2', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, wspace=0.15)
        
        return fig, (ax1, ax2)
    
    def hyperrectangle_membership(self, point, center, width, height):
        """
        计算超矩形的隶属函数值
        参照fh_des_JFB_vector.py中的实现
        
        参数:
        point: 2D numpy数组 - 待计算的点
        center: 2D numpy数组 - 超矩形中心
        width: float - 超矩形宽度
        height: float - 超矩形高度
        
        返回:
        float - 隶属函数值
        """
        # 计算半宽
        halfsize = np.array([width/2, height/2])
        
        # 计算每个维度上的距离
        d = np.abs(point - center) - halfsize
        d[d < 0] = 0
        
        # 计算归一化距离
        dd = np.linalg.norm(d) / np.sqrt(len(point))
        
        # 隶属函数: (1 - dd)^6
        m = np.power(1 - dd, 6)
        
        return max(0, min(1, m))  # 确保隶属度在[0,1]范围内
    
    def hypersphere_membership(self, point, center, radius, sigma=0.1):
        """
        计算超球体的隶属函数值
        参照fh_des_JFB_vector_sphere.py中的实现
        
        参数:
        point: 2D numpy数组 - 待计算的点
        center: 2D numpy数组 - 超球体中心
        radius: float - 超球体半径
        sigma: float - 高斯函数的标准差参数
        
        返回:
        float - 隶属函数值
        """
        # 计算欧几里得距离
        distance = np.linalg.norm(point - center)
        
        # 高斯隶属函数: exp(-distance^2/(2*sigma^2))
        m = np.exp(-(distance**2) / (2 * sigma**2))
        
        # 半径衰减因子
        radius_factor = max(0, 1 - (distance / (radius + 1e-9)))
        
        # 隶属函数: 高斯函数 * 半径衰减因子
        membership = m * radius_factor
        
        return max(0, min(1, membership))  # 确保隶属度在[0,1]范围内
    
    def hyperellipsoid_membership(self, point, center, cov_matrix, gamma=1.0, reg_lambda=1e-6):
        """
        计算超椭球体的隶属函数值
        参照fh_des_JFB_vector_he.py中的实现
        
        参数:
        point: 2D numpy数组 - 待计算的点
        center: 2D numpy数组 - 超椭球体中心
        cov_matrix: 2x2 numpy数组 - 协方差矩阵
        gamma: float - 高斯函数的gamma参数
        reg_lambda: float - 正则化参数
        
        返回:
        float - 隶属函数值
        """
        # 计算差值
        delta = point - center
        
        # 添加正则化确保矩阵可逆
        cov_matrix_reg = cov_matrix + reg_lambda * np.eye(cov_matrix.shape[0])
        
        try:
            # 确保协方差矩阵是对称的
            cov_matrix_reg = (cov_matrix_reg + cov_matrix_reg.T) / 2
            
            # 计算逆矩阵
            inv_cov = np.linalg.inv(cov_matrix_reg)
            
            # 计算马氏距离的平方
            mahalanobis_squared = np.dot(np.dot(delta, inv_cov), delta)
            
            # 确保值是非负的
            mahalanobis_squared = max(0.0, mahalanobis_squared)
            
            # 计算马氏距离
            mahalanobis_distance = np.sqrt(mahalanobis_squared)
        except np.linalg.LinAlgError:
            # 如果计算失败，使用欧几里得距离作为备选
            mahalanobis_distance = np.linalg.norm(delta)
        
        # 高斯隶属函数: exp(-gamma * 马氏距离²)
        membership = np.exp(-gamma * (mahalanobis_distance ** 2))
        
        return max(0, min(1, membership))  # 确保隶属度在[0,1]范围内
    
    def double_slope_membership(self, point, center, width, height, alpha=0.5, beta=2.0):
        """
        计算双斜率超矩形的隶属函数值
        参照fh_des_JFB_vector_rectangle.py中的实现
        
        参数:
        point: 2D numpy数组 - 待计算的点
        center: 2D numpy数组 - 超矩形中心
        width: float - 超矩形宽度
        height: float - 超矩形高度
        alpha: float - 内部区域斜率参数
        beta: float - 外部区域斜率参数
        
        返回:
        float - 隶属函数值
        """
        # 计算半宽
        halfsize = np.array([width/2, height/2])
        
        # 计算每个维度上的相对距离
        rel_dist = np.abs(point - center) / (halfsize + 1e-9)
        
        # 双斜率隶属函数
        # 内部区域 (rel_dist <= 1): 1 - alpha * rel_dist
        m_internal = 1 - alpha * (rel_dist - np.minimum(rel_dist, np.ones_like(rel_dist)))
        
        # 外部区域 (rel_dist > 1): exp(-beta * (rel_dist - 1))
        m_external = np.exp(-beta * (rel_dist - np.ones_like(rel_dist)))
        
        # 组合内部和外部隶属度
        m = np.where(rel_dist <= 1, m_internal, m_external)
        
        # 对所有维度取最小值作为最终隶属度
        membership = np.min(m)
        
        return max(0, min(1, membership))  # 确保隶属度在[0,1]范围内
    
    def visualize_membership_functions(self):
        """
        可视化四种超盒类型的隶属函数
        """
        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('四种FH_DES超盒隶属函数可视化', fontsize=16)
        
        # 距离范围
        distances = np.linspace(0, 5, 1000)
        
        # 1. 超矩形隶属函数
        ax = axs[0, 0]
        ax.set_title('超矩形隶属函数')
        ax.set_xlabel('距离')
        ax.set_ylabel('隶属度')
        
        # 超矩形参数
        center = np.array([0, 0])
        width = 4
        height = 4
        max_distance = np.sqrt((width/2)**2 + (height/2)**2)
        
        memberships = []
        for d in distances:
            point = np.array([d, 0])  # 在x轴上移动的点
            mem = self.hyperrectangle_membership(point, center, width, height)
            memberships.append(mem)
        
        ax.plot(distances, memberships, color='#B2BEDD', linewidth=2)  # 莫兰迪蓝
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # 2. 超球体隶属函数
        ax = axs[0, 1]
        ax.set_title('超球体隶属函数')
        ax.set_xlabel('距离')
        ax.set_ylabel('隶属度')
        
        # 超球体参数
        radius = 2
        gamma = 1.0
        
        memberships = []
        for d in distances:
            point = np.array([d, 0])  # 在x轴上移动的点
            mem = self.hypersphere_membership(point, center, radius, gamma)
            memberships.append(mem)
        
        ax.plot(distances, memberships, color='#C1D0C4', linewidth=2)  # 莫兰迪绿
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # 3. 超椭球体隶属函数
        ax = axs[1, 0]
        ax.set_title('超椭球体隶属函数')
        ax.set_xlabel('距离')
        ax.set_ylabel('隶属度')
        
        # 超椭球体参数
        cov_matrix = np.array([[3, 1.5], [1.5, 2]])
        gamma = 0.5
        
        memberships = []
        for d in distances:
            point = np.array([d, 0])  # 在x轴上移动的点
            mem = self.hyperellipsoid_membership(point, center, cov_matrix, gamma)
            memberships.append(mem)
        
        ax.plot(distances, memberships, color='#DCC0CE', linewidth=2)  # 莫兰迪粉
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # 4. 双斜率超矩形隶属函数
        ax = axs[1, 1]
        ax.set_title('双斜率超矩形隶属函数')
        ax.set_xlabel('距离')
        ax.set_ylabel('隶属度')
        
        # 双斜率超矩形参数
        alpha = 0.5
        beta = 2.0
        
        memberships = []
        for d in distances:
            point = np.array([d, 0])  # 在x轴上移动的点
            mem = self.double_slope_membership(point, center, width, height, alpha, beta)
            memberships.append(mem)
        
        ax.plot(distances, memberships, color='#D4CBB7', linewidth=2)  # 莫兰迪棕
        
        # 标记边界
        half_diagonal = np.sqrt((width/2)**2 + (height/2)**2)
        ax.axvline(x=half_diagonal, color='gray', linestyle='--', alpha=0.7, label=f'边界距离: {half_diagonal:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        
        # 调整布局
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axs
    
    def visualize_comparison(self):
        """
        Visualize four types of hyperboxes in a 2x2 grid with gradient colors showing membership decay with distance
        """
        # Create a 2x2 grid of subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Comparison of Four Hyperbox Types and Membership Decay', fontsize=16, fontname='Times New Roman')
        
        # Unified center point for all subplots
        center = (5, 5)
        
        # Simulate two new sample points
        sample1 = (3, 4)  # First sample point
        sample2 = (6.5, 7)  # Second sample point
        samples = [sample1, sample2]
        sample_labels = ['Sample 1', 'Sample 2']
        sample_colors = ['#E8B79C', '#A8C2CB']  # Morandi orange and Morandi blue-gray
        
        # 1. Hyperrectangle (top-left subplot)
        ax = axs[0, 0]
        ax.set_title('Rectangular hyperbox', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Feature 1', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('Feature 2', fontsize=10, fontname='Times New Roman')
        
        # Set tick label fonts
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        rect_width, rect_height = 4, 4
        self.draw_hyperrectangle(ax, center, width=rect_width, height=rect_height, color='#B2BEDD', alpha=0.3, label='Hyperrectangle')
        
        # Create grid points for membership visualization
        x_range = np.linspace(2, 8, 50)
        y_range = np.linspace(2, 8, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        memberships = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                memberships[i, j] = self.hyperrectangle_membership(point, np.array(center), rect_width, rect_height)
        
        # Draw gradient heatmap
        im = ax.imshow(memberships, extent=[2, 8, 2, 8], origin='lower', cmap='Blues', alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Membership Degree')
        cbar.set_label('Membership Degree', fontname='Times New Roman')
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Add center point marker
        ax.plot(center[0], center[1], 'o', color='black', markersize=6, label='Center Point')
        
        # Draw and label sample points
        for i, (sample, label, color) in enumerate(zip(samples, sample_labels, sample_colors)):
            ax.plot(sample[0], sample[1], 'x', color=color, markersize=8, mew=2, label=label)
            
            # Calculate and display distance from sample to center
            distance = np.sqrt((sample[0] - center[0])**2 + (sample[1] - center[1])**2)
            ax.text(sample[0] + 0.2, sample[1] + 0.2, f'd={distance:.2f}', 
                    fontsize=10, color=color, bbox=dict(facecolor='white', alpha=0.7),
                    fontname='Times New Roman')
        
        ax.legend(prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 2. Hypersphere (top-right subplot)
        ax = axs[0, 1]
        ax.set_title('Spherical hyperbox', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Feature 1', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('Feature 2', fontsize=10, fontname='Times New Roman')
        
        # Set tick label fonts
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        sphere_radius = 2
        self.draw_hypersphere(ax, center, radius=sphere_radius, color='#C1D0C4', alpha=0.3, label='Hypersphere')
        
        # Calculate memberships for grid points
        memberships = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                # Adjust sigma parameter to make sphere fuzziness more obvious
                memberships[i, j] = self.hypersphere_membership(point, np.array(center), sphere_radius, sigma=0.5)
        
        # Draw gradient heatmap
        im = ax.imshow(memberships, extent=[2, 8, 2, 8], origin='lower', cmap='Greens', alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Membership Degree')
        cbar.set_label('Membership Degree', fontname='Times New Roman')
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Add center point marker
        ax.plot(center[0], center[1], 'o', color='black', markersize=6, label='Center Point')
        
        # Draw and label sample points
        for i, (sample, label, color) in enumerate(zip(samples, sample_labels, sample_colors)):
            ax.plot(sample[0], sample[1], 'x', color=color, markersize=8, mew=2, label=label)
            
            # Calculate and display distance from sample to center
            distance = np.sqrt((sample[0] - center[0])**2 + (sample[1] - center[1])**2)
            ax.text(sample[0] + 0.2, sample[1] + 0.2, f'd={distance:.2f}', 
                    fontsize=10, color=color, bbox=dict(facecolor='white', alpha=0.7),
                    fontname='Times New Roman')
        
        ax.legend(prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 3. Hyperellipsoid (bottom-left subplot)
        ax = axs[1, 0]
        ax.set_title('Ellipsoidal hyperbox', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Feature 1', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('Feature 2', fontsize=10, fontname='Times New Roman')
        
        # Set tick label fonts
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        cov_matrix = np.array([[3, 1.5], [1.5, 2]])
        self.draw_hyperellipsoid(ax, center, cov_matrix, color='#DCC0CE', alpha=0.3, label='Hyperellipsoid', scale=1.5)
        
        # Calculate memberships for grid points
        memberships = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                memberships[i, j] = self.hyperellipsoid_membership(point, np.array(center), cov_matrix)
        
        # Draw gradient heatmap
        im = ax.imshow(memberships, extent=[2, 8, 2, 8], origin='lower', cmap='Reds', alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Membership Degree')
        cbar.set_label('Membership Degree', fontname='Times New Roman')
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Add center point marker
        ax.plot(center[0], center[1], 'o', color='black', markersize=6, label='Center Point')
        
        # Draw and label sample points
        for i, (sample, label, color) in enumerate(zip(samples, sample_labels, sample_colors)):
            ax.plot(sample[0], sample[1], 'x', color=color, markersize=8, mew=2, label=label)
            
            # Calculate and display distance from sample to center
            distance = np.sqrt((sample[0] - center[0])**2 + (sample[1] - center[1])**2)
            ax.text(sample[0] + 0.2, sample[1] + 0.2, f'd={distance:.2f}', 
                    fontsize=10, color=color, bbox=dict(facecolor='white', alpha=0.7),
                    fontname='Times New Roman')
        
        ax.legend(prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # 4. Double-slope Hyperrectangle (bottom-right subplot)
        ax = axs[1, 1]
        ax.set_title('Bi-slope rectangular hyperbox', fontsize=12, fontname='Times New Roman')
        ax.set_xlabel('Feature 1', fontsize=10, fontname='Times New Roman')
        ax.set_ylabel('Feature 2', fontsize=10, fontname='Times New Roman')
        
        # Set tick label fonts
        for tick in ax.get_xticklabels():
            tick.set_fontname('Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        dslope_width, dslope_height = 4, 4
        self.draw_double_slope_rectangle(ax, center, width=dslope_width, height=dslope_height, color='#D4CBB7', alpha=0.3, label='Double-slope Hyperrectangle')
        
        # Calculate memberships for grid points
        memberships = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                point = np.array([X[i, j], Y[i, j]])
                memberships[i, j] = self.double_slope_membership(point, np.array(center), dslope_width, dslope_height)
        
        # Draw gradient heatmap
        im = ax.imshow(memberships, extent=[2, 8, 2, 8], origin='lower', cmap='Purples', alpha=0.8)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Membership Degree')
        cbar.set_label('Membership Degree', fontname='Times New Roman')
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Add center point marker
        ax.plot(center[0], center[1], 'o', color='black', markersize=6, label='Center Point')
        
        # Draw and label sample points
        for i, (sample, label, color) in enumerate(zip(samples, sample_labels, sample_colors)):
            ax.plot(sample[0], sample[1], 'x', color=color, markersize=8, mew=2, label=label)
            
            # Calculate and display distance from sample to center
            distance = np.sqrt((sample[0] - center[0])**2 + (sample[1] - center[1])**2)
            ax.text(sample[0] + 0.2, sample[1] + 0.2, f'd={distance:.2f}', 
                    fontsize=10, color=color, bbox=dict(facecolor='white', alpha=0.7),
                    fontname='Times New Roman')
        
        ax.legend(prop={'family': 'Times New Roman'})
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig, axs

    def visualize_fuzzy_hyperbox_concept(self):
        """
        Visualize the concept of fuzzy hyperboxes:
        - Different hyperbox representations (axis-aligned, center-radius, ellipsoid)
        - Mapping between distance functions and fuzzy membership functions
        - Visual representation of membership degrees
        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={'width_ratios': [1.2, 0.8]})
        
        # 1. Left: Visualization of different hyperbox representations
        ax1.set_title('Different Representations of Fuzzy Hyperboxes', fontsize=16, fontname='Times New Roman')
        ax1.set_xlabel('Feature 1', fontsize=12, fontname='Times New Roman')
        ax1.set_ylabel('Feature 2', fontsize=12, fontname='Times New Roman')
        
        # Set tick label fonts
        for tick in ax1.get_xticklabels():
            tick.set_fontname('Times New Roman')
        for tick in ax1.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Unified center point (used as reference)
        center = (5, 5)
        
        # a) Rectangular hyperbox (defined by minimum corner v and maximum corner w)
        # v=(v₁,…,v_d)⊤ for component lower bounds, w=(w₁,…,w_d)⊤ for component upper bounds
        v = (3, 3)  # Minimum corner (v1, v2)
        w = (7, 7)  # Maximum corner (w1, w2)
        width = w[0] - v[0]
        height = w[1] - v[1]
        
        # Calculate center of rectangular hyperbox for drawing
        rect_center = ((v[0] + w[0])/2, (v[1] + w[1])/2)
        
        # Draw rectangular hyperbox with color fading by distance
        self.draw_hyperrectangle(ax1, rect_center, width=width, height=height, 
                                color='#8EA3D1', alpha=0.3, label='Rectangular Hyperbox')
        
        # Add corner markers - unified marker size
        ax1.plot([v[0], v[0]], [v[1], v[1]], 'o', color='#8EA3D1', markersize=10, markerfacecolor='#8EA3D1')
        ax1.plot([w[0], w[0]], [w[1], w[1]], 'o', color='#8EA3D1', markersize=10, markerfacecolor='#8EA3D1')
        
        # Add symbol markers for bounds - unified font size
        ax1.text(v[0]-0.3, v[1]-0.3, f'v', 
                fontsize=18, color='#8EA3D1', ha='right', va='top', fontname='Times New Roman')
        ax1.text(w[0]+0.3, w[1]+0.3, f'w', 
                fontsize=18, color='#8EA3D1', ha='left', va='bottom', fontname='Times New Roman')
        
        # b) Spherical hyperbox (μ for center, s for radius)
        mu = (5, 5)  # Center μ
        s = 2.2      # Radius s
        self.draw_hypersphere(ax1, mu, radius=s, 
                            color='#8DBF9F', alpha=0.3, label='Spherical Hyperbox')
        
        # Add radius marker - solid double arrow, no specific value
        ax1.arrow(mu[0], mu[1], s, 0, 
                color='#8DBF9F', linewidth=2, alpha=0.7, 
                head_width=0.2, head_length=0.2, 
                length_includes_head=True)
        ax1.arrow(mu[0]+s, mu[1], -s, 0, 
                color='#8DBF9F', linewidth=2, alpha=0.7, 
                head_width=0.2, head_length=0.2, 
                length_includes_head=True)
        ax1.text(mu[0]+s/2, mu[1]-0.3, f's', 
                fontsize=18, color='#8DBF9F', ha='center', va='top', fontname='Times New Roman')
        
        # c) Ellipsoidal hyperbox (color fading by distance)
        cov_matrix = np.array([[2.5, 1.5], [1.5, 1.8]])  # Covariance matrix
        self.draw_hyperellipsoid(ax1, center, cov_matrix, 
                                color='#E89898', alpha=0.3, label='Ellipsoidal Hyperbox', scale=1.5)
        
        # Add eigenvector indicators of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        
        for i in range(2):
            length = 2 * np.sqrt(eigenvalues[i])
            vec = eigenvectors[:, i] * length
            ax1.arrow(center[0], center[1], vec[0], vec[1], 
                     color='#E89898', linewidth=2.5, alpha=0.8, head_width=0.2)
            ax1.arrow(center[0], center[1], -vec[0], -vec[1], 
                     color='#E89898', linewidth=2.5, alpha=0.8, head_width=0.2)
        
        # Add center marker (for spherical and ellipsoidal) - modified to white fill
        ax1.plot(center[0], center[1], 'o', color='black', markersize=12, markerfacecolor='white', linewidth=2)
        ax1.text(center[0]+0.3, center[1]+0.3, r'$\mu$', fontsize=12, color='white', fontname='Times New Roman')
        
        # Draw gradient heatmap to visualize membership
        x_range = np.linspace(2, 8, 50)
        y_range = np.linspace(2, 8, 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Calculate distance from each grid point to center (Euclidean distance as example)
        distances = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
        
        # Fuzzy membership function: map distance to membership degree
        # Using Gaussian membership function as example: exp(-gamma * distance^2)
        gamma = 0.2
        memberships = np.exp(-gamma * (distances ** 2))
        
        # Draw gradient heatmap
        im = ax1.imshow(memberships, extent=[2, 8, 2, 8], origin='lower', cmap='viridis', alpha=0.3)
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label='Membership Degree', shrink=0.8)
        cbar.ax.set_ylabel('Membership Degree', fontname='Times New Roman')
        for tick in cbar.ax.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Add new sample points, calculate distance and membership
        # Sample 1: Near the center
        sample1 = (4, 5.5)
        # Sample 2: Far from the center
        sample2 = (7.5, 6.5)
        samples = [sample1, sample2]
        sample_colors = ['#FF9933', '#3366CC']  # Orange and blue
        sample_labels = ['Sample 1', 'Sample 2']
        
        # Calculate Gaussian membership function parameters
        max_distance = 5
        linear_membership_max = 5
        
        for i, (sample, color, label) in enumerate(zip(samples, sample_colors, sample_labels)):
            # Draw sample points - unified marker size
            ax1.plot(sample[0], sample[1], 'o', color=color, markersize=12, markerfacecolor=color, linewidth=2, label=label)
            ax1.plot(sample[0], sample[1], 'x', color='white', markersize=10, mew=3)
            
            # Calculate distance from sample to center
            distance = np.sqrt((sample[0] - center[0])**2 + (sample[1] - center[1])**2)
            
            # Calculate membership degrees for different hyperboxes
            # Membership for rectangular hyperbox
            rect_membership = self.hyperrectangle_membership(np.array(sample), np.array(center), width, height)
            
            # Membership for spherical hyperbox
            sphere_membership = self.hypersphere_membership(np.array(sample), np.array(mu), s, sigma=0.5)
            
            # Membership for ellipsoidal hyperbox
            ellipsoid_membership = self.hyperellipsoid_membership(np.array(sample), np.array(center), cov_matrix)
            
            # Display sample information - enclosed in rectangle, placed more aesthetically
            info_text = f'd={distance:.2f}\n\n' + \
                       f'\u03bc_rect={rect_membership:.2f}\n' + \
                       f'\u03bc_sphere={sphere_membership:.2f}\n' + \
                       f'\u03bc_ellipsoid={ellipsoid_membership:.2f}'
            
            # Adjust text position based on sample location for better aesthetics
            if sample[0] > 5:  # Right side sample
                text_x = sample[0] + 0.6
                ha = 'left'
            else:  # Left side sample
                text_x = sample[0] - 2.2
                ha = 'right'
                
            # Vertical position unified at the same level as the sample point
            text_y = sample[1]
                
            ax1.text(text_x, text_y, info_text, 
                    fontsize=9, color=color, ha=ha, va='center',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, linewidth=1, boxstyle='round,pad=0.4'),
                    fontname='Times New Roman')
            
            # Mark sample distance and membership on right plot (enclosed in rectangle, placed aesthetically)
            ax2.axvline(x=distance, color=color, linestyle=':', linewidth=2, alpha=0.8)
            # Position at lower right corner, adjusted by sample index
            y_pos = 0.2 - i*0.08  # Add spacing
            ax2.text(5.5, y_pos, f'{label}: d={distance:.2f}', 
                    fontsize=10, color=color,
                    ha='right', va='center',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, linewidth=1, boxstyle='round,pad=0.3'),
                    fontname='Times New Roman')
        
        # 2. Right: Mapping from distance function to membership
        ax2.set_title('Mapping from Distance Function to Membership', fontsize=16, fontname='Times New Roman')
        ax2.set_xlabel('Distance', fontsize=12, fontname='Times New Roman')
        ax2.set_ylabel('Membership Degree', fontsize=12, fontname='Times New Roman')
        
        # Set tick label fonts for right plot
        for tick in ax2.get_xticklabels():
            tick.set_fontname('Times New Roman')
        for tick in ax2.get_yticklabels():
            tick.set_fontname('Times New Roman')
        
        # Distance range
        distance_range = np.linspace(0, 6, 100)
        
        # Different fuzzy membership functions
        # a) Gaussian membership function
        gaussian_membership = np.exp(-gamma * (distance_range ** 2))
        ax2.plot(distance_range, gaussian_membership, 
                color='#4682B4', linewidth=3, label='Gaussian Membership')
        
        # b) Linear membership function
        max_distance = 5
        linear_membership = np.maximum(0, 1 - distance_range / max_distance)
        ax2.plot(distance_range, linear_membership, 
                color='#2E8B57', linewidth=3, linestyle='--', label='Linear Membership')
        
        # c) Power membership function
        power_membership = np.power(np.maximum(0, 1 - distance_range / max_distance), 2)
        ax2.plot(distance_range, power_membership, 
                color='#B22222', linewidth=3, linestyle='-.', label='Power Membership')
        
        # Remove d=2.5 marker point
        
        
        # Add legend and grid
        ax2.legend(fontsize=11, prop={'family': 'Times New Roman'})
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # Add legend to left plot
        ax1.legend(fontsize=11, loc='upper right', prop={'family': 'Times New Roman'})
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, wspace=0.1)
        
        return fig, (ax1, ax2)

# 示例用法
if __name__ == "__main__":
    visualizer = HyperboxVisualizer()
    
    # # 可视化所有超盒类型
    # print("正在生成四种超盒的独立可视化...")
    # fig1, _ = visualizer.visualize_all_hyperboxes(examples=3)
    # plt.savefig('pictures/hyperboxes_visualization.png', dpi=300, bbox_inches='tight')
    # plt.close(fig1)  # 关闭图形以释放内存
    
    # 可视化比较
    print("正在生成四种超盒的比较可视化...")
    fig2, _ = visualizer.visualize_comparison()
    plt.savefig('pictures/hyperboxes_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)  # 关闭图形以释放内存
    
    # # 可视化隶属函数
    # print("正在生成四种超盒的隶属函数可视化...")
    # fig3, _ = visualizer.visualize_membership_functions()
    # plt.savefig('pictures/hyperboxes_membership_functions.png', dpi=300, bbox_inches='tight')
    # plt.close(fig3)  # 关闭图形以释放内存
    
    # # 可视化超盒重叠及样本隶属度对比
    # print("正在生成四种超盒重叠及样本隶属度对比可视化...")
    # fig4, _ = visualizer.visualize_overlap()
    # plt.savefig('pictures/hyperboxes_overlap.png', dpi=300, bbox_inches='tight')
    # plt.close(fig4)  # 关闭图形以释放内存
    
    # # 可视化模糊超盒概念
    # print("正在生成模糊超盒概念可视化...")
    # fig5, _ = visualizer.visualize_fuzzy_hyperbox_concept()
    # plt.savefig('pictures/hyperboxes_fuzzy_concept.png', dpi=300, bbox_inches='tight')
    # plt.close(fig5)  # 关闭图形以释放内存
    
    print("可视化已完成，图像已保存为：")
    print("- 'pictures/hyperboxes_visualization.png'")
    print("- 'pictures/hyperboxes_comparison.png'")
    print("- 'pictures/hyperboxes_membership_functions.png'")
    print("- 'pictures/hyperboxes_overlap.png'")
    print("- 'pictures/hyperboxes_fuzzy_concept.png'")
    print("显示可视化结果...")
    plt.show()