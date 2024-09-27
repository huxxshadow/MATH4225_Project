import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from tqdm import tqdm

# 清空所有图形
plt.close('all')

# 生成二维合成数据 (3个高斯分布)
np.random.seed(0)
data = np.vstack([np.random.multivariate_normal([0, 0], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([3, 5], np.eye(2), 100),
                  np.random.multivariate_normal([6, 0], np.eye(2) * 0.5, 100)])

# 设置高斯混合模型的参数
K = 3  # 高斯分布的个数
N, D = data.shape

# 随机初始化均值、协方差矩阵和混合系数
mu = np.random.rand(K, D)
cov = np.array([np.eye(D)] * K)
pi = np.random.dirichlet([1]*K)
responsibilities = np.zeros((N, K))  # 初始化责任度矩阵

# 变分分布中的 γ 参数，表示每个数据点属于每个簇的概率
gamma = np.zeros((N, K))


def plot_gmm(data, mu, cov, iteration):
    """绘制当前的高斯分布和数据点"""
    plt.scatter(data[:, 0], data[:, 1], color='green', alpha=0.3)
    for i in range(K):
        draw_ellipse(mu[i], cov[i], alpha=0.2)
    plt.title(f'Iteration {iteration}')
    plt.show()


def draw_ellipse(mean, cov, ax=None, color='red', alpha=1.0):
    """绘制高斯分布的椭圆"""
    from matplotlib.patches import Ellipse
    if ax is None:
        ax = plt.gca()
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=mean, width=lambda_[0] * 2, height=lambda_[1] * 2, angle=np.rad2deg(np.arccos(v[0, 0])))
    ell.set_facecolor(color)
    ell.set_alpha(alpha)
    ax.add_artist(ell)


# 迭代更新高斯混合模型参数
num_iterations = 100
alpha_0 = 1.0  # 先验的 Dirichlet 分布参数
for iteration in tqdm(range(num_iterations)):

    # E步：更新责任度 γ_{nk}，即 q(z_i^k)
    for n in range(N):
        for k in range(K):
            gamma[n, k] = pi[k] * multivariate_normal.pdf(data[n], mean=mu[k], cov=cov[k])
        gamma[n, :] /= np.sum(gamma[n, :])  # 归一化处理

    # M步：更新参数
    N_k = np.sum(gamma, axis=0)  # 每个簇的责任度之和
    for k in range(K):
        # 更新均值 μ_k
        mu[k] = np.sum(gamma[:, k].reshape(-1, 1) * data, axis=0) / N_k[k]
        # 更新协方差矩阵 Λ_k^{-1}
        cov[k] = np.dot((gamma[:, k].reshape(-1, 1) * (data - mu[k])).T, data - mu[k]) / N_k[k]
        # 更新混合系数 π_k
        pi[k] = (N_k[k] + alpha_0 - 1) / (N + K * (alpha_0 - 1))  # Dirichlet 分布的变分更新
        # print("mu:", mu)

    # 绘制当前的高斯混合模型
    # if iteration in [0, 15, 60, 120]:

plot_gmm(data, mu, cov, 100)