import numpy as np
from scipy.special import digamma, gammaln, psi
import matplotlib.pyplot as plt

# 清空所有图形
plt.close('all')

# 生成二维合成数据 (3个高斯分布)
np.random.seed(0)
data = np.vstack([np.random.multivariate_normal([0, 0], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([3, 5], np.eye(2), 100),
                  np.random.multivariate_normal([6, 0], np.eye(2) * 0.5, 100)])
N, D = data.shape  # 数据点数量和维度

# 设置高斯混合模型的参数
K = 3  # 高斯分布的个数

# 初始化先验参数
alpha_0 = np.ones(K)  # Dirichlet 分布的先验参数
m_0 = np.zeros(D)  # 均值的先验均值
beta_0 = 1.0  # 均值的先验精度
W_0 = np.eye(D)  # 协方差矩阵的先验
v_0 = D  # Wishart 分布的自由度

# 初始化变分参数
alpha = alpha_0 + N / K  # 更新 Dirichlet 参数
beta = beta_0 + N / K
m = np.random.rand(K, D)
W = np.array([np.eye(D) for _ in range(K)])
v = v_0 + N / K

# 责任度矩阵（r_nk），表示 q(z_i^k)
r = np.zeros((N, K))


def compute_E_log_pi(alpha):
    """计算 E[log(π_k)] """
    return digamma(alpha) - digamma(np.sum(alpha))


def compute_E_log_Lambda(v, W):
    """计算 E[log(|Λ_k|)] """
    D = W[0].shape[0]
    E_log_Lambda = np.zeros(K)
    for k in range(K):
        E_log_Lambda[k] = np.sum(digamma(0.5 * (v[k] + 1 - np.arange(1, D + 1)))) + D * np.log(2) + np.log(
            np.linalg.det(W[k]))
    return E_log_Lambda


def compute_E_Lambda_mu(mu, m, beta, v, W):
    """计算 E[(x_n - μ_k)^T Λ_k (x_n - μ_k)] """
    E = np.zeros(N)
    for n in range(N):
        temp = np.zeros(K)
        for k in range(K):
            xm = (data[n] - m[k]).reshape(-1, 1)
            temp[k] = D / beta[k] + v[k] * xm.T @ W[k] @ xm
        E[n] = temp
    return E

def logsum_exp(a, axis=None):
    """计算 log(sum(exp(a)))，避免数值下溢"""
    a_max = np.max(a, axis=axis, keepdims=True)
    return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))


def draw_ellipse(mean, cov, ax=None, color='red', alpha=1.0):
    """绘制高斯分布的等高线椭圆"""
    from matplotlib.patches import Ellipse
    if ax is None:
        ax = plt.gca()
    lambda_, v = np.linalg.eigh(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=mean, width=lambda_[0] * 2, height=lambda_[1] * 2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), color=color)
    ell.set_alpha(alpha)
    ax.add_artist(ell)

# 迭代更新高斯混合模型参数
num_iterations = 100
for iteration in range(num_iterations):

    # E步：更新责任度 r_{nk}
    E_log_pi = compute_E_log_pi(alpha)
    E_log_Lambda = compute_E_log_Lambda(v, W)
    ln_rho = np.zeros((N, K))
    for n in range(N):
        for k in range(K):
            xm = (data[n] - m[k]).reshape(-1, 1)
            E_quadform = D / beta[k] + v[k] * (xm.T @ W[k] @ xm)
            ln_rho[n, k] = E_log_pi[k] + 0.5 * E_log_Lambda[k] - 0.5 * E_quadform - D / 2 * np.log(2 * np.pi)
    # 归一化 r
    ln_r = ln_rho - logsum_exp(ln_rho, axis=1)
    r = np.exp(ln_r)

    # M步：更新变分参数
    N_k = np.sum(r, axis=0) + 1e-10  # 加上一个小值防止除零
    x_bar = (r.T @ data) / N_k.reshape(-1, 1)

    # 更新 alpha（Dirichlet 分布参数）
    alpha = alpha_0 + N_k

    # 更新 beta, m, W, v
    for k in range(K):
        beta[k] = beta_0 + N_k[k]
        m[k] = (beta_0 * m_0 + N_k[k] * x_bar[k]) / beta[k]
        diff = data - x_bar[k]
        S_k = sum([r[n, k] * np.outer(diff[n], diff[n]) for n in range(N)])
        W_k_inv = np.linalg.inv(W_0) + N_k[k] * S_k + (beta_0 * N_k[k] / beta[k]) * np.outer(x_bar[k] - m_0,
                                                                                             x_bar[k] - m_0)
        W[k] = np.linalg.inv(W_k_inv)
        v[k] = v_0 + N_k[k]

    # 可视化每隔一定次数的结果
    if iteration % 20 == 0 or iteration == num_iterations - 1:
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], c=np.argmax(r, axis=1), cmap='viridis', alpha=0.5)
        for k in range(K):
            draw_ellipse(m[k], np.linalg.inv(v[k] * W[k]), color='red', alpha=0.2)
        plt.title(f'Iteration {iteration}')
        plt.show()


