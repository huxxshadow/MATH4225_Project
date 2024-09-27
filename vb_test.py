import numpy as np
from scipy.special import digamma, gammaln, psi
from scipy.linalg import inv, det
import matplotlib.pyplot as plt
from tqdm import tqdm

# 生成二维合成数据 (3个高斯分布)
np.random.seed(0)
data = np.vstack([np.random.multivariate_normal([0, 0], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([3, 5], np.eye(2), 100),
                  np.random.multivariate_normal([6, 0], np.eye(2) * 0.5, 100)])

# 设置高斯混合模型的参数
K = 3  # 高斯分布的个数
N, D = data.shape

# 初始化变分贝叶斯参数
np.random.seed(0)
alpha_0 = 0.1  # Dirichlet先验参数
beta_0 = 1e-3
m_0 = np.mean(data, axis=0)
W_0 = np.eye(D)
v_0 = D

# 初始化q分布的参数
alpha = np.ones(K) * (alpha_0 + N / K)
beta = np.ones(K) * (beta_0 + N / K)
m = np.random.randn(K, D)
W = np.array([np.eye(D)] * K)
v = np.ones(K) * (v_0 + N / K)
responsibilities = np.zeros((N, K))  # 责任度矩阵

def plot_gmm(data, m, W, iteration):
    """绘制当前的高斯分布和数据点"""
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], color='green', alpha=0.3)
    for i in range(K):
        draw_ellipse(m[i], inv(W[i] * v[i]), alpha=0.2)
    plt.title(f'Iteration {iteration}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def draw_ellipse(mean, cov, ax=None, color='red', alpha=1.0):
    """绘制高斯分布的椭圆"""
    from matplotlib.patches import Ellipse
    if ax is None:
        ax = plt.gca()
    lambda_, v_ = np.linalg.eigh(cov)
    lambda_ = np.sqrt(lambda_)
    ell = Ellipse(xy=mean, width=lambda_[0] * 2, height=lambda_[1] * 2,
                  angle=np.rad2deg(np.arccos(v_[0, 0])))
    ell.set_facecolor(color)
    ell.set_alpha(alpha)
    ax.add_artist(ell)

def compute_ln_lambda(k):
    """计算期望的 ln|Lambda_k|"""
    return np.sum(digamma(0.5 * (v[k] + 1 - np.arange(1, D + 1)))) + D * np.log(2) + np.log(det(W[k]))

def compute_E_q_ln_pi():
    """计算期望的 ln(pi)"""
    return digamma(alpha) - digamma(np.sum(alpha))

# 迭代更新参数
num_iterations = 100
for iteration in tqdm(range(num_iterations)):

    E_ln_pi = compute_E_q_ln_pi()
    ln_lambda = np.array([compute_ln_lambda(k) for k in range(K)])

    # E步：计算责任度
    for n in range(N):
        for k in range(K):
            diff = data[n] - m[k]
            E_quad = D / beta[k] + v[k] * np.dot(diff.T, np.dot(W[k], diff))
            responsibilities[n, k] = np.exp(E_ln_pi[k] + 0.5 * ln_lambda[k] - 0.5 * E_quad)
        responsibilities[n, :] /= np.sum(responsibilities[n, :]) + 1e-12  # 归一化，防止除以零

    Nk = responsibilities.sum(axis=0) + 1e-10  # 防止0除

    # M步：更新参数
    alpha = alpha_0 + Nk

    for k in range(K):
        # 更新beta
        beta[k] = beta_0 + Nk[k]
        # 更新m
        m[k] = (beta_0 * m_0 + Nk[k] * np.sum(responsibilities[:, k].reshape(-1, 1) * data, axis=0)) / beta[k]
        # 更新W
        diff_data = data - m[k]
        S = np.dot((responsibilities[:, k].reshape(-1, 1) * diff_data).T, diff_data)
        diff_m = m[k] - m_0
        W_k_inv = inv(W_0) + Nk[k] * S + (beta_0 * Nk[k]) / beta[k] * np.outer(diff_m, diff_m)

        # 检查 W_k_inv 是否有 NaN 或 inf
        if np.isnan(W_k_inv).any() or np.isinf(W_k_inv).any():
            print(f"Encountered NaN or inf in W_k_inv at iteration {iteration}, component {k}")
            continue  # 跳过这一轮迭代

        W[k] = inv(W_k_inv)
        # 更新v
        v[k] = v_0 + Nk[k]

    # 绘制当前的高斯混合模型
    # if iteration in [0, 15, 60, 99]:
    #     plot_gmm(data, m, W, iteration)

# 最终绘制结果
plot_gmm(data, m, W, num_iterations)