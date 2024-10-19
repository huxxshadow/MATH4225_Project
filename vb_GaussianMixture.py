
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
from scipy.special import digamma, gammaln, logsumexp, betaln
from sklearn.mixture import BayesianGaussianMixture

# clear all memory
plt.close('all')

# np.random.seed(3)




# 均值列表
init_means = [
    [0, 15],   # 第1个高斯分布的均值
    [15, 18],  # 第2个高斯分布的均值
    [20, 30],  # 第3个高斯分布的均值
    [11, 15],  # 第4个高斯分布的均值
    [10, 30],  # 第5个高斯分布的均值
    [12, 45]   # 第6个高斯分布的均值
]

# 协方差矩阵列表
init_covariances = [
    np.eye(2) * 10,  # 第1个高斯分布的协方差矩阵
    np.eye(2) * 5,   # 第2个高斯分布的协方差矩阵
    np.eye(2) * 7,   # 第3个高斯分布的协方差矩阵
    np.eye(2) * 8,   # 第4个高斯分布的协方差矩阵
    np.eye(2) * 9,   # 第5个高斯分布的协方差矩阵
    np.eye(2) * 1    # 第6个高斯分布的协方差矩阵
]

# 根据均值和协方差生成数据
data = np.vstack([
    np.random.multivariate_normal(mean, covariance, 50)
    for mean, covariance in zip(init_means, init_covariances)
])

# 使用scikit-learn中的变分贝叶斯高斯混合模型 作为对照
gmm_vb = BayesianGaussianMixture(tol=1e-7, n_components=6, covariance_type='full', random_state=2,
                                 init_params='random_from_data',
                                 weight_concentration_prior_type="dirichlet_process",
                                 verbose=2, verbose_interval=1)
gmm_vb.fit(data)

# 每次迭代更新的参数


# resp
# means
# covariances


# 手动实现变分贝叶斯高斯混合模型
# 设置高斯混合模型的参数
# 初始化变分贝叶斯参数

K = 6  # 高斯分布的个数 components
N, D = data.shape  # n_samples, n_features
print("数据的形状:", data.shape)

# resp = np.zeros((N, K))  # responsibilities  Z的后验概率期望,近似文章中的Z,表示每个数据点属于每个簇的概率
# indices = np.random.choice(N, size=K, replace=False)
# resp[indices, np.arange(K)] = 1  # 初始化resp矩阵，每个数据点属于每个簇的概率

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
resp = np.zeros((N, K))
resp[np.arange(N), kmeans.labels_] = 1



alpha_0 = 1 / K  # 用于每个高斯成分的权重pi 的先验参数,Dirichlet分布的参数
beta_0 = 1  # 均值means的先验权重 表示对这个均值假设的信任程度
v_0 = D  # 协方差矩阵cov的先验自由度
reg_covar = 1e-7  # 用于协方差矩阵的正则化参数 防止协方差矩阵奇异性和数值不稳定的工具，通常是一个小的正数
mean_prior = np.mean(data, axis=0)  # 均值的先验均值  表示我们认为均值应该位于什么位置
covariance_prior = np.atleast_2d(np.cov(data.T))  # 协方差矩阵的先验  表示我们认为协方差矩阵应该是什么样子

lower_bound = -np.inf
max_lower_bound = -np.inf

sum_resp = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # resp的和

# 基于当前的数据责任度 (resp) 计算得到的样本均值，它是对每个高斯成分的软分配加权平均。
old_means = np.dot(resp.T, data) / sum_resp[:, np.newaxis] 

# 是在当前数据责任度下估计的协方差矩阵，直接反映数据的分布情况。
old_covariances = np.empty((K, D, D))
for k in range(K):
    diff = data - old_means[k]
    old_covariances[k] = np.dot(resp[:, k] * diff.T, diff) / sum_resp[k]
    old_covariances[k].flat[:: D + 1] += reg_covar

# alpha = alpha_0 + sum_resp
# 这里采用了Dirichlet过程，Dirichlet过程混合模型（DPMM）是一种无限混合模型，它使用Dirichlet过程作为混合系数的先验分布。
# 它是 Dirichlet 分布的扩展，用于处理无限多个类别。它允许模型从数据中自适应选择类别的数量，而不是事先固定类别数量。
alpha = (1.0 + sum_resp, (alpha_0 + np.hstack((np.cumsum(sum_resp[::-1])[-2::-1], 0))),)

beta = beta_0 + sum_resp

# 基于先验信息和样本估计更新得到的后验均值。它结合了先验均值（mean_prior）和从数据中估计得到的均值（即 old_means）进行加权更新。
means = (beta_0 * mean_prior + sum_resp[:, np.newaxis] * old_means) / beta[:, np.newaxis] 

v = v_0 + sum_resp

# 是结合了先验信息和数据的贝叶斯后验协方差矩阵，经过了更复杂的后验推断，包含了先验信息和均值偏差的影响。
covariances = np.empty((K, D, D))

for k in range(K):
    diff = old_means[k] - mean_prior
    covariances[k] = (
            covariance_prior
            + sum_resp[k] * old_covariances[k]
            + sum_resp[k]
            * beta_0
            / beta[k]
            * np.outer(diff, diff)
    )
# Contrary to the original bishop book, we normalize the covariances
covariances /= v[:, np.newaxis, np.newaxis]

# 协方差矩阵的逆，即精度矩阵的 Cholesky 分解，（下三角矩阵）
precisions_cholesky = np.array([
    linalg.cholesky(np.linalg.inv(covariances[k]), lower=True)
    for k in range(K)
])


# 修改绘制椭圆的函数，接受颜色和标签参数
def draw_ellipse(mean, cov, ax=None, color='red', alpha=1.0, label=None, linewidth=2):
    """Draw an ellipse representing a Gaussian distribution."""
    if ax is None:
        ax = plt.gca()
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * np.sqrt(eigenvalues)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  edgecolor=color, facecolor='none', linewidth=linewidth, label=label)
    ell.set_alpha(alpha)
    ax.add_artist(ell)


num_iterations = 1000
best_params = None
tolerance = 1e-6  # 设定一个较小的阈值
for iteration in range(1, num_iterations + 1):
    # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
    # because the precision matrix is normalized.

    # 计算新 lower_bound
    if abs(lower_bound - max_lower_bound) < tolerance:
        print(f"Converged at iteration {iteration}")
        break

    #  log Λk
    log_det = np.sum(np.log(precisions_cholesky.reshape(K, -1)[:, ::D + 1]), axis=1)
    # 中间量
    log_prob = np.array([
        -0.5 * np.sum((np.dot(data - means[k], precisions_cholesky[k]) ** 2), axis=1)
        + log_det[k]
        for k in range(K)
    ]).T
    # log p(x|μk,inv(Ak) )
    log_gauss = log_prob - 0.5 * D * np.log(2 * np.pi) - 0.5 * D * np.log(v_0)

    # E[log|Λk|]
    log_lambda = D * np.log(2) + np.sum(
        digamma(0.5 * (v[:, np.newaxis] - np.arange(D))), axis=1
    )
    # 中间量
    log_det_precisions_chol = log_det - 0.5 * D * np.log(v)

    # E[log|Λk|]
    log_wishart = -(
            v * log_det_precisions_chol
            + v * D * 0.5 * np.log(2)
            + np.sum(gammaln(0.5 * (v - np.arange(D)[:, np.newaxis])), axis=0)
    )

    # log_norm_alpha = _log_dirichlet_norm(alpha)
    # log_norm_alpha = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

    # logB(alpha)
    log_norm_alpha = np.sum(-np.sum(
        betaln(alpha[0], alpha[1])
    ))

    # We remove `n_features * np.log(self.degrees_of_freedom_)` because
    # the precision matrix is normalized
    #
    # log_alpha = digamma(alpha) - digamma(
    #     np.sum(alpha))

    digamma_sum = digamma(alpha[0] + alpha[1])
    digamma_a = digamma(alpha[0])
    digamma_b = digamma(alpha[1])
    # Eπ[log πk] 公式 (52)第一项
    log_alpha = digamma_a - digamma_sum + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))

    # 计算 log_rho，对应公式 (52)
    weighted_log_prob = log_gauss + 0.5 * (log_lambda - D / beta) + log_alpha
    # 计算 log_rho 的归一化项
    log_prob_norm = logsumexp(weighted_log_prob, axis=1)

    log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    # 公式: resp = ρ/sum(ρ) 对应公式 (55)
    resp = np.exp(log_resp)






    # lower_bound = (
    #         -np.sum(np.exp(log_resp) * log_resp)
    #         - log_wishart
    #         - log_norm_alpha
    #         - 0.5 * D * np.sum(np.log(beta))
    # )
    lower_bound = (
            -np.sum(resp * log_resp)
            - np.sum(log_wishart)
            - log_norm_alpha
            - 0.5 * D * np.sum(np.log(beta))
    )
    print("lower_bound:", lower_bound)
    print(f"Iteration {iteration + 1}:")

    if (lower_bound > max_lower_bound or max_lower_bound == -np.inf) and iteration > 1:
        max_lower_bound = lower_bound
        best_n_iter = iteration
        best_params = (means.copy(), precisions_cholesky.copy(), alpha, beta.copy(), v.copy(), covariances.copy())

    sum_resp = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    old_means = np.dot(resp.T, data) / sum_resp[:, np.newaxis]
    old_means = np.dot(resp.T, data) / sum_resp[:, np.newaxis]
    old_covariances = np.array([
        np.cov(data.T, aweights=resp[:, k], bias=True) + reg_covar * np.eye(D)
        for k in range(K)
    ])

    # alpha = alpha_0 + sum_resp
    alpha = (1.0 + sum_resp, alpha_0 + np.hstack((np.cumsum(sum_resp[::-1])[-2::-1], 0)))
    beta = beta_0 + sum_resp
    means = (beta_0 * mean_prior + sum_resp[:, np.newaxis] * old_means) / beta[:, np.newaxis]
    v = v_0 + sum_resp

    covariances = np.empty((K, D, D))

    for k in range(K):
        diff = old_means[k] - mean_prior
        covariances[k] = (
                covariance_prior
                + sum_resp[k] * old_covariances[k]
                + (sum_resp[k] * beta_0 / beta[k]) * np.outer(diff, diff)
        )
    # Contrary to the original bishop book, we normalize the covariances
    covariances /= v[:, np.newaxis, np.newaxis]
    precisions_cholesky = np.array([
        linalg.cholesky(np.linalg.inv(covariances[k]), lower=True)
        for k in range(K)
    ])

means, precisions_cholesky, alpha, beta, v, covariances = best_params


# print("best_params:", best_params)


# 在同一张图上绘制手动实现的模型和scikit-learn模型
# 在同一张图上绘制手动实现的模型和scikit-learn模型
def plot_comparison(data, manual_means, manual_covs, sklearn_model, iteration):
    """Plot the manual GMM and scikit-learn GMM on two subplots."""
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    # Plot manual GMM results on the first subplot
    axs[0].scatter(data[:, 0], data[:, 1], color='green', alpha=0.3, label='Data')
    for i in range(K):
        draw_ellipse(manual_means[i], manual_covs[i], ax=axs[0], color='red', alpha=0.3, linewidth=6,
                     label='Manual GMM' if i == 0 else None)

    # 绘制真实的高斯分布椭圆
    for i, (mean, cov) in enumerate(zip(init_means, init_covariances)):
        draw_ellipse(mean, cov, ax=axs[0], color='black', alpha=0.8, linewidth=2,
                     label='Real Gaussian Distribution' if i == 0 else None)

    axs[0].set_title('Manual GMM and Real Gaussian Distribution')
    axs[0].legend()
    axs[0].set_xlabel('Feature 1')
    axs[0].set_ylabel('Feature 2')
    axs[0].grid(True)

    # Plot scikit-learn GMM results on the second subplot
    axs[1].scatter(data[:, 0], data[:, 1], color='green', alpha=0.3, label='Data')
    for i in range(sklearn_model.means_.shape[0]):
        draw_ellipse(sklearn_model.means_[i], sklearn_model.covariances_[i], ax=axs[1], color='blue', alpha=0.8, linewidth=2,
                     label='scikit-learn GMM' if i == 0 else None)

    # 绘制真实的高斯分布椭圆
    for i, (mean, cov) in enumerate(zip(init_means, init_covariances)):
        draw_ellipse(mean, cov, ax=axs[1], color='black', alpha=0.8, linewidth=2,
                     label='Real Gaussian Distribution' if i == 0 else None)

    axs[1].set_title('scikit-learn GMM and Real Gaussian Distribution')
    axs[1].legend()
    axs[1].set_xlabel('Feature 1')
    axs[1].set_ylabel('Feature 2')
    axs[1].grid(True)

    plt.suptitle(f'Iteration {iteration} - Manual vs scikit-learn GMM Comparison')
    plt.show()


# 假设这些是手动实现的结果
manual_means = means  # 你手动计算的均值
manual_covs = covariances  # 你手动计算的协方差矩阵

# 调用 plot_comparison 函数
plot_comparison(data, manual_means, manual_covs, gmm_vb, num_iterations)
