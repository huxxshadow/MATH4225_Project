import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import linalg
from scipy.special import digamma, gammaln, logsumexp, betaln
from sklearn.mixture import BayesianGaussianMixture

from sklearn.mixture._bayesian_mixture import _log_dirichlet_norm

# clear all memory
plt.close('all')

# 生成二维合成数据 (3个高斯分布) 这里协方差采用np.eye(2)表示变量是独立的
np.random.seed(3)
data = np.vstack([np.random.multivariate_normal([0, 15], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([12, 15], np.eye(2), 100),
                  np.random.multivariate_normal([6, 0], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([10, 15], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([20, 20], np.eye(2) * 0.5, 100),
                  np.random.multivariate_normal([30, 30], np.eye(2) * 0.5, 100)
                  ])

# 使用scikit-learn中的变分贝叶斯高斯混合模型
gmm_vb = BayesianGaussianMixture(tol=1e-7, n_components=6, covariance_type='full', random_state=2,
                                 init_params='random_from_data',
                                 weight_concentration_prior_type="dirichlet_process",
                                 verbose=2, verbose_interval=1)
gmm_vb.fit(data)

# 数据标准化
K = 6  # 高斯分布的个数 components
N, D = data.shape  # n_samples, n_features
print("数据的形状:", data.shape)

resp = np.zeros((N, K))  # responsibilities
indices = np.random.choice(
    N, size=K, replace=False
)
resp[indices, np.arange(K)] = 1
alpha_0 = 1 / K
beta_0 = 1
v_0 = D
reg_covar = 1e-7
covariance_prior = np.atleast_2d(np.cov(data.T))
mean_prior = np.mean(data, axis=0)
lower_bound = -np.inf
max_lower_bound = -np.inf

zk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # weights
old_means = np.dot(resp.T, data) / zk[:, np.newaxis]
old_covariances = np.empty((K, D, D))
for k in range(K):
    diff = data - old_means[k]
    old_covariances[k] = np.dot(resp[:, k] * diff.T, diff) / zk[k]
    old_covariances[k].flat[:: D + 1] += reg_covar

# alpha = alpha_0 + zk
alpha = (1.0 + zk,(alpha_0+ np.hstack((np.cumsum(zk[::-1])[-2::-1], 0))),)

beta = beta_0 + zk
means = (beta_0 * mean_prior + zk[:, np.newaxis] * old_means) / beta[:, np.newaxis]

v = v_0 + zk

covariances = np.empty((K, D, D))

for k in range(K):
    diff = old_means[k] - mean_prior
    covariances[k] = (
            covariance_prior
            + zk[k] * old_covariances[k]
            + zk[k]
            * beta_0
            / beta[k]
            * np.outer(diff, diff)
    )
# Contrary to the original bishop book, we normalize the covariances
covariances /= v[:, np.newaxis, np.newaxis]

estimate_precision_error_message = (
    "Fitting the mixture model failed because some components have "
    "ill-defined empirical covariance (for instance caused by singleton "
    "or collapsed samples). Try to decrease the number of components, "
    "or increase reg_covar."
)
precisions_cholesky = np.empty((K, D, D))
for k, covariance in enumerate(covariances):
    try:
        cov_cholesky = linalg.cholesky(covariance, lower=True)
    except np.linalg.LinAlgError:
        raise ValueError(estimate_precision_error_message)
    precisions_cholesky[k] = linalg.solve_triangular(
        cov_cholesky, np.eye(D), lower=True
    ).T

    #


# m = means
# W = np.linalg.inv(covariances)
# mu_0 = mean_prior
# W_0 = np.linalg.inv(covariance_prior)


# # 设置先验参数
# alpha_0 = 1/K
# beta_0 = 1.0
# mu_0 = np.mean(data, axis=0)
# W_0 = np.eye(D)
# v_0 = D

# # 随机初始化均值、协方差矩阵和混合系数
# alpha = np.ones(K) * alpha_0
# beta = np.ones(K) * (beta_0 + N / K)
# m = data[np.random.choice(N, K, replace=False)]  # 初始化为数据中的随机样本


# v = np.ones(K) * (v_0 + N / K)
# W = np.array([np.eye(D) for _ in range(K)])
#
# cov = np.array([np.eye(D)] * K)
# pi = np.random.dirichlet([1] * K)  # π Dirichlet 分布的参数
# z = np.zeros((N, K))  # z   每个数据点属于每个高斯分布的概率


# 修改绘制椭圆的函数，接受颜色和标签参数
def draw_ellipse(_mean, _cov, ax=None, color='red', alpha=1.0, label=None, linewidth=2):
    """绘制高斯分布的椭圆"""
    if ax is None:
        ax = plt.gca()
    _lambda, _v = np.linalg.eigh(_cov)  # 计算特征值和特征向量
    _lambda = np.sqrt(_lambda)  # 椭圆的半长轴和半短轴
    ell = Ellipse(xy=_mean, width=_lambda[0] * 2, height=_lambda[1] * 2,
                  angle=np.rad2deg(np.arccos(_v[0, 0])), edgecolor=color,
                  facecolor='none', linewidth=linewidth, label=label)
    ell.set_alpha(alpha)
    ax.add_artist(ell)


# def logsumexp(a, axis=None):
#     """计算 log(sum(exp(a)))，避免数值下溢"""
#     a_max = np.max(a, axis=axis, keepdims=True)
#     return a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))


num_iterations = 100
best_params = None

for iteration in range(num_iterations):
    # We removed `.5 * n_features * np.log(self.degrees_of_freedom_)`
    # because the precision matrix is normalized.
    log_det = np.sum(
        np.log(precisions_cholesky.reshape(K, -1)[:, :: D + 1]), 1
    )

    log_prob = np.empty((N, K))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
        y = np.dot(data, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)

    log_gauss = (-0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det
                 ) - 0.5 * D * np.log(v_0)

    log_lambda = D * np.log(2.0) + np.sum(
        digamma(
            0.5
            * (v - np.arange(0, D)[:, np.newaxis])
        ),
        0,
    )

    log_det_precisions_chol = np.sum(
            np.log(precisions_cholesky.reshape(K, -1)[:, :: D + 1]), 1
        ) - 0.5 * D * np.log(v)

    log_wishart = -(
            v * log_det_precisions_chol
            + v * D * 0.5 * math.log(2.0)
            + np.sum(
        gammaln(0.5 * (v - np.arange(D)[:, np.newaxis])),
        0,
    )
    )

    # log_norm_alpha = _log_dirichlet_norm(alpha)
    # log_norm_alpha = np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))

    log_norm_alpha = np.sum(-np.sum(
                betaln(alpha[0], alpha[1])
            ))

    # We remove `n_features * np.log(self.degrees_of_freedom_)` because
    # the precision matrix is normalized
    #
    # log_alpha = digamma(alpha) - digamma(
    #     np.sum(alpha))

    digamma_sum = digamma(
        alpha[0] + alpha[1]
    )
    digamma_a = digamma(alpha[0])
    digamma_b = digamma(alpha[1])
    log_alpha= digamma_a- digamma_sum+ np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))


    weighted_log_prob = log_gauss + 0.5 * (log_lambda - D / beta) + log_alpha

    log_prob_norm = logsumexp(weighted_log_prob, axis=1)

    with np.errstate(under="ignore"):
        # ignore underflow
        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
    # np.mean(log_prob_norm), log_resp

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
        best_params = (means, precisions_cholesky, alpha, beta, v, covariances)


    zk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps  # weights
    old_means = np.dot(resp.T, data) / zk[:, np.newaxis]
    old_covariances = np.empty((K, D, D))
    for k in range(K):
        diff = data - old_means[k]
        old_covariances[k] = np.dot(resp[:, k] * diff.T, diff) / zk[k]
        old_covariances[k].flat[:: D + 1] += reg_covar

    # alpha = alpha_0 + zk
    alpha=(1.0 + zk, (alpha_0 + np.hstack((np.cumsum(zk[::-1])[-2::-1], 0))),)

    beta = beta_0 + zk
    means = (beta_0 * mean_prior + zk[:, np.newaxis] * old_means) / beta[:, np.newaxis]

    v = v_0 + zk

    covariances = np.empty((K, D, D))

    for k in range(K):
        diff = old_means[k] - mean_prior
        covariances[k] = (
                covariance_prior
                + zk[k] * old_covariances[k]
                + zk[k]
                * beta_0
                / beta[k]
                * np.outer(diff, diff)
        )
    # Contrary to the original bishop book, we normalize the covariances
    covariances /= v[:, np.newaxis, np.newaxis]

    estimate_precision_error_message = (
        "Fitting the mixture model failed because some components have "
        "ill-defined empirical covariance (for instance caused by singleton "
        "or collapsed samples). Try to decrease the number of components, "
        "or increase reg_covar."
    )
    precisions_cholesky = np.empty((K, D, D))
    for k, covariance in enumerate(covariances):
        try:
            cov_cholesky = linalg.cholesky(covariance, lower=True)
        except np.linalg.LinAlgError:
            raise ValueError(estimate_precision_error_message)
        precisions_cholesky[k] = linalg.solve_triangular(
            cov_cholesky, np.eye(D), lower=True
        ).T

means, precisions_cholesky, alpha, beta, v, covariances = best_params
# print("best_params:", best_params)




# 在同一张图上绘制手动实现的模型和scikit-learn模型
def plot_comparison(data, manual_means, manual_covs, sklearn_model, iteration):
    """在同一张图上绘制手动实现的模型和scikit-learn的模型"""
    plt.scatter(data[:, 0], data[:, 1], color='green', alpha=0.3, label='Data')

    # 绘制手动实现的结果
    for i in range(K):
        print("manual_means[i]:", manual_means[i])
        print("manual_covs[i]:", manual_covs[i])
        draw_ellipse(manual_means[i], manual_covs[i], color='red', alpha=0.5, linewidth=5,
                     label='Manual GMM' if i == 0 else None)

    # 绘制scikit-learn实现的结果
    for i in range(sklearn_model.means_.shape[0]):
        print("sklearn_model.means_[i]:", sklearn_model.means_[i])
        print("sklearn_model.covariances_[i]:", sklearn_model.covariances_[i])
        draw_ellipse(sklearn_model.means_[i], sklearn_model.covariances_[i], color='blue', alpha=0.8,
                     label='scikit-learn GMM' if i == 0 else None)

    plt.title(f'Iteration {iteration} - Comparison of Manual and scikit-learn GMM')
    plt.legend()
    plt.show()


# 假设这些是手动实现的结果
manual_means = means  # 你手动计算的均值
manual_covs = covariances  # 你手动计算的协方差矩阵

# scikit-learn 模型已经训练完成
sklearn_model = gmm_vb  # gmm_vb 是之前训练的 BayesianGaussianMixture 模型

# 调用 plot_comparison 函数
plot_comparison(data, manual_means, manual_covs, sklearn_model, iteration=100)
