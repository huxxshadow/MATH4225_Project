import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
from scipy.special import digamma, gammaln, logsumexp, betaln
from sklearn.mixture import BayesianGaussianMixture

# 清除所有内存
plt.close('all')

# 数据生成部分
# 均值列表
means = [
    [0, 15],   # 第1个高斯分布的均值
    [150, 180],  # 第2个高斯分布的均值
    [20, 30],  # 第3个高斯分布的均值
    [110, 150],  # 第4个高斯分布的均值
    [100, 30],  # 第5个高斯分布的均值
    [12, 450]   # 第6个高斯分布的均值
]

# 协方差矩阵列表
covariances = [
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
    for mean, covariance in zip(means, covariances)
])

true_means = np.array(means)
true_covariances = np.array(covariances)

# 使用scikit-learn中的变分贝叶斯高斯混合模型 作为对照
gmm_vb = BayesianGaussianMixture(tol=1e-7, n_components=6, covariance_type='full', random_state=2,
                                 init_params='kmeans',
                                 weight_concentration_prior_type="dirichlet_process",
                                 verbose=0, verbose_interval=10)
gmm_vb.fit(data)

# 定义绘制椭圆的函数
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

# 定义运行变分贝叶斯高斯混合模型的函数
def run_vbgmm(data, resp_init, K, num_iterations=1000, tol=1e-3):
    N, D = data.shape
    resp = resp_init.copy()

    alpha_0 = 1 / K  # Dirichlet先验参数
    beta_0 = 1
    v_0 = D
    reg_covar = 1e-7
    mean_prior = np.mean(data, axis=0)
    covariance_prior = np.atleast_2d(np.cov(data.T))

    lower_bound = -np.inf
    max_lower_bound = -np.inf

    # 初始化参数
    sum_resp = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    old_means = np.dot(resp.T, data) / sum_resp[:, np.newaxis]
    old_covariances = np.empty((K, D, D))
    for k in range(K):
        diff = data - old_means[k]
        old_covariances[k] = np.dot(resp[:, k] * diff.T, diff) / sum_resp[k]
        old_covariances[k].flat[:: D + 1] += reg_covar

    alpha = (1.0 + sum_resp, (alpha_0 + np.hstack((np.cumsum(sum_resp[::-1])[-2::-1], 0))), )
    beta = beta_0 + sum_resp
    means = (beta_0 * mean_prior + sum_resp[:, np.newaxis] * old_means) / beta[:, np.newaxis]
    v = v_0 + sum_resp
    covariances = np.empty((K, D, D))
    for k in range(K):
        diff = old_means[k] - mean_prior
        covariances[k] = (
                covariance_prior
                + sum_resp[k] * old_covariances[k]
                + sum_resp[k] * beta_0 / beta[k] * np.outer(diff, diff)
        )
    covariances /= v[:, np.newaxis, np.newaxis]
    precisions_cholesky = np.array([
        linalg.cholesky(np.linalg.inv(covariances[k]), lower=True)
        for k in range(K)
    ])

    num_iterations = num_iterations
    best_params = None
    best_n_iter = 0

    for iteration in range(1, num_iterations + 1):
        # 计算 log_rho
        log_det = np.sum(np.log(precisions_cholesky.reshape(K, -1)[:, ::D + 1]), axis=1)
        log_prob = np.array([
            -0.5 * np.sum((np.dot(data - means[k], precisions_cholesky[k]) ** 2), axis=1)
            + log_det[k]
            for k in range(K)
        ]).T
        log_gauss = log_prob - 0.5 * D * np.log(2 * np.pi) - 0.5 * D * np.log(v_0)

        log_lambda = D * np.log(2) + np.sum(
            digamma(0.5 * (v[:, np.newaxis] - np.arange(D))), axis=1
        )
        log_det_precisions_chol = log_det - 0.5 * D * np.log(v)

        log_wishart = -(
                v * log_det_precisions_chol
                + v * D * 0.5 * np.log(2)
                + np.sum(gammaln(0.5 * (v - np.arange(D)[:, np.newaxis])), axis=0)
        )

        log_norm_alpha = np.sum(-np.sum(
            betaln(alpha[0], alpha[1])
        ))

        digamma_sum = digamma(alpha[0] + alpha[1])
        digamma_a = digamma(alpha[0])
        digamma_b = digamma(alpha[1])
        log_alpha = digamma_a - digamma_sum + np.hstack((0, np.cumsum(digamma_b - digamma_sum)[:-1]))

        weighted_log_prob = log_gauss + 0.5 * (log_lambda - D / beta) + log_alpha
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)

        log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        resp = np.exp(log_resp)

        # 计算 lower_bound
        lower_bound = (
                -np.sum(resp * log_resp)
                - np.sum(log_wishart)
                - log_norm_alpha
                - 0.5 * D * np.sum(np.log(beta))
        )

        print(lower_bound - max_lower_bound)
        if abs(lower_bound - max_lower_bound) < tol:
            print(f"Converged at iteration {iteration}")
            break

        if (lower_bound > max_lower_bound or max_lower_bound == -np.inf) and iteration > 1:
            max_lower_bound = lower_bound
            best_n_iter = iteration
            best_params = (means.copy(), covariances.copy(), precisions_cholesky.copy(),
                           alpha, beta.copy(), v.copy(), lower_bound)



        # M 步参数更新
        sum_resp = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        old_means = np.dot(resp.T, data) / sum_resp[:, np.newaxis]
        old_covariances = np.empty((K, D, D))
        for k in range(K):
            diff = data - old_means[k]
            old_covariances[k] = np.dot(resp[:, k] * diff.T, diff) / sum_resp[k]
            old_covariances[k].flat[:: D + 1] += reg_covar

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
        covariances /= v[:, np.newaxis, np.newaxis]

        precisions_cholesky = np.array([
            linalg.cholesky(np.linalg.inv(covariances[k]), lower=True)
            for k in range(K)
        ])

    if best_params is not None:
        means, covariances, precisions_cholesky, alpha, beta, v, lower_bound = best_params
    else:
        print("Did not converge")
    return means, covariances, precisions_cholesky, alpha, beta, v, lower_bound

# 设置高斯混合模型的参数
K = 6  # 高斯分布的个数 components
N, D = data.shape  # n_samples, n_features
print("数据的形状:", data.shape)

# 方式1：使用KMeans进行初始化
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
resp_kmeans = np.zeros((N, K))
resp_kmeans[np.arange(N), kmeans.labels_] = 1

# 运行VB-GMM算法（KMeans初始化）
means_kmeans, covariances_kmeans, precisions_cholesky_kmeans, \
    alpha_kmeans, beta_kmeans, v_kmeans, lower_bound_kmeans = \
    run_vbgmm(data, resp_kmeans, K, num_iterations=1000)

# 方式2：使用随机初始化
resp_random = np.zeros((N, K))  # responsibilities Z的后验概率期望
indices = np.random.choice(N, size=K, replace=False)
resp_random[indices, np.arange(K)] = 1  # 初始化resp矩阵，每个数据点属于每个簇的概率

# 运行VB-GMM算法（随机初始化）
means_random, covariances_random, precisions_cholesky_random, \
    alpha_random, beta_random, v_random, lower_bound_random = \
    run_vbgmm(data, resp_random, K, num_iterations=1000)

# 定义绘图函数
def plot_comparison(data, results_list, titles,colors):
    """Plot the GMM results on subplots."""
    n_plots = len(results_list)
    fig, axs = plt.subplots(1, n_plots, figsize=(8 * n_plots, 8))
    if n_plots == 1:
        axs = [axs]
    for idx, (means, covs) in enumerate(results_list):
        ax = axs[idx]
        ax.scatter(data[:, 0], data[:, 1], color='green', alpha=0.3, label='Data')
        for i in range(means.shape[0]):
            draw_ellipse(means[i], covs[i], ax=ax, color=colors[idx], alpha=0.5, linewidth=6,
                         label='Estimated GMM' if i == 0 else None)
        # 绘制真实的高斯分布椭圆
        for i, (mean, cov) in enumerate(zip(true_means, true_covariances)):
            draw_ellipse(mean, cov, ax=ax, color='black', alpha=0.8, linewidth=2,
                         label='True Gaussian Distribution' if i == 0 else None)
        ax.set_title(titles[idx])
        ax.legend()
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.grid(True)
    plt.show()

# 调用绘图函数
results_list = [
    (means_kmeans, covariances_kmeans),
    (means_random, covariances_random),
    (gmm_vb.means_, gmm_vb.covariances_)
]

titles = [
    'VB-GMM with KMeans initialization',
    'VB-GMM with Random initialization',
    'scikit-learn BayesianGaussianMixture'
]

colors=["red","blue","purple"]


plot_comparison(data, results_list, titles,colors)