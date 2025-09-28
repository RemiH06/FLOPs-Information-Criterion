from __future__ import annotations
from typing import Any, Tuple, List
import math

# Intentamos importar el core en C++; si no está, seguimos con las fórmulas Python.
try:
    from . import _core
except Exception:
    _core = None  # fallback puro Python

# -----------------------------
# Utils (mismas fórmulas que C++)
# -----------------------------
def _has(name: str) -> bool:
    return (_core is not None) and hasattr(_core, name)

def sq(x: int) -> int:
    return int(x) * int(x)

def cube(x: int) -> int:
    x = int(x); return x * x * x

def matmul_flops(m: int, k: int, n: int) -> int:
    if _has("matmul_flops"): return _core.matmul_flops(m, k, n)
    return 2 * int(m) * int(k) * int(n)

# -----------------------------
# Modelos lineales / GLM
# -----------------------------
def linear_regression_fit(n: int, d: int, fit_intercept: bool=True, method: str="qr") -> int:
    if _has("linear_regression_fit"): return _core.linear_regression_fit(n, d, fit_intercept, method)
    if fit_intercept: d += 1
    if method == "normal":
        return n*d*d + cube(d) + n*d
    return 2 * n * d * d

def linear_regression_predict(n: int, d: int) -> int:
    if _has("linear_regression_predict"): return _core.linear_regression_predict(n, d)
    return 2 * n * d

def ridge_fit(n: int, d: int, fit_intercept: bool=True) -> int:
    if _has("ridge_fit"): return _core.ridge_fit(n, d, fit_intercept)
    if fit_intercept: d += 1
    return n*d*d + cube(d) + n*d

def lasso_fit(n: int, d: int, iters: int=100, fit_intercept: bool=True) -> int:
    if _has("lasso_fit"): return _core.lasso_fit(n, d, iters, fit_intercept)
    if fit_intercept: d += 1
    return iters * (n * d)

def elasticnet_fit(n: int, d: int, iters: int=100, fit_intercept: bool=True) -> int:
    if _has("elasticnet_fit"): return _core.elasticnet_fit(n, d, iters, fit_intercept)
    if fit_intercept: d += 1
    return iters * (3 * n * d // 2)

def logreg_fit(n: int, d: int, iters: int=100, fit_intercept: bool=True) -> int:
    if _has("logreg_fit"): return _core.logreg_fit(n, d, iters, fit_intercept)
    if fit_intercept: d += 1
    return iters * (2 * n * d + d * d)

def logreg_predict(n: int, d: int, fit_intercept: bool=True) -> int:
    if _has("logreg_predict"): return _core.logreg_predict(n, d, fit_intercept)
    if fit_intercept: d += 1
    return 2 * n * d + n

def poisson_regression_fit(n: int, d: int, iters: int=50, fit_intercept: bool=True) -> int:
    if _has("poisson_regression_fit"): return _core.poisson_regression_fit(n, d, iters, fit_intercept)
    if fit_intercept: d += 1
    return iters * (2 * n * d + d * d)

def huber_regression_fit(n: int, d: int, iters: int=50, fit_intercept: bool=True) -> int:
    if _has("huber_regression_fit"): return _core.huber_regression_fit(n, d, iters, fit_intercept)
    if fit_intercept: d += 1
    return iters * (2 * n * d + (d * d) // 2)

# -----------------------------
# SVM / SVR / Kernel methods
# -----------------------------
def svm_linear_fit(n: int, d: int, iters: int=50, fit_intercept: bool=True) -> int:
    if _has("svm_linear_fit"): return _core.svm_linear_fit(n, d, iters, fit_intercept)
    if fit_intercept: d += 1
    return iters * (n * d)

def svm_linear_predict(n: int, d: int, fit_intercept: bool=True) -> int:
    if _has("svm_linear_predict"): return _core.svm_linear_predict(n, d, fit_intercept)
    if fit_intercept: d += 1
    return 2 * n * d

def svm_kernel_fit(n: int, d: int, iters: int=100) -> int:
    if _has("svm_kernel_fit"): return _core.svm_kernel_fit(n, d, iters)
    return n*n*d + iters * n * n

def svm_kernel_predict(n_pred: int, sv: int, d: int) -> int:
    if _has("svm_kernel_predict"): return _core.svm_kernel_predict(n_pred, sv, d)
    return 2 * n_pred * sv * d

def svr_fit(n: int, d: int, iters: int=100, linear: bool=True) -> int:
    if _has("svr_fit"): return _core.svr_fit(n, d, iters, linear)
    return iters * (n * d) if linear else (n*n*d + iters * n * n)

def svr_predict(n_pred: int, eff: int, d: int, linear: bool=True) -> int:
    if _has("svr_predict"): return _core.svr_predict(n_pred, eff, d, linear)
    return 2 * n_pred * d if linear else 2 * n_pred * eff * d

def kernel_ridge_fit(n: int) -> int:
    if _has("kernel_ridge_fit"): return _core.kernel_ridge_fit(n)
    return cube(n)

# -----------------------------
# Vecinos / Naive Bayes
# -----------------------------
def knn_fit(n: int, d: int) -> int:
    if _has("knn_fit"): return _core.knn_fit(n, d)
    return 0

def knn_predict(n_train: int, d: int, n_query: int) -> int:
    if _has("knn_predict"): return _core.knn_predict(n_train, d, n_query)
    return 2 * n_query * n_train * d

def naive_bayes_gaussian_fit(n: int, d: int, classes: int) -> int:
    if _has("naive_bayes_gaussian_fit"): return _core.naive_bayes_gaussian_fit(n, d, classes)
    return 2 * n * d + classes * d

def naive_bayes_gaussian_predict(n: int, d: int, classes: int) -> int:
    if _has("naive_bayes_gaussian_predict"): return _core.naive_bayes_gaussian_predict(n, d, classes)
    return 2 * n * d * classes

# -----------------------------
# Árboles / Ensambles
# -----------------------------
def decision_tree_fit(n: int, d: int, depth: int=10) -> int:
    if _has("decision_tree_fit"): return _core.decision_tree_fit(n, d, depth)
    return depth * n * d

def decision_tree_predict(n: int, depth: int=10) -> int:
    if _has("decision_tree_predict"): return _core.decision_tree_predict(n, depth)
    return n * depth

def random_forest_fit(n: int, d: int, trees: int=100, depth: int=10) -> int:
    if _has("random_forest_fit"): return _core.random_forest_fit(n, d, trees, depth)
    return trees * decision_tree_fit(n, d, depth)

def random_forest_predict(n: int, trees: int=100, depth: int=10) -> int:
    if _has("random_forest_predict"): return _core.random_forest_predict(n, trees, depth)
    return trees * decision_tree_predict(n, depth)

def extra_trees_fit(n: int, d: int, trees: int=100, depth: int=10) -> int:
    if _has("extra_trees_fit"): return _core.extra_trees_fit(n, d, trees, depth)
    return trees * decision_tree_fit(n, d, depth)

def gradient_boosting_fit(n: int, d: int, trees: int=100, depth: int=3) -> int:
    if _has("gradient_boosting_fit"): return _core.gradient_boosting_fit(n, d, trees, depth)
    return trees * decision_tree_fit(n, d, depth)

def gradient_boosting_predict(n: int, trees: int=100, depth: int=3) -> int:
    if _has("gradient_boosting_predict"): return _core.gradient_boosting_predict(n, trees, depth)
    return trees * decision_tree_predict(n, depth)

def xgboost_fit(n: int, d: int, rounds: int=200, depth: int=6) -> int:
    if _has("xgboost_fit"): return _core.xgboost_fit(n, d, rounds, depth)
    return rounds * decision_tree_fit(n, d, depth)

def xgboost_predict(n: int, rounds: int=200, depth: int=6) -> int:
    if _has("xgboost_predict"): return _core.xgboost_predict(n, rounds, depth)
    return rounds * decision_tree_predict(n, depth)

def lightgbm_fit(n: int, d: int, rounds: int=200, depth: int=8) -> int:
    if _has("lightgbm_fit"): return _core.lightgbm_fit(n, d, rounds, depth)
    return rounds * decision_tree_fit(n, d, depth)

def lightgbm_predict(n: int, rounds: int=200, depth: int=8) -> int:
    if _has("lightgbm_predict"): return _core.lightgbm_predict(n, rounds, depth)
    return rounds * decision_tree_predict(n, depth)

def catboost_fit(n: int, d: int, rounds: int=200, depth: int=6) -> int:
    if _has("catboost_fit"): return _core.catboost_fit(n, d, rounds, depth)
    return rounds * decision_tree_fit(n, d, depth)

# -----------------------------
# Reducción de dimensión
# -----------------------------
def pca_fit(n: int, d: int, k: int) -> int:
    if _has("pca_fit"): return _core.pca_fit(n, d, k)
    return n*d*k + d*k*k

def pca_transform(n: int, d: int, k: int) -> int:
    if _has("pca_transform"): return _core.pca_transform(n, d, k)
    return matmul_flops(n, d, k)

def lda_fit(n: int, d: int, classes: int) -> int:
    if _has("lda_fit"): return _core.lda_fit(n, d, classes)
    return n*d + cube(d)

def qda_fit(n: int, d: int, classes: int) -> int:
    if _has("qda_fit"): return _core.qda_fit(n, d, classes)
    return n*d*d + classes * cube(d)

def svd_fit(n: int, d: int) -> int:
    if _has("svd_fit"): return _core.svd_fit(n, d)
    return n*d*d if n >= d else d*n*n

def nmf_fit(n: int, d: int, k: int, iters: int=200) -> int:
    if _has("nmf_fit"): return _core.nmf_fit(n, d, k, iters)
    return iters * 2 * n * d * k

# -----------------------------
# Clustering / Mixturas
# -----------------------------
def kmeans_fit(n: int, d: int, k: int, iters: int=100) -> int:
    if _has("kmeans_fit"): return _core.kmeans_fit(n, d, k, iters)
    return iters * 2 * n * d * k

def kmeans_predict(n: int, d: int, k: int) -> int:
    if _has("kmeans_predict"): return _core.kmeans_predict(n, d, k)
    return 2 * n * d * k

def gmm_fit(n: int, d: int, k: int, iters: int=50) -> int:
    if _has("gmm_fit"): return _core.gmm_fit(n, d, k, iters)
    return iters * (2 * n * k * d + k * d * d)

def gmm_predict(n: int, d: int, k: int) -> int:
    if _has("gmm_predict"): return _core.gmm_predict(n, d, k)
    return n * k * (d + 1)

def dbscan_fit(n: int, d: int) -> int:
    if _has("dbscan_fit"): return _core.dbscan_fit(n, d)
    return n * n * d

def spectral_clustering_fit(n: int, k: int) -> int:
    if _has("spectral_clustering_fit"): return _core.spectral_clustering_fit(n, k)
    return n * n * k

def agglomerative_clustering_fit(n: int, d: int) -> int:
    if _has("agglomerative_clustering_fit"): return _core.agglomerative_clustering_fit(n, d)
    return n * n * d + n * n * int(math.log2(max(2, n)))

def pls_fit(n: int, d: int, t: int=10) -> int:
    if _has("pls_fit"): return _core.pls_fit(n, d, t)
    return t * n * d

# -----------------------------
# Series de tiempo
# -----------------------------
def arima_fit(T: int, p: int, d_: int, q: int, iters: int=50) -> int:
    if _has("arima_fit"): return _core.arima_fit(T, p, d_, q, iters)
    r = p + q
    return iters * T * r * r

def arima_predict(H: int, p: int, q: int) -> int:
    if _has("arima_predict"): return _core.arima_predict(H, p, q)
    return H * (p + q)

# -----------------------------
# Redes (MLP/RNN/Conv/Transformer)
# -----------------------------
def mlp_fit(n: int, in_d: int, hidden: int, out_d: int, iters: int=10) -> int:
    if _has("mlp_fit"): return _core.mlp_fit(n, in_d, hidden, out_d, iters)
    forward = n * (in_d*hidden + hidden*out_d)
    return iters * 2 * forward

def mlp_predict(n: int, in_d: int, hidden: int, out_d: int) -> int:
    if _has("mlp_predict"): return _core.mlp_predict(n, in_d, hidden, out_d)
    return n * (in_d*hidden + hidden*out_d)

def rnn_fit(n: int, T: int, d_in: int, h: int, iters: int=5) -> int:
    if _has("rnn_fit"): return _core.rnn_fit(n, T, d_in, h, iters)
    per_step = n * (d_in*h + h*h)
    return iters * 2 * T * per_step

def lstm_fit(n: int, T: int, d_in: int, h: int, iters: int=5) -> int:
    if _has("lstm_fit"): return _core.lstm_fit(n, T, d_in, h, iters)
    per_step = 4 * n * (d_in*h + h*h)
    return iters * 2 * T * per_step

def gru_fit(n: int, T: int, d_in: int, h: int, iters: int=5) -> int:
    if _has("gru_fit"): return _core.gru_fit(n, T, d_in, h, iters)
    per_step = 3 * n * (d_in*h + h*h)
    return iters * 2 * T * per_step

def conv2d_flops(N: int, H: int, W: int, Cin: int, Cout: int,
                 kH: int, kW: int, Ho: int, Wo: int) -> int:
    if _has("conv2d_flops"): return _core.conv2d_flops(N, H, W, Cin, Cout, kH, kW, Ho, Wo)
    return 2 * N * Ho * Wo * Cout * (kH * kW * Cin)

def attention_flops(N: int, L: int, d_model: int, heads: int) -> int:
    if _has("attention_flops"): return _core.attention_flops(N, L, d_model, heads)
    d_h = d_model // max(1, heads)
    proj = 3 * N * L * d_model * d_model
    attn = 2 * N * heads * L * L * d_h
    apply = 2 * N * heads * L * L * d_h
    out = N * L * d_model * d_model
    return proj + attn + apply + out

def transformer_ffn_flops(N: int, L: int, d_model: int, d_ff: int) -> int:
    if _has("transformer_ffn_flops"): return _core.transformer_ffn_flops(N, L, d_model, d_ff)
    return 2 * N * L * d_model * d_ff

def transformer_block_flops(N: int, L: int, d_model: int, heads: int, d_ff: int) -> int:
    if _has("transformer_block_flops"): return _core.transformer_block_flops(N, L, d_model, heads, d_ff)
    return attention_flops(N, L, d_model, heads) + transformer_ffn_flops(N, L, d_model, d_ff)

# -----------------------------
# Autoencoder / Misc
# -----------------------------
def autoencoder_fit(n: int, d_in: int, d_latent: int, iters: int=10) -> int:
    if _has("autoencoder_fit"): return _core.autoencoder_fit(n, d_in, d_latent, iters)
    fwd = n * (d_in*d_latent + d_latent*d_in)
    return 2 * iters * fwd

def autoencoder_encode(n: int, d_in: int, d_latent: int) -> int:
    if _has("autoencoder_encode"): return _core.autoencoder_encode(n, d_in, d_latent)
    return n * d_in * d_latent

def autoencoder_decode(n: int, d_latent: int, d_out: int) -> int:
    if _has("autoencoder_decode"): return _core.autoencoder_decode(n, d_latent, d_out)
    return n * d_latent * d_out

def sgd_epoch_flops(n_samples: int, d: int) -> int:
    if _has("sgd_epoch_flops"): return _core.sgd_epoch_flops(n_samples, d)
    return n_samples * d

# -----------------------------
# Manifold / embeddings
# -----------------------------
def tsne_fit(n: int, d: int, iters: int=1000, barnes_hut: bool=True) -> int:
    if _has("tsne_fit"): return _core.tsne_fit(n, d, iters, barnes_hut)
    if barnes_hut:
        return iters * n * int(math.log2(max(2, n))) + n * d
    return iters * n * n + n * d

def umap_fit(n: int, d: int, epochs: int=200, k: int=15) -> int:
    if _has("umap_fit"): return _core.umap_fit(n, d, epochs, k)
    return n * k * d + epochs * n * k

# -----------------------------
# Conveniencias sklearn (duck-typed)
# -----------------------------
def _shape(X) -> Tuple[int, int]:
    try:
        return int(X.shape[0]), int(X.shape[1])
    except Exception:
        return (len(X), 1)

def estimate_sklearn_fit(model: Any, X, y=None) -> int | None:
    n, d = _shape(X)
    name = model.__class__.__name__
    # Puedes mapear aquí lo que uses más:
    if name == "LinearRegression":
        return linear_regression_fit(n, d, getattr(model, "fit_intercept", True), "qr")
    if name == "Ridge":
        return ridge_fit(n, d, getattr(model, "fit_intercept", True))
    if name == "LogisticRegression":
        return logreg_fit(n, d, getattr(model, "max_iter", 100), getattr(model, "fit_intercept", True))
    # Agrega más mapeos si te sirven…
    return None

def estimate_sklearn_predict(model: Any, X) -> int | None:
    n, d = _shape(X)
    name = model.__class__.__name__
    if name in ("LinearRegression", "Ridge"):
        return linear_regression_predict(n, d)
    if name == "LogisticRegression":
        return logreg_predict(n, d, getattr(model, "fit_intercept", True))
    return None
