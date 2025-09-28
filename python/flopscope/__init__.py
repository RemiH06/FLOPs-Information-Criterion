from .api import (
    # utils
    matmul_flops, sq, cube,

    # lineales / GLM
    linear_regression_fit, linear_regression_predict, ridge_fit,
    lasso_fit, elasticnet_fit, logreg_fit, logreg_predict,
    poisson_regression_fit, huber_regression_fit,

    # SVM / kernel
    svm_linear_fit, svm_linear_predict, svm_kernel_fit, svm_kernel_predict,
    svr_fit, svr_predict, kernel_ridge_fit,

    # vecinos / NB
    knn_fit, knn_predict, naive_bayes_gaussian_fit, naive_bayes_gaussian_predict,

    # árboles / ensambles
    decision_tree_fit, decision_tree_predict,
    random_forest_fit, random_forest_predict,
    extra_trees_fit, gradient_boosting_fit, gradient_boosting_predict,
    xgboost_fit, xgboost_predict, lightgbm_fit, lightgbm_predict, catboost_fit,

    # reducción / proyecciones
    pca_fit, pca_transform, lda_fit, qda_fit, svd_fit, nmf_fit, pls_fit,

    # clustering / mixtures
    kmeans_fit, kmeans_predict, gmm_fit, gmm_predict,
    dbscan_fit, spectral_clustering_fit, agglomerative_clustering_fit,

    # series de tiempo
    arima_fit, arima_predict,

    # redes
    mlp_fit, mlp_predict, rnn_fit, lstm_fit, gru_fit,
    conv2d_flops, attention_flops, transformer_ffn_flops, transformer_block_flops,
    autoencoder_fit, autoencoder_encode, autoencoder_decode,
    sgd_epoch_flops, tsne_fit, umap_fit,

    # conveniencias sklearn
    estimate_sklearn_fit, estimate_sklearn_predict,
)

# (Opcional) expón el probe BLAS si lo estás usando en este paquete:
try:
    from .blas_probe import count_blas_flops, last_blas_flops, last_blas_result
    __all__ = [*list(globals().keys())]  # exporta todo lo anterior + probe
except Exception:
    __all__ = [*list(globals().keys())]
