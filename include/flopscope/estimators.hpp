#pragma once
#include <cstdint>
#include <string>
#include <algorithm>

namespace flopscope {

// -----------------------------
// Utilidades
// -----------------------------
inline std::int64_t sq(std::int64_t x){ return x * x; }
inline std::int64_t cube(std::int64_t x){ return x * x * x; }

// MatMul: A (m×k) * B (k×n)
inline std::int64_t matmul_flops(std::int64_t m, std::int64_t k, std::int64_t n){
    return 2LL * m * k * n;
}

// -----------------------------
// Modelos lineales / GLM
// -----------------------------
inline std::int64_t linear_regression_fit(std::int64_t n, std::int64_t d, bool fit_intercept=true, const std::string& method="qr"){
    if(fit_intercept) d += 1;
    if(method == "normal"){          // (X^T X) + solve
        return n*d*d + cube(d) + n*d;
    }
    // QR (Householder) costo dominante ~ 2 n d^2  (aprox)
    return 2LL*n*d*d;
}

inline std::int64_t linear_regression_predict(std::int64_t n, std::int64_t d){
    return 2LL * n * d;
}

inline std::int64_t ridge_fit(std::int64_t n, std::int64_t d, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // normal equations con regularización (mismo costo)
    return n*d*d + cube(d) + n*d;
}

inline std::int64_t lasso_fit(std::int64_t n, std::int64_t d, std::int64_t iters=100, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // Coordinate Descent: por iteración ~ O(n d)
    return iters * (n*d);
}

inline std::int64_t elasticnet_fit(std::int64_t n, std::int64_t d, std::int64_t iters=100, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // Similar a LASSO CD (ligeramente mayor): ~ 1.5 n d por iteración
    return iters * (3LL*n*d/2);
}

inline std::int64_t logreg_fit(std::int64_t n, std::int64_t d, std::int64_t iters=100, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // iterativo: forward/backward + Hessian-lite ~ (2 n d + d^2) por iteración
    return iters * (2LL*n*d + d*d);
}

inline std::int64_t logreg_predict(std::int64_t n, std::int64_t d, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // dot + sigmoide
    return 2LL*n*d + n;
}

inline std::int64_t poisson_regression_fit(std::int64_t n, std::int64_t d, std::int64_t iters=50, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // IRLS-like: similar orden a logreg
    return iters * (2LL*n*d + d*d);
}

inline std::int64_t huber_regression_fit(std::int64_t n, std::int64_t d, std::int64_t iters=50, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // Gradiente/IRLS aproximado
    return iters * (2LL*n*d + d*d/2);
}

// -----------------------------
// SVM / SVR / Kernel methods
// -----------------------------
inline std::int64_t svm_linear_fit(std::int64_t n, std::int64_t d, std::int64_t iters=50, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    // Pegasos/SGD aproximado: ~ O(iters * n d)
    return iters * (n*d);
}

inline std::int64_t svm_linear_predict(std::int64_t n, std::int64_t d, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    return 2LL*n*d;
}

inline std::int64_t svm_kernel_fit(std::int64_t n, std::int64_t d, std::int64_t iters=100){
    // Construir kernel O(n^2 d), optimización SMO aproximada O(iters * n^2)
    return n*n*d + iters * n*n;
}

inline std::int64_t svm_kernel_predict(std::int64_t n_pred, std::int64_t sv, std::int64_t d){
    // Para RBF: costo ~ O(n_pred * sv * d) para distancias, o O(n_pred * sv) si precalculado
    return 2LL * n_pred * sv * d;
}

inline std::int64_t svr_fit(std::int64_t n, std::int64_t d, std::int64_t iters=100, bool linear=true){
    if(linear){
        // similar a SVM linear
        return iters * (n*d);
    }
    // kernel
    return n*n*d + iters * n*n;
}

inline std::int64_t svr_predict(std::int64_t n_pred, std::int64_t eff, std::int64_t d, bool linear=true){
    if(linear) return 2LL * n_pred * d;
    return 2LL * n_pred * eff * d; // eff ~ #SVs
}

inline std::int64_t kernel_ridge_fit(std::int64_t n){
    // Resolver (K + λI)α = y, con K (n×n): factorización ~ O(n^3), construir K ~ O(n^2 d) (omitir d aquí si ya dado K)
    return cube(n);
}

// -----------------------------
// Vecinos / Naive Bayes
// -----------------------------
inline std::int64_t knn_fit(std::int64_t /*n*/, std::int64_t /*d*/){
    // Entrenamiento ~ 0 (almacenamiento)
    return 0;
}

inline std::int64_t knn_predict(std::int64_t n_train, std::int64_t d, std::int64_t n_query){
    // Distancias: 2 * n_query * n_train * d
    return 2LL * n_query * n_train * d;
}

inline std::int64_t naive_bayes_gaussian_fit(std::int64_t n, std::int64_t d, std::int64_t classes){
    // medias y var por clase: O(n d) + agregados por clase
    return 2LL * n * d + classes * d;
}

inline std::int64_t naive_bayes_gaussian_predict(std::int64_t n, std::int64_t d, std::int64_t classes){
    // log-prob por clase: ~ O(n d classes)
    return 2LL * n * d * classes;
}

// -----------------------------
// Árboles y bosques
// -----------------------------
inline std::int64_t decision_tree_fit(std::int64_t n, std::int64_t d, std::int64_t depth=10){
    // aprox: O(n d log n) ~ depth * n d
    return depth * n * d;
}

inline std::int64_t decision_tree_predict(std::int64_t n, std::int64_t depth=10){
    // recorrer ~ depth comparaciones
    return n * depth;
}

inline std::int64_t random_forest_fit(std::int64_t n, std::int64_t d, std::int64_t trees=100, std::int64_t depth=10){
    return trees * decision_tree_fit(n, d, depth);
}

inline std::int64_t random_forest_predict(std::int64_t n, std::int64_t trees=100, std::int64_t depth=10){
    return trees * decision_tree_predict(n, depth);
}

inline std::int64_t extra_trees_fit(std::int64_t n, std::int64_t d, std::int64_t trees=100, std::int64_t depth=10){
    // similar RF (splits más baratos, aproximamos igual)
    return trees * decision_tree_fit(n, d, depth);
}

inline std::int64_t gradient_boosting_fit(std::int64_t n, std::int64_t d, std::int64_t trees=100, std::int64_t depth=3){
    // árboles pequeños secuenciales
    return trees * decision_tree_fit(n, d, depth);
}

inline std::int64_t gradient_boosting_predict(std::int64_t n, std::int64_t trees=100, std::int64_t depth=3){
    // suma de árboles débiles
    return trees * decision_tree_predict(n, depth);
}

// Boosting modernos (aprox similares)
inline std::int64_t xgboost_fit(std::int64_t n, std::int64_t d, std::int64_t rounds=200, std::int64_t depth=6){
    return rounds * decision_tree_fit(n, d, depth);
}
inline std::int64_t xgboost_predict(std::int64_t n, std::int64_t rounds=200, std::int64_t depth=6){
    return rounds * decision_tree_predict(n, depth);
}

inline std::int64_t lightgbm_fit(std::int64_t n, std::int64_t d, std::int64_t rounds=200, std::int64_t depth=8){
    return rounds * decision_tree_fit(n, d, depth);
}
inline std::int64_t lightgbm_predict(std::int64_t n, std::int64_t rounds=200, std::int64_t depth=8){
    return rounds * decision_tree_predict(n, depth);
}

inline std::int64_t catboost_fit(std::int64_t n, std::int64_t d, std::int64_t rounds=200, std::int64_t depth=6){
    return rounds * decision_tree_fit(n, d, depth);
}

// -----------------------------
// Reducción de dimensión / Proyecciones
// -----------------------------
inline std::int64_t pca_fit(std::int64_t n, std::int64_t d, std::int64_t k){
    // SVD (n×d) para k comps: ~ O(n d k) + O(d k^2)
    return n*d*k + d*k*k;
}

inline std::int64_t pca_transform(std::int64_t n, std::int64_t d, std::int64_t k){
    // proyección X (n×d) * W (d×k)
    return matmul_flops(n, d, k);
}

inline std::int64_t lda_fit(std::int64_t n, std::int64_t d, std::int64_t classes){
    // medias por clase + S_w, S_b y eigendecomp de d×d ~ O(n d + d^3)
    return n*d + cube(d);
}

inline std::int64_t qda_fit(std::int64_t n, std::int64_t d, std::int64_t classes){
    // cov por clase + inversas d×d: classes*(n_c d^2 + d^3)
    return n*d*d + classes * cube(d);
}

inline std::int64_t svd_fit(std::int64_t n, std::int64_t d){
    // SVD completa: ~ O(min(n d^2, d n^2))
    if(n >= d) return n * d * d;
    return d * n * n;
}

inline std::int64_t nmf_fit(std::int64_t n, std::int64_t d, std::int64_t k, std::int64_t iters=200){
    // multiplicative updates: por iter ~ 2 n d k
    return iters * 2LL * n * d * k;
}

// -----------------------------
// Clustering / Mixturas
// -----------------------------
inline std::int64_t kmeans_fit(std::int64_t n, std::int64_t d, std::int64_t k, std::int64_t iters=100){
    // asignación + centroides por iter: ~ 2 n d k
    return iters * 2LL * n * d * k;
}

inline std::int64_t kmeans_predict(std::int64_t n, std::int64_t d, std::int64_t k){
    // asignar al cluster más cercano
    return 2LL * n * d * k;
}

inline std::int64_t gmm_fit(std::int64_t n, std::int64_t d, std::int64_t k, std::int64_t iters=50){
    // EM: E-step ~ O(n k d), M-step ~ O(n k d + k d^2)
    return iters * (2LL*n*k*d + k*d*d);
}

inline std::int64_t gmm_predict(std::int64_t n, std::int64_t d, std::int64_t k){
    // responsabilidades: ~ O(n k d + n k) 
    return n*k*(d + 1);
}

inline std::int64_t dbscan_fit(std::int64_t n, std::int64_t d){
    // sin index: O(n^2 d); con index, menor. Usamos peor caso.
    return n*n*d;
}

inline std::int64_t spectral_clustering_fit(std::int64_t n, std::int64_t k){
    // eigen de Laplaciano n×n para k vectores: ~ O(n^2 k)
    return n*n*k;
}

inline std::int64_t agglomerative_clustering_fit(std::int64_t n, std::int64_t d){
    // matriz de distancias + merges: ~ O(n^2 d + n^2 log n)
    return n*n*d + n*n*static_cast<std::int64_t>(std::log2(std::max<std::int64_t>(2,n)));
}

// -----------------------------
// PLS / Proyección supervisada
// -----------------------------
inline std::int64_t pls_fit(std::int64_t n, std::int64_t d, std::int64_t t=10){
    // t componentes: por comp ~ O(n d)
    return t * n * d;
}

// -----------------------------
// Series de tiempo
// -----------------------------
inline std::int64_t arima_fit(std::int64_t T, std::int64_t p, std::int64_t d, std::int64_t q, std::int64_t iters=50){
    (void)d;
    // MLE por iter ~ O(T (p+q)^2) (aprox)
    std::int64_t r = p + q;
    return iters * T * r * r;
}

inline std::int64_t arima_predict(std::int64_t H, std::int64_t p, std::int64_t q){
    // proyección recursiva ~ O(H (p+q))
    return H * (p + q);
}

// -----------------------------
// MLP / Redes recurrentes / Convoluciones
// -----------------------------
inline std::int64_t mlp_fit(std::int64_t n, std::int64_t in_d, std::int64_t hidden, std::int64_t out_d, std::int64_t iters=10){
    // FC forward+backward ~ 2 * (n*(in_d*hidden + hidden*out_d)) por pasada
    std::int64_t forward = n * (in_d*hidden + hidden*out_d);
    return iters * 2LL * forward;
}

inline std::int64_t mlp_predict(std::int64_t n, std::int64_t in_d, std::int64_t hidden, std::int64_t out_d){
    return n * (in_d*hidden + hidden*out_d);
}

inline std::int64_t rnn_fit(std::int64_t n, std::int64_t T, std::int64_t d_in, std::int64_t h, std::int64_t iters=5){
    // simple RNN: por paso ~ O(n (d_in*h + h*h)), BPTT duplica
    std::int64_t per_step = n * (d_in*h + h*h);
    return iters * 2LL * T * per_step;
}

inline std::int64_t lstm_fit(std::int64_t n, std::int64_t T, std::int64_t d_in, std::int64_t h, std::int64_t iters=5){
    // LSTM: 4 puertas -> ~4x RNN
    std::int64_t per_step = 4LL * n * (d_in*h + h*h);
    return iters * 2LL * T * per_step;
}

inline std::int64_t gru_fit(std::int64_t n, std::int64_t T, std::int64_t d_in, std::int64_t h, std::int64_t iters=5){
    // GRU: ~3 puertas
    std::int64_t per_step = 3LL * n * (d_in*h + h*h);
    return iters * 2LL * T * per_step;
}

// Conv2D FLOPs: entrada (N, H, W, Cin), kernel (kH, kW, Cin, Cout), salida (N, Ho, Wo, Cout)
inline std::int64_t conv2d_flops(std::int64_t N, std::int64_t H, std::int64_t W,
                                 std::int64_t Cin, std::int64_t Cout,
                                 std::int64_t kH, std::int64_t kW,
                                 std::int64_t Ho, std::int64_t Wo){
    // MACs: N * Ho * Wo * Cout * (kH*kW*Cin); FLOPs ~ 2*MACs
    return 2LL * N * Ho * Wo * Cout * (kH * kW * Cin);
}

// -----------------------------
// Atención / Transformer
// -----------------------------
// Atención d_model con h cabezas, secuencia L, batch N
inline std::int64_t attention_flops(std::int64_t N, std::int64_t L, std::int64_t d_model, std::int64_t heads){
    std::int64_t d_h = d_model / std::max<std::int64_t>(1, heads);
    // Proyecciones Q,K,V: 3 * (N*L*d_model*d_model)
    std::int64_t proj = 3LL * N * L * d_model * d_model;
    // Scaled dot-product: Q*K^T ~ N*heads * L * L * d_h  (mult+add => 2x)
    std::int64_t attn = 2LL * N * heads * L * L * d_h;
    // Atención * V: N*heads * L * L * d_h (2x)
    std::int64_t apply = 2LL * N * heads * L * L * d_h;
    // Proyección de salida: N*L*d_model*d_model
    std::int64_t out = N * L * d_model * d_model;
    return proj + attn + apply + out;
}

inline std::int64_t transformer_ffn_flops(std::int64_t N, std::int64_t L, std::int64_t d_model, std::int64_t d_ff){
    // FFN: [d_model -> d_ff -> d_model], forward ~ N*L*(d_model*d_ff + d_ff*d_model)
    return 2LL * N * L * d_model * d_ff;
}

inline std::int64_t transformer_block_flops(std::int64_t N, std::int64_t L, std::int64_t d_model, std::int64_t heads, std::int64_t d_ff){
    // atención + FFN (ignoro LayerNorm y biases)
    return attention_flops(N, L, d_model, heads) + transformer_ffn_flops(N, L, d_model, d_ff);
}

// -----------------------------
// Autoencoder (lineal simple)
// -----------------------------
inline std::int64_t autoencoder_fit(std::int64_t n, std::int64_t d_in, std::int64_t d_latent, std::int64_t iters=10){
    // encoder d_in->d_latent, decoder d_latent->d_in, forward+backward
    std::int64_t fwd = n * (d_in*d_latent + d_latent*d_in);
    return 2LL * iters * fwd;
}
inline std::int64_t autoencoder_encode(std::int64_t n, std::int64_t d_in, std::int64_t d_latent){
    return n * d_in * d_latent;
}
inline std::int64_t autoencoder_decode(std::int64_t n, std::int64_t d_latent, std::int64_t d_out){
    return n * d_latent * d_out;
}

// -----------------------------
// Embeddings / SGD genérico por época
// -----------------------------
inline std::int64_t sgd_epoch_flops(std::int64_t n_samples, std::int64_t d){
    // una pasada: ~ O(n d)
    return n_samples * d;
}

// -----------------------------
// Manifold / Embeddings no lineales
// -----------------------------
inline std::int64_t tsne_fit(std::int64_t n, std::int64_t d, std::int64_t iters=1000, bool barnes_hut=true){
    if(barnes_hut){
        // O(n log n) por iter (aprox) + inicial proba O(n d)
        return iters * n * static_cast<std::int64_t>(std::log2(std::max<std::int64_t>(2,n))) + n*d;
    }
    // densa: O(n^2) por iter
    return iters * n * n + n*d;
}

inline std::int64_t umap_fit(std::int64_t n, std::int64_t d, std::int64_t epochs=200, std::int64_t k=15){
    // kNN graph ~ n k d; optimización por época ~ n k
    return n*k*d + epochs * n * k;
}

} // namespace flopscope
