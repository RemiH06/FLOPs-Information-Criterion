from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from flopscope import estimate_sklearn_fit, estimate_sklearn_predict, matmul_flops

def main():
    X, y = make_regression(n_samples=5000, n_features=40, random_state=0)
    m = LinearRegression()
    fit_flops = estimate_sklearn_fit(m, X, y)
    m.fit(X, y)
    pred_flops = estimate_sklearn_predict(m, X)
    print({"fit_flops": fit_flops, "predict_flops": pred_flops, "total": fit_flops + pred_flops})
    print({"gemm_flops": matmul_flops(2000, 1000, 256)})

if __name__ == "__main__":
    main()
