from __future__ import annotations
from typing import Any, Tuple
from . import _core

def _shape(X) -> Tuple[int,int]:
    try:
        n, d = int(X.shape[0]), int(X.shape[1])
    except Exception:
        n, d = (len(X), 1)
    return n, d

matmul_flops = _core.matmul_flops
linear_regression_fit = _core.linear_regression_fit
linear_regression_predict = _core.linear_regression_predict
ridge_fit = _core.ridge_fit
logreg_fit = _core.logreg_fit
logreg_predict = _core.logreg_predict

def estimate_sklearn_fit(model, X, y=None):
    n, d = _shape(X)
    name = model.__class__.__name__
    if name == "LinearRegression":
        return _core.linear_regression_fit(n, d, getattr(model, "fit_intercept", True))
    if name == "Ridge":
        return _core.ridge_fit(n, d, getattr(model, "fit_intercept", True))
    if name == "LogisticRegression":
        iters = getattr(model, "max_iter", 100)
        return _core.logreg_fit(n, d, iters, getattr(model, "fit_intercept", True))
    return None

def estimate_sklearn_predict(model, X):
    n, d = _shape(X)
    name = model.__class__.__name__
    if name in ("LinearRegression", "Ridge"):
        return _core.linear_regression_predict(n, d)
    if name == "LogisticRegression":
        return _core.logreg_predict(n, d, getattr(model, "fit_intercept", True))
    return None
