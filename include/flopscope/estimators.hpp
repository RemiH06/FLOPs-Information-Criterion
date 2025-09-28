#pragma once
#include <cstdint>
#include <string>

namespace flopscope {

inline std::int64_t matmul_flops(std::int64_t m, std::int64_t k, std::int64_t n){
    return 2LL * m * k * n;
}

inline std::int64_t linear_regression_fit(std::int64_t n, std::int64_t d, bool fit_intercept=true, const std::string& method="qr"){
    if(fit_intercept) d += 1;
    if(method == "normal"){
        return n*d*d + d*d*d + n*d;
    }
    return n*d*d + d*d;
}

inline std::int64_t linear_regression_predict(std::int64_t n, std::int64_t d){
    return 2LL * n * d;
}

inline std::int64_t ridge_fit(std::int64_t n, std::int64_t d, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    return n*d*d + d*d*d + n*d;
}

inline std::int64_t logreg_fit(std::int64_t n, std::int64_t d, std::int64_t iters=100, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    return iters * (2LL*n*d + d*d);
}

inline std::int64_t logreg_predict(std::int64_t n, std::int64_t d, bool fit_intercept=true){
    if(fit_intercept) d += 1;
    return 2LL*n*d + n;
}

} // namespace flopscope
