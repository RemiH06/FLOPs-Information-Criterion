#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "flopscope/estimators.hpp"

namespace py = pybind11;
using namespace flopscope;

PYBIND11_MODULE(_core, m) {
    m.doc() = "FLOPs estimators (C++ core)";
    m.def("matmul_flops", &matmul_flops, "Approximate FLOPs for dense matrix multiply",
          py::arg("m"), py::arg("k"), py::arg("n"));
    m.def("linear_regression_fit", &linear_regression_fit, py::arg("n"), py::arg("d"),
          py::arg("fit_intercept")=true, py::arg("method")="qr");
    m.def("linear_regression_predict", &linear_regression_predict, py::arg("n"), py::arg("d"));
    m.def("ridge_fit", &ridge_fit, py::arg("n"), py::arg("d"), py::arg("fit_intercept")=true);
    m.def("logreg_fit", &logreg_fit, py::arg("n"), py::arg("d"), py::arg("iters")=100, py::arg("fit_intercept")=true);
    m.def("logreg_predict", &logreg_predict, py::arg("n"), py::arg("d"), py::arg("fit_intercept")=true);
}
