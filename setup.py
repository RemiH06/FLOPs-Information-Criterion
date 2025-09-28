from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "flopscope._core",
        ["src/bindings.cpp"],
        include_dirs=[pybind11.get_include(), "include"],
        language="c++",
        extra_compile_args=["-std=c++17"]
    )
]

setup(
    packages=["flopscope"],
    package_dir={"flopscope": "python/flopscope"},
    ext_modules=ext_modules,
)
