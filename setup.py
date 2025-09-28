from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import os
import sys
import platform

# Detectar el sistema operativo
system = platform.system()

# Configuración específica del compilador
extra_compile_args = ['-std=c++17', '-O3']
extra_link_args = []

if system == "Windows":
    extra_compile_args.extend(['/std:c++17', '/O2'])
    extra_link_args.extend([])
elif system == "Darwin":  # macOS
    extra_compile_args.extend(['-stdlib=libc++'])
else:  # Linux
    extra_compile_args.extend(['-fPIC'])

# Definir la extensión
ext_modules = [
    Pybind11Extension(
        "flop_counter_cpp",
        sources=[
            "src/flop_counter.cpp",
            "src/pybind_interface.cpp"
        ],
        include_dirs=[
            "include",
            pybind11.get_include()
        ],
        language='c++',
        cxx_std=17,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args
    ),
]

setup(
    name="universal-flop-counter",
    version="0.1.0",
    author="Tu Nombre",
    author_email="tu.email@ejemplo.com",
    description="Contador universal de FLOPs para cualquier modelo de ML en Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tu-usuario/universal-flop-counter",
    packages=["flop_counter"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    extras_require={
        "torch": ["torch>=1.8.0"],
        "tensorflow": ["tensorflow>=2.4.0"],
        "all": ["torch>=1.8.0", "tensorflow>=2.4.0"],
    },
    include_package_data=True,
    zip_safe=False,
)