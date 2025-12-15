# libsemx

**libsemx** is a high-performance, unified engine for Structural Equation Modeling (SEM) and Generalized Linear Mixed Models (GLMM), built with a modern C++20 core and providing idiomatic front-ends for Python and R.

## Features

*   **Unified Framework**: Seamlessly combines SEM (Latent Variables, Path Analysis) and Mixed Models (Random Intercepts/Slopes, Hierarchical structures).
*   **High Performance**: C++20 core with Eigen for linear algebra, supporting sparse matrices and optimized gradient computation.
*   **Flexible Covariance Structures**: Supports Unstructured, Diagonal, Compound Symmetry, AR(1), Toeplitz, Kronecker products, and user-defined kernels (e.g., for Genomic Selection).
*   **Diverse Outcome Families**: Gaussian, Binomial (Logit/Probit), Poisson, Negative Binomial, Gamma, Weibull, Exponential, Log-Normal, Log-Logistic, and Ordinal outcomes.
*   **Advanced Estimation**: Maximum Likelihood (ML), Restricted Maximum Likelihood (REML), and Laplace Approximation for non-Gaussian outcomes.
*   **Missing Data**: Full Information Maximum Likelihood (FIML) support.
*   **Multi-Group Analysis**: Support for multiple groups with parameter constraints.

## Installation

### Prerequisites

*   **C++ compiler** with C++20 support
*   **CMake** (for building the C++ core)
*   **Eigen library** (automatically fetched by CMake)
*   **pybind11** (for Python bindings, automatically fetched by CMake)
*   **Python ≥ 3.9** (for Python front-end)
*   **R** with Rcpp and RcppEigen (for R front-end)

### Python Installation

From the repository root:

```bash
# 1. Build the C++ library
mkdir -p build && cd build
cmake ../cpp
make
cd ..

# 2. Install the Python package (development mode)
cd python
pip install -e .
```

Or for a regular installation:

```bash
cd python
pip install .
```

### R Installation

From R:

```R
install.packages("Rpkg/semx", repos = NULL, type = "source")
```

Or from the command line:

```bash
R CMD INSTALL Rpkg/semx
```

## Structure

*   `cpp/`: Core C++ library and tests.
*   `python/`: Python bindings (`semx`) and package.
*   `Rpkg/`: R package (`semx`) and bindings.
*   `data/`: Example datasets.
*   `docs/`: Documentation and design blueprints.
*   `validation/`: Validation scripts comparing against `lavaan`, `lme4`, `sommer`, etc.

## Author

Deniz Akdemir

## License

MIT License
