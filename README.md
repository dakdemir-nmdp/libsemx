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
