#include "libsemx/covariance_structure.hpp"
#include "libsemx/scaled_fixed_covariance.hpp"
#include "libsemx/multi_kernel_covariance.hpp"
#include "libsemx/genomic_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <cctype>
#include <string_view>
#include <optional>

namespace libsemx {

namespace {
[[nodiscard]] std::size_t triangular_number(std::size_t n) {
    return n * (n + 1) / 2;
}

[[nodiscard]] double positive_to_correlation(double positive_value) {
    // Map (0, inf) -> (-1, 1)
    return (positive_value - 1.0) / (positive_value + 1.0);
}

[[nodiscard]] double positive_to_correlation_derivative(double positive_value) {
    const double denom = positive_value + 1.0;
    return 2.0 / (denom * denom);
}

std::string normalize_structure_id(const std::string& id) {
    std::string lowered;
    lowered.reserve(id.size());
    for (unsigned char ch : id) {
        if (ch == '-') {
            lowered.push_back('_');
        } else {
            lowered.push_back(static_cast<char>(std::tolower(ch)));
        }
    }
    return lowered;
}

std::optional<std::size_t> parse_factor_rank(const std::string& normalized) {
    auto parse_tail = [](std::string_view tail) -> std::optional<std::size_t> {
        if (tail.empty()) {
            return std::nullopt;
        }
        std::size_t idx = 0;
        while (idx < tail.size() && (tail[idx] == '_' || tail[idx] == '-' || tail[idx] == '(' || tail[idx] == 'q')) {
            ++idx;
        }
        if (idx >= tail.size()) {
            return std::nullopt;
        }
        std::size_t value = 0;
        for (; idx < tail.size(); ++idx) {
            const char ch = tail[idx];
            if (ch == ')' || ch == ' ') {
                break;
            }
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                return std::nullopt;
            }
            value = value * 10 + static_cast<std::size_t>(ch - '0');
        }
        if (value == 0) {
            return std::nullopt;
        }
        return value;
    };

    if (normalized == "factor_analytic" || normalized == "fa") {
        return 1;
    }
    if (normalized.rfind("fa", 0) == 0) {
        return parse_tail(normalized.substr(2));
    }
    constexpr std::string_view kPrefix = "factor_analytic";
    if (normalized.rfind(kPrefix, 0) == 0) {
        auto rank = parse_tail(normalized.substr(kPrefix.size()));
        if (rank) {
            return rank;
        }
        return 1;
    }
    return std::nullopt;
}

struct ToeplitzAutocovarianceData {
    std::vector<double> autocov;
    std::vector<std::vector<double>> autocov_grad;
};

ToeplitzAutocovarianceData compute_toeplitz_autocovariance(std::size_t dim,
                                                           const std::vector<double>& kappas,
                                                           bool need_gradients) {
    ToeplitzAutocovarianceData data;
    data.autocov.assign(dim, 0.0);
    if (dim == 0) {
        return data;
    }
    data.autocov[0] = 1.0;
    if (dim == 1) {
        return data;
    }

    const std::size_t order = dim - 1;
    std::vector<std::vector<double>> phi(dim, std::vector<double>(dim, 0.0));
    std::vector<std::vector<std::vector<double>>> phi_grad;
    if (need_gradients) {
        data.autocov_grad.assign(dim, std::vector<double>(order, 0.0));
        phi_grad.assign(order, std::vector<std::vector<double>>(dim, std::vector<double>(dim, 0.0)));
    }

    for (std::size_t k = 1; k < dim; ++k) {
        const double kappa = kappas[k - 1];
        phi[k][k] = kappa;
        for (std::size_t j = 1; j < k; ++j) {
            phi[k][j] = phi[k - 1][j] - kappa * phi[k - 1][k - j];
        }

        double r_k = 0.0;
        for (std::size_t j = 1; j <= k; ++j) {
            r_k += phi[k][j] * data.autocov[k - j];
        }
        data.autocov[k] = r_k;

        if (!need_gradients) {
            continue;
        }

        for (std::size_t l = 0; l < order; ++l) {
            phi_grad[l][k][k] = (l == (k - 1)) ? 1.0 : 0.0;
            for (std::size_t j = 1; j < k; ++j) {
                double partial = phi_grad[l][k - 1][j];
                if (l == (k - 1)) {
                    partial -= phi[k - 1][k - j];
                }
                partial -= kappa * phi_grad[l][k - 1][k - j];
                phi_grad[l][k][j] = partial;
            }

            double deriv = 0.0;
            for (std::size_t j = 1; j <= k; ++j) {
                deriv += phi_grad[l][k][j] * data.autocov[k - j];
                deriv += phi[k][j] * data.autocov_grad[k - j][l];
            }
            data.autocov_grad[k][l] = deriv;
        }
    }
    return data;
}
}  // namespace

CovarianceStructure::CovarianceStructure(std::size_t dimension, std::size_t parameter_count)
    : dimension_(dimension), parameter_count_(parameter_count) {
    if (dimension_ == 0) {
        throw std::invalid_argument("covariance dimension must be positive");
    }
    // parameter_count can be 0 for fixed structures
}

std::vector<double> CovarianceStructure::materialize(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    std::vector<double> matrix(dimension_ * dimension_, 0.0);
    fill_covariance(parameters, matrix);
    return matrix;
}

void CovarianceStructure::validate_parameters(const std::vector<double>& parameters) const {
    if (parameters.size() != parameter_count_) {
        throw std::invalid_argument("parameter count mismatch for covariance structure");
    }
}

std::vector<std::vector<double>> CovarianceStructure::parameter_gradients(const std::vector<double>& /*parameters*/) const {
    throw std::runtime_error("parameter_gradients not implemented for this covariance structure");
}

UnstructuredCovariance::UnstructuredCovariance(std::size_t dimension)
    : CovarianceStructure(dimension, triangular_number(dimension)) {}

void UnstructuredCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    std::vector<double> L(dim * dim, 0.0);
    std::size_t param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            L[row * dim + col] = parameters[param_index++];
        }
    }

    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            double sum = 0.0;
            const std::size_t limit = std::min(i, j);
            for (std::size_t k = 0; k <= limit; ++k) {
                sum += L[i * dim + k] * L[j * dim + k];
            }
            matrix[i * dim + j] = sum;
        }
    }
}

void UnstructuredCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    const std::size_t dim = dimension();
    std::size_t param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            const double value = parameters[param_index++];
            if (row == col && !(value > 0.0)) {
                throw std::invalid_argument("unstructured covariance requires positive diagonal elements");
            }
        }
    }
}

std::vector<std::vector<double>> UnstructuredCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t pc = parameter_count();
    std::vector<std::vector<double>> grads(pc, std::vector<double>(dim * dim, 0.0));

    std::vector<double> L(dim * dim, 0.0);
    std::size_t param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            L[row * dim + col] = parameters[param_index++];
        }
    }

    param_index = 0;
    for (std::size_t row = 0; row < dim; ++row) {
        for (std::size_t col = 0; col <= row; ++col) {
            auto& grad = grads[param_index++];
            for (std::size_t j = 0; j < dim; ++j) {
                double value = L[j * dim + col];
                grad[row * dim + j] += value;
                grad[j * dim + row] += value;
            }
        }
    }
    return grads;
}

DiagonalCovariance::DiagonalCovariance(std::size_t dimension)
    : CovarianceStructure(dimension, dimension) {}

void DiagonalCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    for (std::size_t i = 0; i < dim; ++i) {
        matrix[i * dim + i] = parameters[i];
    }
}

void DiagonalCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    for (double value : parameters) {
        if (!(value > 0.0)) {
            throw std::invalid_argument("diagonal covariance requires strictly positive entries");
        }
    }
}

std::vector<std::vector<double>> DiagonalCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t pc = parameter_count();
    std::vector<std::vector<double>> grads(pc, std::vector<double>(dim * dim, 0.0));

    for (std::size_t i = 0; i < dim; ++i) {
        // param i corresponds to (i, i)
        grads[i][i * dim + i] = 1.0;
    }
    return grads;
}

CompoundSymmetryCovariance::CompoundSymmetryCovariance(std::size_t dimension)
    : CovarianceStructure(dimension, 2) {}

void CompoundSymmetryCovariance::fill_covariance(const std::vector<double>& parameters,
                                                 std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    const double unique_variance = parameters[0];
    const double shared_variance = parameters[1];
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            matrix[i * dim + j] = (i == j) ? (unique_variance + shared_variance) : shared_variance;
        }
    }
}

void CompoundSymmetryCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    if (!(parameters[0] > 0.0)) {
        throw std::invalid_argument("compound symmetry requires positive unique variance");
    }
    if (!(parameters[1] > 0.0)) {
        throw std::invalid_argument("compound symmetry requires positive shared variance");
    }
}

std::vector<std::vector<double>> CompoundSymmetryCovariance::parameter_gradients(
    const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    std::vector<std::vector<double>> grads(2, std::vector<double>(dim * dim, 0.0));

    auto& unique_grad = grads[0];
    auto& shared_grad = grads[1];
    for (std::size_t i = 0; i < dim; ++i) {
        unique_grad[i * dim + i] = 1.0;
        shared_grad[i * dim + i] = 1.0;
        for (std::size_t j = i + 1; j < dim; ++j) {
            shared_grad[i * dim + j] = 1.0;
            shared_grad[j * dim + i] = 1.0;
        }
    }
    return grads;
}

AR1Covariance::AR1Covariance(std::size_t dimension)
    : CovarianceStructure(dimension, dimension > 1 ? 2 : 1) {}

void AR1Covariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    const double variance = parameters[0];
    const double rho = (parameter_count() > 1) ? positive_to_correlation(parameters[1]) : 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            const std::size_t dist = (i > j) ? (i - j) : (j - i);
            const double scale = (dist == 0) ? 1.0 : std::pow(rho, static_cast<double>(dist));
            matrix[i * dim + j] = variance * scale;
        }
    }
}

void AR1Covariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    if (!(parameters[0] > 0.0)) {
        throw std::invalid_argument("AR(1) variance must be positive");
    }
    if (parameter_count() > 1 && !(parameters[1] > 0.0)) {
        throw std::invalid_argument("AR(1) correlation parameter must be positive");
    }
}

std::vector<std::vector<double>> AR1Covariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t pc = parameter_count();
    std::vector<std::vector<double>> grads(pc, std::vector<double>(dim * dim, 0.0));

    const double variance = parameters[0];
    const double rho = (pc > 1) ? positive_to_correlation(parameters[1]) : 0.0;

    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            const std::size_t dist = (i > j) ? (i - j) : (j - i);
            const double base = (dist == 0) ? 1.0 : std::pow(rho, static_cast<double>(dist));
            grads[0][i * dim + j] = base;
            if (pc > 1 && dist > 0) {
                const double drho = positive_to_correlation_derivative(parameters[1]);
                const double pow_term = (dist == 1) ? 1.0 : std::pow(rho, static_cast<double>(dist - 1));
                grads[1][i * dim + j] = variance * dist * pow_term * drho;
            }
        }
    }
    return grads;
}

ToeplitzCovariance::ToeplitzCovariance(std::size_t dimension)
    : CovarianceStructure(dimension, dimension == 0 ? 0 : dimension) {}

void ToeplitzCovariance::fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    if (dim == 0) {
        return;
    }
    const double variance = parameters[0];
    std::vector<double> kappas;
    if (dim > 1) {
        kappas.reserve(dim - 1);
        for (std::size_t idx = 1; idx < parameters.size(); ++idx) {
            kappas.push_back(positive_to_correlation(parameters[idx]));
        }
    }
    auto moments = compute_toeplitz_autocovariance(dim, kappas, false);
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            const std::size_t dist = (i > j) ? (i - j) : (j - i);
            matrix[i * dim + j] = variance * moments.autocov[dist];
        }
    }
}

void ToeplitzCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    if (dimension() == 0) {
        throw std::invalid_argument("Toeplitz covariance requires positive dimension");
    }
    if (!(parameters[0] > 0.0)) {
        throw std::invalid_argument("Toeplitz covariance requires positive variance");
    }
    for (std::size_t idx = 1; idx < parameters.size(); ++idx) {
        if (!(parameters[idx] > 0.0)) {
            throw std::invalid_argument("Toeplitz correlation parameters must be positive");
        }
    }
}

std::vector<std::vector<double>> ToeplitzCovariance::parameter_gradients(const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t pc = parameter_count();
    std::vector<std::vector<double>> grads(pc, std::vector<double>(dim * dim, 0.0));

    const double variance = parameters[0];
    std::vector<double> kappas;
    std::vector<double> dkappa;
    if (dim > 1) {
        kappas.reserve(dim - 1);
        dkappa.reserve(dim - 1);
        for (std::size_t idx = 1; idx < parameters.size(); ++idx) {
            kappas.push_back(positive_to_correlation(parameters[idx]));
            dkappa.push_back(positive_to_correlation_derivative(parameters[idx]));
        }
    }

    auto data = compute_toeplitz_autocovariance(dim, kappas, true);

    auto& grad_variance = grads[0];
    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = 0; j < dim; ++j) {
            const std::size_t dist = (i > j) ? (i - j) : (j - i);
            grad_variance[i * dim + j] = data.autocov[dist];
        }
    }

    if (pc == 1) {
        return grads;
    }

    for (std::size_t param_idx = 1; param_idx < pc; ++param_idx) {
        auto& grad = grads[param_idx];
        const std::size_t lag_index = param_idx - 1;
        const double chain = dkappa[lag_index];
        for (std::size_t i = 0; i < dim; ++i) {
            for (std::size_t j = 0; j < dim; ++j) {
                const std::size_t dist = (i > j) ? (i - j) : (j - i);
                grad[i * dim + j] = variance * data.autocov_grad[dist][lag_index] * chain;
            }
        }
    }
    return grads;
}

FactorAnalyticCovariance::FactorAnalyticCovariance(std::size_t dimension, std::size_t rank)
    : CovarianceStructure(dimension, dimension * rank + dimension), rank_(rank) {
    if (rank_ == 0 || rank_ >= dimension) {
        throw std::invalid_argument("factor-analytic covariance requires 0 < rank < dimension");
    }
}

void FactorAnalyticCovariance::fill_covariance(const std::vector<double>& parameters,
                                               std::vector<double>& matrix) const {
    const std::size_t dim = dimension();
    const std::size_t loadings_count = dim * rank_;
    const double* loadings = parameters.data();
    const double* uniques = parameters.data() + loadings_count;

    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t j = i; j < dim; ++j) {
            double value = 0.0;
            for (std::size_t f = 0; f < rank_; ++f) {
                value += loadings[i * rank_ + f] * loadings[j * rank_ + f];
            }
            if (i == j) {
                value += uniques[i];
            }
            matrix[i * dim + j] = value;
            if (i != j) {
                matrix[j * dim + i] = value;
            }
        }
    }
}

void FactorAnalyticCovariance::validate_parameters(const std::vector<double>& parameters) const {
    CovarianceStructure::validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t loadings_count = dim * rank_;
    for (std::size_t i = 0; i < dim; ++i) {
        const double unique = parameters[loadings_count + i];
        if (!(unique > 0.0)) {
            throw std::invalid_argument("factor-analytic uniquenesses must be positive");
        }
    }
}

std::vector<std::vector<double>> FactorAnalyticCovariance::parameter_gradients(
    const std::vector<double>& parameters) const {
    validate_parameters(parameters);
    const std::size_t dim = dimension();
    const std::size_t loadings_count = dim * rank_;
    std::vector<std::vector<double>> grads(parameter_count(), std::vector<double>(dim * dim, 0.0));

    const double* loadings = parameters.data();

    for (std::size_t i = 0; i < dim; ++i) {
        for (std::size_t f = 0; f < rank_; ++f) {
            const std::size_t param_index = i * rank_ + f;
            auto& grad = grads[param_index];
            for (std::size_t j = 0; j < dim; ++j) {
                const double value = loadings[j * rank_ + f];
                grad[i * dim + j] += value;
                grad[j * dim + i] += value;
            }
        }
    }

    for (std::size_t i = 0; i < dim; ++i) {
        grads[loadings_count + i][i * dim + i] = 1.0;
    }

    return grads;
}


std::unique_ptr<CovarianceStructure> create_covariance_structure(
    const CovarianceSpec& spec,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data) {

    const std::string normalized = normalize_structure_id(spec.structure);

    if (normalized == "unstructured") {
        return std::make_unique<UnstructuredCovariance>(spec.dimension);
    } else if (normalized == "diagonal") {
        return std::make_unique<DiagonalCovariance>(spec.dimension);
    } else if (normalized == "scaled_fixed") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
            throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        if (fixed_it->second.empty()) {
            throw std::runtime_error("Fixed covariance data is empty for: " + spec.id);
        }
        return std::make_unique<ScaledFixedCovariance>(fixed_it->second[0], spec.dimension);
    } else if (normalized == "genomic" || normalized == "grm") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end() || fixed_it->second.empty()) {
            throw std::runtime_error("Missing genomic kernel data for: " + spec.id);
        }
        const auto& kernel = fixed_it->second.front();
        if (kernel.size() != spec.dimension * spec.dimension) {
            throw std::runtime_error("Genomic kernel dimension mismatch for covariance: " + spec.id);
        }
        return std::make_unique<GenomicKernelCovariance>(kernel, spec.dimension);
    } else if (normalized == "multi_kernel") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
            throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        return std::make_unique<MultiKernelCovariance>(fixed_it->second, spec.dimension);
    } else if (normalized == "multi_kernel_simplex") {
        auto fixed_it = fixed_covariance_data.find(spec.id);
        if (fixed_it == fixed_covariance_data.end()) {
            throw std::runtime_error("Missing fixed covariance data for: " + spec.id);
        }
        return std::make_unique<MultiKernelCovariance>(fixed_it->second, spec.dimension, true);
    } else if (normalized == "compound_symmetry" || normalized == "cs") {
        return std::make_unique<CompoundSymmetryCovariance>(spec.dimension);
    } else if (normalized == "ar1") {
        return std::make_unique<AR1Covariance>(spec.dimension);
    } else if (normalized == "toeplitz") {
        return std::make_unique<ToeplitzCovariance>(spec.dimension);
    } else if (auto rank = parse_factor_rank(normalized)) {
        if (*rank >= spec.dimension || *rank == 0) {
            throw std::runtime_error("Factor-analytic rank must satisfy 0 < rank < dimension for covariance: " + spec.id);
        }
        return std::make_unique<FactorAnalyticCovariance>(spec.dimension, *rank);
    } else {
        throw std::runtime_error("Unknown covariance structure: " + spec.structure);
    }
}

std::vector<bool> build_covariance_positive_mask(const CovarianceSpec& spec,
                                                 const CovarianceStructure& structure) {
    std::size_t count = structure.parameter_count();
    std::vector<bool> mask(count, false);
    if (count == 0) {
        return mask;
    }

    const std::string normalized = normalize_structure_id(spec.structure);

    if (normalized == "diagonal") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "unstructured") {
        std::size_t idx = 0;
        for (std::size_t row = 0; row < spec.dimension; ++row) {
            for (std::size_t col = 0; col <= row; ++col) {
                if (idx < mask.size()) {
                    mask[idx] = (row == col);
                }
                ++idx;
            }
        }
        return mask;
    }

    if (normalized == "scaled_fixed") {
        mask[0] = true;
        return mask;
    }

    if (normalized == "genomic" || normalized == "grm") {
        mask[0] = true;
        return mask;
    }

    if (normalized == "multi_kernel") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "multi_kernel_simplex") {
        if (!mask.empty()) {
            mask[0] = true; // sigma_sq is positive
            // Remaining parameters (weights) are free (softmax inputs)
        }
        return mask;
    }

    if (normalized == "compound_symmetry" || normalized == "cs") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "ar1") {
        mask[0] = true;
        if (mask.size() > 1) {
            mask[1] = true;
        }
        return mask;
    }

    if (normalized == "toeplitz") {
        std::fill(mask.begin(), mask.end(), true);
        return mask;
    }

    if (normalized == "factor_analytic" || normalized.rfind("fa", 0) == 0 || normalized.rfind("factor_analytic", 0) == 0) {
        // Uniquenesses are at the end and must be positive
        // Loadings are free
        // Structure: [loadings..., uniquenesses...]
        // Loadings count = dim * rank
        // Uniquenesses count = dim
        if (auto rank = parse_factor_rank(normalized)) {
             std::size_t loadings_count = spec.dimension * (*rank);
             for (std::size_t i = 0; i < spec.dimension; ++i) {
                 if (loadings_count + i < mask.size()) {
                     mask[loadings_count + i] = true;
                 }
             }
        }
        return mask;
    }

    return mask;
}

}  // namespace libsemx
