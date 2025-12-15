#pragma once

#include <cstddef>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

#include <Eigen/SparseCore>

#include "libsemx/model_ir.hpp"

namespace libsemx {

class CovarianceStructure {
public:
    virtual ~CovarianceStructure() = default;

    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }

    [[nodiscard]] std::vector<double> materialize(const std::vector<double>& parameters) const;

    [[nodiscard]] virtual bool is_sparse() const { return false; }

    [[nodiscard]] virtual Eigen::SparseMatrix<double> materialize_sparse(const std::vector<double>& parameters) const;

    [[nodiscard]] virtual std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const;

    [[nodiscard]] virtual std::vector<Eigen::SparseMatrix<double>> parameter_gradients_sparse(const std::vector<double>& parameters) const;

protected:
    CovarianceStructure(std::size_t dimension, std::size_t parameter_count);

    virtual void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const = 0;

    virtual void validate_parameters(const std::vector<double>& parameters) const;

private:
    std::size_t dimension_;
    std::size_t parameter_count_;
};

[[nodiscard]] std::unique_ptr<CovarianceStructure> create_covariance_structure(
    const CovarianceSpec& spec,
    const std::unordered_map<std::string, std::vector<std::vector<double>>>& fixed_covariance_data = {});

[[nodiscard]] std::vector<bool> build_covariance_positive_mask(const CovarianceSpec& spec,
                                                               const CovarianceStructure& structure);


class UnstructuredCovariance final : public CovarianceStructure {
public:
    explicit UnstructuredCovariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class ExplicitCovariance final : public CovarianceStructure {
public:
    explicit ExplicitCovariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class DiagonalCovariance final : public CovarianceStructure {
public:
    explicit DiagonalCovariance(std::size_t dimension);

    [[nodiscard]] bool is_sparse() const override { return true; }

    [[nodiscard]] Eigen::SparseMatrix<double> materialize_sparse(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<Eigen::SparseMatrix<double>> parameter_gradients_sparse(const std::vector<double>& parameters) const override;

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class CompoundSymmetryCovariance final : public CovarianceStructure {
public:
    explicit CompoundSymmetryCovariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class AR1Covariance final : public CovarianceStructure {
public:
    explicit AR1Covariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class ToeplitzCovariance final : public CovarianceStructure {
public:
    explicit ToeplitzCovariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class FactorAnalyticCovariance final : public CovarianceStructure {
public:
    FactorAnalyticCovariance(std::size_t dimension, std::size_t rank);

    [[nodiscard]] std::size_t rank() const noexcept { return rank_; }

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;

private:
    std::size_t rank_;
};

class ARMA11Covariance final : public CovarianceStructure {
public:
    explicit ARMA11Covariance(std::size_t dimension);

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;
};

class RBFKernel final : public CovarianceStructure {
public:
    RBFKernel(const std::vector<double>& coordinates, std::size_t dimension);

    [[nodiscard]] const std::vector<double>& coordinates() const noexcept { return coordinates_; }

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

private:
    std::vector<double> coordinates_;  // n x d coordinate matrix stored row-major
    std::vector<double> distance_matrix_;  // Precomputed pairwise distances
    void precompute_distances();
};

class ExponentialKernel final : public CovarianceStructure {
public:
    ExponentialKernel(const std::vector<double>& coordinates, std::size_t dimension);

    [[nodiscard]] const std::vector<double>& coordinates() const noexcept { return coordinates_; }

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

private:
    std::vector<double> coordinates_;  // n x d coordinate matrix stored row-major
    std::vector<double> distance_matrix_;  // Precomputed pairwise distances
    void precompute_distances();
};

}  // namespace libsemx
