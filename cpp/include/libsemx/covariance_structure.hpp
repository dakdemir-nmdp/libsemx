#pragma once

#include <cstddef>
#include <vector>

namespace libsemx {

class CovarianceStructure {
public:
    virtual ~CovarianceStructure() = default;

    [[nodiscard]] std::size_t dimension() const noexcept { return dimension_; }

    [[nodiscard]] std::size_t parameter_count() const noexcept { return parameter_count_; }

    [[nodiscard]] std::vector<double> materialize(const std::vector<double>& parameters) const;

    [[nodiscard]] virtual std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const;

protected:
    CovarianceStructure(std::size_t dimension, std::size_t parameter_count);

    virtual void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const = 0;

    virtual void validate_parameters(const std::vector<double>& parameters) const;

private:
    std::size_t dimension_;
    std::size_t parameter_count_;
};

class UnstructuredCovariance final : public CovarianceStructure {
public:
    explicit UnstructuredCovariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

class DiagonalCovariance final : public CovarianceStructure {
public:
    explicit DiagonalCovariance(std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    void validate_parameters(const std::vector<double>& parameters) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(const std::vector<double>& parameters) const override;
};

}  // namespace libsemx
