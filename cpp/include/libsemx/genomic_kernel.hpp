#pragma once

#include "libsemx/covariance_structure.hpp"

#include <cstddef>
#include <vector>

namespace libsemx {

struct GenomicKernelOptions {
    bool center = true;
    bool normalize = true;
};

class GenomicRelationshipMatrix {
public:
    static std::vector<double> vanraden(const std::vector<double>& markers,
                                        std::size_t n_individuals,
                                        std::size_t n_markers,
                                        GenomicKernelOptions options = {});

    static std::vector<double> kronecker(const std::vector<double>& left,
                                         std::size_t left_dim,
                                         const std::vector<double>& right,
                                         std::size_t right_dim);
};

// Covariance wrapper that scales a precomputed genomic kernel (typically a GRM).
class GenomicKernelCovariance final : public CovarianceStructure {
public:
    GenomicKernelCovariance(std::vector<double> kernel, std::size_t dimension);

protected:
    void fill_covariance(const std::vector<double>& parameters, std::vector<double>& matrix) const override;

    [[nodiscard]] std::vector<std::vector<double>> parameter_gradients(
        const std::vector<double>& parameters) const override;

private:
    std::vector<double> kernel_;
};

}  // namespace libsemx
