#include "libsemx/mixed_outcome.hpp"
#include "libsemx/outcome_family_factory.hpp"
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <iostream>

namespace libsemx {

MixedOutcome::MixedOutcome(const std::string& config) {
    // Format: mixed;family1,count1;family2,count2;...
    if (config.substr(0, 6) != "mixed;") {
        throw std::invalid_argument("Invalid mixed outcome config: " + config);
    }

    std::string content = config.substr(6);
    std::stringstream ss(content);
    std::string segment;
    std::size_t current_offset = 0;

    while (std::getline(ss, segment, ';')) {
        std::stringstream segment_ss(segment);
        std::string family_name;
        std::string count_str;
        
        if (!std::getline(segment_ss, family_name, ',') || !std::getline(segment_ss, count_str)) {
            throw std::invalid_argument("Invalid mixed outcome segment: " + segment);
        }

        std::size_t count = std::stoul(count_str);
        
        families_.push_back({
            OutcomeFamilyFactory::create(family_name),
            count,
            current_offset
        });

        current_offset += count;
    }
}

OutcomeEvaluation MixedOutcome::evaluate(double observed,
                                          double linear_predictor,
                                          double dispersion,
                                          double status,
                                          const std::vector<double>& extra_params) const {
    // status is used as the 0-based index of the family to use
    std::size_t index = static_cast<std::size_t>(status);
    
    if (index >= families_.size()) {
        std::cerr << "MixedOutcome error: index " << index << " >= size " << families_.size() << std::endl;
        throw std::out_of_range("MixedOutcome index out of range: " + std::to_string(index));
    }

    const auto& sub = families_[index];
    
    // Slice extra_params
    std::vector<double> sub_extra_params;
    if (sub.extra_param_count > 0) {
        if (sub.extra_param_offset + sub.extra_param_count > extra_params.size()) {
             std::cerr << "MixedOutcome error: extra params out of range. Offset: " << sub.extra_param_offset 
                       << " Count: " << sub.extra_param_count << " Total: " << extra_params.size() << std::endl;
             throw std::out_of_range("MixedOutcome extra params out of range");
        }
        sub_extra_params.assign(
            extra_params.begin() + sub.extra_param_offset,
            extra_params.begin() + sub.extra_param_offset + sub.extra_param_count
        );
    }

    if (!sub.family) {
        std::cerr << "MixedOutcome error: sub-family is null for index " << index << std::endl;
        throw std::runtime_error("MixedOutcome sub-family is null");
    }

    auto eval = sub.family->evaluate(observed, linear_predictor, dispersion, 1.0, sub_extra_params);


    // Map gradients back to full extra_params vector
    std::vector<double> full_d_extra(extra_params.size(), 0.0);
    std::vector<double> full_hessian(extra_params.size(), 0.0); 
    
    if (!eval.d_extra_params.empty()) {
        for (std::size_t i = 0; i < eval.d_extra_params.size(); ++i) {
            full_d_extra[sub.extra_param_offset + i] = eval.d_extra_params[i];
        }
    }
    
    if (!eval.d_hessian_d_extra_params.empty()) {
        for (std::size_t i = 0; i < eval.d_hessian_d_extra_params.size(); ++i) {
            full_hessian[sub.extra_param_offset + i] = eval.d_hessian_d_extra_params[i];
        }
    }

    eval.d_extra_params = std::move(full_d_extra);
    eval.d_hessian_d_extra_params = std::move(full_hessian);

    return eval;
}


bool MixedOutcome::has_dispersion() const noexcept {
    // If any sub-family has dispersion, we say yes?
    // Or we assume yes and let sub-families ignore it.
    return true;
}

double MixedOutcome::default_dispersion(std::size_t n) const {
    return 1.0;
}

}  // namespace libsemx
