#include "libsemx/outcome_family_factory.hpp"
#include "libsemx/gaussian_outcome.hpp"
#include "libsemx/binomial_outcome.hpp"
#include "libsemx/negative_binomial_outcome.hpp"
#include "libsemx/weibull_outcome.hpp"
#include "libsemx/ordinal_outcome.hpp"

#include <memory>
#include <stdexcept>
#include <string>

namespace libsemx {

std::unique_ptr<OutcomeFamily> OutcomeFamilyFactory::create(const std::string& family_name) {
    if (family_name == "gaussian") {
        return std::make_unique<GaussianOutcome>();
    }
    if (family_name == "binomial") {
        return std::make_unique<BinomialOutcome>();
    }
    if (family_name == "negative_binomial" || family_name == "nbinom") {
        return std::make_unique<NegativeBinomialOutcome>();
    }
    if (family_name == "weibull" || family_name == "weibull_aft") {
        return std::make_unique<WeibullOutcome>();
    if (family_name == "ordinal" || family_name == "probit") {
        return std::make_unique<OrdinalOutcome>();
    }
    throw std::invalid_argument("Unknown outcome family: " + family_name);
}   throw std::invalid_argument("Unknown outcome family: " + family_name);
}

}  // namespace libsemx