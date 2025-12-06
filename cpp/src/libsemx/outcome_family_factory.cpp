#include "libsemx/outcome_family_factory.hpp"
#include "libsemx/gaussian_outcome.hpp"
#include "libsemx/binomial_outcome.hpp"
#include "libsemx/poisson_outcome.hpp"
#include "libsemx/negative_binomial_outcome.hpp"
#include "libsemx/weibull_outcome.hpp"
#include "libsemx/exponential_outcome.hpp"
#include "libsemx/lognormal_outcome.hpp"
#include "libsemx/loglogistic_outcome.hpp"
#include "libsemx/ordinal_outcome.hpp"
#include "libsemx/fixed_outcome.hpp"

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
    if (family_name == "poisson") {
        return std::make_unique<PoissonOutcome>();
    }
    if (family_name == "negative_binomial" || family_name == "nbinom") {
        return std::make_unique<NegativeBinomialOutcome>();
    }
    if (family_name == "weibull" || family_name == "weibull_aft") {
        return std::make_unique<WeibullOutcome>();
    }
    if (family_name == "exponential" || family_name == "exponential_aft") {
        return std::make_unique<ExponentialOutcome>();
    }
    if (family_name == "lognormal" || family_name == "lognormal_aft") {
        return std::make_unique<LognormalOutcome>();
    }
    if (family_name == "loglogistic" || family_name == "loglogistic_aft") {
        return std::make_unique<LogLogisticOutcome>();
    }
    if (family_name == "ordinal" || family_name == "probit") {
        return std::make_unique<OrdinalOutcome>();
    }
    if (family_name == "fixed" || family_name == "none") {
        return std::make_unique<FixedOutcome>();
    }
    throw std::invalid_argument("Unknown outcome family: " + family_name);
}

}  // namespace libsemx