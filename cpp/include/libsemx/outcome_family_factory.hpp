#pragma once

#include "libsemx/outcome_family.hpp"

#include <memory>
#include <string>

namespace libsemx {

class OutcomeFamilyFactory {
public:
    static std::unique_ptr<OutcomeFamily> create(const std::string& family_name);
};

}  // namespace libsemx