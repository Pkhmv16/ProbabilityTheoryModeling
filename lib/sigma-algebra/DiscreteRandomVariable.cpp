#include "DiscreteRandomVariable.hpp"

namespace ptm {

DiscreteRandomVariable::DiscreteRandomVariable(const OutcomeSpace& omega,
                                               const ProbabilityMeasure& P,
                                               std::vector<double> values)
    : omega_(omega),
      P_(P),
      values_(std::move(values)) {}

std::optional<double> DiscreteRandomVariable::Value(OutcomeSpace::OutcomeId id) const {
    if (id >= values_.size()) {
      return std::nullopt;
    }
    return values_[id];
}

double DiscreteRandomVariable::ExpectedValue() const {
    const std::size_t n = omega_.GetSize();
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      if (i < values_.size()) {
        result += values_[i] * P_.GetAtomicProbability(i);
      }
    }
    return result;
}

} // namespace ptm