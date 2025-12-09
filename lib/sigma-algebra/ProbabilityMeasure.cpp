#include "ProbabilityMeasure.hpp"

namespace ptm {

ProbabilityMeasure::ProbabilityMeasure(const OutcomeSpace& omega) : omega_(omega), atom_probs_(omega.GetSize(), 0.0) {
}

double ProbabilityMeasure::GetAtomicProbability(OutcomeSpace::OutcomeId id) const {
    if (id >= atom_probs_.size()) {
        return 0.0;
    }
    return atom_probs_[id];
}

void ProbabilityMeasure::SetAtomicProbability(OutcomeSpace::OutcomeId id, double p) {
    if (id >= atom_probs_.size()) {
        throw std::out_of_range("Outcome ID is outside Omega");
    }
    atom_probs_[id] = p;
}

bool ProbabilityMeasure::IsValid(double eps) const {
    if (atom_probs_.size() != omega_.GetSize()) {
        return false;
    }
    double sum = 0.0;
    for (double p : atom_probs_) {
        if (p < -eps) {
        return false;
        }
        sum += p;
    }
    return std::fabs(sum - 1.0) <= eps;
}

double ProbabilityMeasure::Probability(const Event& event) const {
    const std::size_t n = std::min(event.GetSize(), omega_.GetSize());
    double result = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        if (event.Contains(i)) {
        result += GetAtomicProbability(i);
        }
    }
    return result;
}

} // namespace ptm
