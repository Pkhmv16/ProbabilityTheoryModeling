#include <stdexcept>
#include "ExponentialDistribution.hpp"

namespace ptm {
ExponentialDistribution::ExponentialDistribution(double lambda) : lambda_(lambda) {
    if (lambda <= 0) {
        throw std::invalid_argument("lambda must be > 0");
    }
}

double ExponentialDistribution::Pdf(double x) const {
    if (x < 0) {
        return 0;
    }

    return lambda_ * std::exp(-lambda_ * x);
}

double ExponentialDistribution::Cdf(double x) const {
    if (x < 0) {
        return 0;
    }

    return 1 - lambda_ * std::exp(-lambda_ * x);
}

double ExponentialDistribution::Sample(std::mt19937& rng) const {
    double u = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    return -std::log(u) / lambda_;
}

double ExponentialDistribution::TheoreticalMean() const {
    return 1 / lambda_;
}

double ExponentialDistribution::TheoreticalVariance() const {
    return 1 / (lambda_ * lambda_);
}
} // namespace ptm
