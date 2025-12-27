#include <stdexcept>
#include "GeometricDistribution.hpp"

namespace ptm {
GeometricDistribution::GeometricDistribution(double p) : p_(p) {
    if (p < 0 || p > 1) {
        throw std::invalid_argument("p must be in [0,1]");
    }
}

double GeometricDistribution::Pdf(double x) const {
    if (x < 0 || std::round(x) != x) {
        return 0;
    }

    return std::pow(1 - p_, x - 1) * p_;
}

double GeometricDistribution::Cdf(double x) const {
    if (x < 1) {
        return 0;
    }

    unsigned int k = static_cast<unsigned int>(std::floor(x));

    return 1 - std::pow(1 - p_, k);
}

double GeometricDistribution::Sample(std::mt19937& rng) const {
    double u = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    return std::floor(std::log(1 - u) / std::log(1 - p_));
}

double GeometricDistribution::TheoreticalMean() const {
    return 1 / p_;
}

double GeometricDistribution::TheoreticalVariance() const {
    return (1 - p_) / (p_ * p_);
}
} // namespace ptm
