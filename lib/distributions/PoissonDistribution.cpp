#include <stdexcept>
#include "PoissonDistribution.hpp"

namespace ptm {
PoissonDistribution::PoissonDistribution(double lambda) : lambda_(lambda) {
    if (lambda <= 0) {
        throw std::invalid_argument("lambda must be positive");
    }
}

double PoissonDistribution::Pdf(double x) const {
    if (x < 0 || std::round(x) != x) {
        return 0;
    }

    double k = 1;

    for (int i = 1; i < x; ++i) {
        k *= i;
    }
    return std::pow(lambda_, x) / k * std::exp(-lambda_);
}

double PoissonDistribution::Cdf(double x) const {
    if (x < 0) {
        return 0;
    }

    unsigned int k_max = static_cast<unsigned int>(std::floor(x));
    double sum = 0;

    for (int k = 0; k <= k_max; ++k) {
        sum += Pdf(static_cast<double>(k));
    }

    return sum;
}

double PoissonDistribution::Sample(std::mt19937& rng) const {
    double l = std::exp(-lambda_);
    int k = 0;
    double p = 1;

    while (p > l) {
        ++k;
        p *= (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    }
    return static_cast<double>(k - 1);
}

double PoissonDistribution::TheoreticalMean() const {
    return lambda_;
}

double PoissonDistribution::TheoreticalVariance() const {
    return lambda_;
}
} // namespace ptm
