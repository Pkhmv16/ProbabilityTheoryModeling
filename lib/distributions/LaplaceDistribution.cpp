#include <stdexcept>
#include "LaplaceDistribution.hpp"

namespace ptm {
LaplaceDistribution::LaplaceDistribution(double mu, double b) : mu_(mu), b_(b) {
    if (b_ <= 0) {
        throw std::invalid_argument("b must be positive");
    }
}

double LaplaceDistribution::Pdf(double x) const {
    return (1 / (2 * b_)) * std::exp(-std::abs(x - mu_) / b_);
}

double LaplaceDistribution::Cdf(double x) const {
    if (x < mu_) {
        return 0.5 * std::exp((x - mu_) / b_);
    }
    return 1 - 0.5 * std::exp(-(x - mu_) / b_);
}

double LaplaceDistribution::Sample(std::mt19937& rng) const {
    double u = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    double random_sign = (u < 0.5) ? -1 : 1;
    return mu_ - b_ * random_sign * std::log(1 - 2 * std::abs(u));
}

double LaplaceDistribution::TheoreticalMean() const {
    return mu_;
}

double LaplaceDistribution::TheoreticalVariance() const {
    return 2 * b_ * b_;
}
} // namespace ptm
