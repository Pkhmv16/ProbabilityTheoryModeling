#include <stdexcept>
#include "NormalDistribution.hpp"
#include <numbers>

namespace ptm {
NormalDistribution::NormalDistribution(double mean, double stddev) : mean_(mean), stddev_(stddev) {
    if (stddev <= 0) {
        throw std::invalid_argument("stddev must be positive");
    }
}

double NormalDistribution::Pdf(double x) const {
    double coeff = std::exp(-0.5 * std::pow((x - mean_) / stddev_, 2));
    return 1 / (std::sqrt(2 * std::numbers::pi) * stddev_) * coeff;
}

double NormalDistribution::Cdf(double x) const {
    return 0.5 * (1 + std::erf((x - mean_) / (stddev_ * std::sqrt(2))));
}

double NormalDistribution::Sample(std::mt19937& rng) const {
    double u1 = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    double u2 = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);

    double z0 = std::sqrt(-2 * std::log(u1)) * std::cos(2 * std::numbers::pi * u2);
    return mean_ + stddev_ * z0;
}

double NormalDistribution::TheoreticalMean() const {
    return mean_;
}

double NormalDistribution::TheoreticalVariance() const {
    return stddev_ * stddev_;
}
} // namespace ptm
