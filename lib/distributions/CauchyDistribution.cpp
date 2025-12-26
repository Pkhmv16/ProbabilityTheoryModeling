#include <cmath>
#include <numbers>
#include <stdexcept>
#include "CauchyDistribution.hpp"

namespace ptm {
CauchyDistribution::CauchyDistribution(double x0, double gamma) : x0_(x0), gamma_(gamma) {
    if (gamma <= 0) {
        throw std::invalid_argument("gamma must be > 0");
    }
}

double CauchyDistribution::Pdf(double x) const {
    double coeff = (x - x0_) * (x - x0_) + gamma_ * gamma_;
    return 1 / (std::numbers::pi * coeff);
}

double CauchyDistribution::Cdf(double x) const {
    double coeff = atan((x - x0_) / gamma_);
    return 1 / std::numbers::pi * coeff + 0.5;
}

double CauchyDistribution::Sample(std::mt19937& rng) const {
    double u = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    return x0_ + gamma_ * std::tan(std::numbers::pi * (u - 0.5));
}

double CauchyDistribution::TheoreticalMean() const {
    return NAN;
}

double CauchyDistribution::TheoreticalVariance() const {
    return NAN;
}
}