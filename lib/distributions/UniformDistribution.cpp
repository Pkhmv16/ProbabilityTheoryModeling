#include <cmath>
#include <stdexcept>
#include "UniformDistribution.hpp"

namespace ptm {
UniformDistribution::UniformDistribution(double a, double b) : a_(a), b_(b) {}

double UniformDistribution::Pdf(double x) const {
     if ( a_ <= x <= b_) {
        return 1 / (b_ - a_);
     }

     return 0;
}

double UniformDistribution::Cdf(double x) const {
    if ( x < a_) {
        return 0;
    }

    if ( x >= b_) {
        return 1;
    }

    return (x - a_) / (b_ - a_);
}

double UniformDistribution::Sample(std::mt19937& rng) const {
    double u = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    return a_ + (b_ - a_) * u;
}

double UniformDistribution::TheoreticalMean() const {
    return (a_ + b_) / 2;
}

double UniformDistribution::TheoreticalVariance() const {
    return (b_ - a_) * (b_ - a_) / 12;
}
} // namespace ptm
