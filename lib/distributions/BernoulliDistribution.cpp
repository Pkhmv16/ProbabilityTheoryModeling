#include <stdexcept>
#include "BernoulliDistribution.hpp"

namespace ptm {

BernoulliDistribution::BernoulliDistribution(double x) : p_(x) {
    if (x < 0 || x > 1) {
        throw std::invalid_argument("p must be in [0, 1]");
    }
}

double BernoulliDistribution::Pdf(double x) const {
    if (x == 0) {
        return 1 - p_;
    }

    if (x == 1) {
        return p_;
    }

    return 0;
}

double BernoulliDistribution::Cdf(double x) const {
    if (x < 0) {
        return 0;
    }

    if (x < 1) {
        return 1 - p_;
    }

    return 1;
}

double BernoulliDistribution::Sample(std::mt19937& rng) const {
    double u = (static_cast<double>(rng()) + 0.5) / (static_cast<double>(rng.max()) + 1);
    return u < p_ ? 1 : 0;
}

double BernoulliDistribution::TheoreticalMean() const {
    return p_;
}

double BernoulliDistribution::TheoreticalVariance() const {
    return p_ * (1 - p_);
}

} // namespace ptm
