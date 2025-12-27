#include <stdexcept>
#include "BernoulliDistribution.hpp";
#include "BinomialDistribution.hpp"

namespace ptm {

BinomialDistribution::BinomialDistribution(unsigned int n, double p) : n_(n), p_(p) {
    if (p < 0 || p > 1) {
        throw std::invalid_argument("p must be in [0, 1]");
    }
}

double BinomialDistribution::Pdf(double x) const {
    if (x < 0 || x > n_ || std::round(x) != x) {
        return 0;
    }

    double coeff = 1;

    for (int i = 1; i <= x; ++i) {
        coeff *= static_cast<double>(n_ - i + 1) / static_cast<double>(i);
    }

    return coeff * std::pow(p_, x) * std::pow(1 - p_, n_ - x);
}

double BinomialDistribution::Cdf(double x) const {
    if (x < 0) {
        return 0;
    }
    if (x > n_) {
        return 1;
    }

    double total = 0;

    for (int i = 0; i <= x; ++i) {
        total += Pdf(i);
    }

    return total;
}

double BinomialDistribution::Sample(std::mt19937& rng) const {
    double count = 0;
    BernoulliDistribution bernoulli(p_);
    for (int i = 0; i < n_; ++i) {
        count += bernoulli.Sample(rng);
    }
    return count;
}

double BinomialDistribution::TheoreticalMean() const {
    return n_ * p_;
}

double BinomialDistribution::TheoreticalVariance() const {
    return n_ * p_ * (1 - p_);
}

} // namespace ptm
