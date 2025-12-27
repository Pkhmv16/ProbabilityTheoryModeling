#include "DistributionExperiment.hpp"

namespace ptm {
DistributionExperiment::DistributionExperiment(std::shared_ptr<Distribution> dist, size_t sample_size) :
    dist_(std::move(dist)), sample_size_(sample_size) {
}

ExperimentStats DistributionExperiment::Run(std::mt19937& rng) {
    std::vector<double> samples(sample_size_);

    for (int i = 0; i < sample_size_; ++i) {
        samples[i] = dist_->Sample(rng);
    }

    double empirical_mean = 0;

    for (int i = 0; i < sample_size_; ++i) {
        empirical_mean += samples[i];
    }

    empirical_mean /= sample_size_;

    double empirical_variance = 0;

    for (int i = 0; i < sample_size_; ++i) {
        double iteration_variance = samples[i] - empirical_mean;
        empirical_variance += std::pow(iteration_variance, 2);
    }

    empirical_variance /= sample_size_;
    ExperimentStats stats;
    stats.empirical_mean = empirical_mean;
    stats.empirical_variance = empirical_variance;
    stats.mean_error = dist_->TheoreticalMean() - empirical_mean;
    stats.variance_error = dist_->TheoreticalVariance() - empirical_variance;

    return stats;
}

std::vector<double> DistributionExperiment::EmpiricalCdf(const std::vector<double>& grid,
                                                         std::mt19937& rng,
                                                         std::size_t sample_size) {
    std::vector<double> samples(sample_size);

    for (int i = 0; i < sample_size; ++i) {
        samples[i] = dist_->Sample(rng);
    }

    std::vector<double> cdf(grid.size(), 0);

    for (int i = 0; i < grid.size(); ++i) {
        double counter = 0;
        for (int j = 0; j < sample_size; ++j) {
            if (samples[j] <= grid[i]) {
                ++counter;
            }
        }

        cdf[i] = counter /static_cast<double>(sample_size);
    }
    return cdf;
}

double DistributionExperiment::KolmogorovDistance(const std::vector<double>& grid,
                                                  const std::vector<double>& empirical_cdf) const {
    double distance = 0;

    for (int i = 0; i < grid.size(); ++i) {
        double difference = std::abs(empirical_cdf[i] - dist_->Cdf(grid[i]));
        distance = std::max(distance, difference);
    }

    return distance;
}
} // namespace ptm
