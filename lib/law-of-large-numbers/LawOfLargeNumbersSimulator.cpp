#include "LawOfLargeNumbersSimulator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ptm {

    LawOfLargeNumbersSimulator::LawOfLargeNumbersSimulator(std::shared_ptr<Distribution> dist) : dist_(std::move(dist)) { }

    LLNPathResult LawOfLargeNumbersSimulator::Simulate(std::mt19937& rng, size_t max_n, size_t step) const {
        double mu = dist_->TheoreticalMean();
        LLNPathResult result;
        double sum = 0.0;

        for (size_t n = 1; n <= max_n; ++n) {
            sum += static_cast<double>(dist_->Sample(rng));

            if (n % step == 0) {
                double mean = static_cast<double>(sum / static_cast<double>(n));
                double err = std::abs(mean - mu);

                result.entries.push_back(LLNPathEntry{.n = n, .sample_mean = mean, .abs_error = err,});
            }
        }

        return result;
    }

    std::shared_ptr<Distribution> LawOfLargeNumbersSimulator::GetDistribution() const noexcept {
        return dist_;
    }

}