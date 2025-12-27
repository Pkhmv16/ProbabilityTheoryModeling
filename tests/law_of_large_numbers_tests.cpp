#include <gtest/gtest.h>
#include <random>

#include "lib/distributions/BernoulliDistribution.hpp"
#include "lib/law-of-large-numbers/LawOfLargeNumbersSimulator.hpp"

TEST(LawOfLargeNumbersTest, BernoulliMeanConverges) {
  using namespace ptm;

  std::mt19937 rng(123);

  auto dist = std::make_shared<BernoulliDistribution>(0.3);
  LawOfLargeNumbersSimulator sim(dist);

  size_t max_n = 100000;
  size_t step = 5000;

  LLNPathResult result = sim.Simulate(rng, max_n, step);

  ASSERT_FALSE(result.entries.empty());

  const double theoretical_mean = dist->TheoreticalMean();

  double first_error = result.entries.front().abs_error;
  double last_error = result.entries.back().abs_error;
  double last_mean = result.entries.back().sample_mean;

  EXPECT_GT(first_error, last_error);

  EXPECT_NEAR(last_mean, theoretical_mean, 0.05);

  EXPECT_LT(last_error, 0.05);

  for (std::size_t i = 1; i < result.entries.size(); ++i) {
    EXPECT_EQ(result.entries[i].n, result.entries[i - 1].n + step);
  }
}

TEST(LawOfLargeNumbersTest, HandlesMaxNNotMultipleOfStep) {
  using namespace ptm;

  std::mt19937 rng(123);

  auto dist = std::make_shared<BernoulliDistribution>(0.3);
  LawOfLargeNumbersSimulator sim(dist);

  size_t max_n = 12345;
  size_t step = 2000;

  LLNPathResult result = sim.Simulate(rng, max_n, step);

  ASSERT_FALSE(result.entries.empty());
  EXPECT_EQ(result.entries.size(), max_n / step);
  EXPECT_EQ(result.entries.back().n, (max_n / step) * step);

  for (size_t i = 1; i < result.entries.size(); ++i) {
    EXPECT_EQ(result.entries[i].n, result.entries[i - 1].n + step);
  }
}

TEST(LawOfLargeNumbersTest, MeansAndErrorsAreInReasonableRangeForBernoulli) {
  using namespace ptm;

  std::mt19937 rng(123);

  auto dist = std::make_shared<BernoulliDistribution>(0.3);
  LawOfLargeNumbersSimulator sim(dist);

  size_t max_n = 50000;
  size_t step = 5000;

  LLNPathResult result = sim.Simulate(rng, max_n, step);

  ASSERT_FALSE(result.entries.empty());

  for (const auto& e : result.entries) {
    EXPECT_GE(e.sample_mean, 0.0);
    EXPECT_LE(e.sample_mean, 1.0);
    EXPECT_GE(e.abs_error, 0.0);
    EXPECT_LE(e.abs_error, 1.0);
  }
}

TEST(LawOfLargeNumbersTest, DeterministicWithSameSeed) {
  using namespace ptm;

  auto dist = std::make_shared<BernoulliDistribution>(0.3);
  LawOfLargeNumbersSimulator sim(dist);

  size_t max_n = 30000;
  size_t step = 3000;

  std::mt19937 rng1(123);
  std::mt19937 rng2(123);

  LLNPathResult r1 = sim.Simulate(rng1, max_n, step);
  LLNPathResult r2 = sim.Simulate(rng2, max_n, step);

  ASSERT_EQ(r1.entries.size(), r2.entries.size());

  for (std::size_t i = 0; i < r1.entries.size(); ++i) {
    EXPECT_EQ(r1.entries[i].n, r2.entries[i].n);
    EXPECT_DOUBLE_EQ(r1.entries[i].sample_mean, r2.entries[i].sample_mean);
    EXPECT_DOUBLE_EQ(r1.entries[i].abs_error, r2.entries[i].abs_error);
  }
}