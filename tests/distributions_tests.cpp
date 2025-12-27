#include <gtest/gtest.h>

#include <cmath>

#include "lib/distributions/BernoulliDistribution.hpp"
#include "lib/distributions/BinomialDistribution.hpp"
#include "lib/distributions/CauchyDistribution.hpp"
#include "lib/distributions/DistributionExperiment.hpp"
#include "lib/distributions/ExponentialDistribution.hpp"
#include "lib/distributions/GeometricDistribution.hpp"
#include "lib/distributions/LaplaceDistribution.hpp"
#include "lib/distributions/NormalDistribution.hpp"
#include "lib/distributions/PoissonDistribution.hpp"
#include "lib/distributions/UniformDistribution.hpp"

TEST(DistributionTest, NormalDistributionBasicProperties) {
  using namespace ptm;

  NormalDistribution nd(0.0, 1.0);
  double pdf0 = nd.Pdf(0.0);
  double cdf0 = nd.Cdf(0.0);

  EXPECT_NEAR(pdf0, 0.3989, 1e-3);

  EXPECT_NEAR(cdf0, 0.5, 1e-3);
}

TEST(DistributionExperimentTest, EmpiricalMeanCloseToTheoretical) {
  using namespace ptm;

  std::mt19937 rng(123);

  auto dist = std::make_shared<NormalDistribution>(5.0, 2.0);
  DistributionExperiment experiment(dist, 20000);

  auto stats = experiment.Run(rng);

  EXPECT_NEAR(stats.empirical_mean, dist->TheoreticalMean(), 0.1);
  EXPECT_NEAR(stats.empirical_variance, dist->TheoreticalVariance(), 0.3);
}

TEST(DistributionTest, UniformDistributionBasicProperties) {
  using namespace ptm;

  UniformDistribution ud(0.0, 2.0);
  EXPECT_NEAR(ud.Pdf(1.0), 0.5, 1e-9);
  EXPECT_NEAR(ud.Cdf(0.0), 0.0, 1e-9);
  EXPECT_NEAR(ud.Cdf(2.0), 1.0, 1e-9);

  EXPECT_NEAR(ud.TheoreticalMean(), 1.0, 1e-9);
  EXPECT_NEAR(ud.TheoreticalVariance(), 1.0 / 3.0, 1e-9);
}

TEST(DistributionTest, BernoulliDistributionBasic) {
  using namespace ptm;

  BernoulliDistribution bd(0.3);
  EXPECT_NEAR(bd.Pdf(0.0), 0.7, 1e-9);
  EXPECT_NEAR(bd.Pdf(1.0), 0.3, 1e-9);
  EXPECT_NEAR(bd.Cdf(0.5), 0.7, 1e-9);
  EXPECT_NEAR(bd.TheoreticalMean(), 0.3, 1e-9);
  EXPECT_NEAR(bd.TheoreticalVariance(), 0.21, 1e-9);
}

TEST(DistributionTest, BinomialDistributionBasic) {
  using namespace ptm;

  BinomialDistribution bd(10, 0.5);
  double p5 = bd.Pdf(5.0);
  EXPECT_NEAR(p5, 0.246, 1e-2);

  EXPECT_NEAR(bd.TheoreticalMean(), 5.0, 1e-9);
  EXPECT_NEAR(bd.TheoreticalVariance(), 2.5, 1e-9);
}

TEST(DistributionTest, ExponentialDistributionBasic) {
  using namespace ptm;

  double lambda = 2;
  ExponentialDistribution ex(lambda);

  EXPECT_NEAR(ex.Pdf(1.0), 0.270670557, 1e-6);
  EXPECT_NEAR(ex.Pdf(-1.0), 0, 1e-9);
  EXPECT_NEAR(ex.Cdf(3.0), 0.9975212478, 1e-8);
  EXPECT_NEAR(ex.Cdf(-1.0), 0, 1e-9);

  EXPECT_NEAR(ex.TheoreticalMean(), 0.5, 1e-9);
  EXPECT_NEAR(ex.TheoreticalVariance(), 0.25, 1e-9);
}

TEST(DistributionTest, GeometricDistributionBasic) {
  using namespace ptm;

  double p = 0.4;
  GeometricDistribution gd(p);

  EXPECT_NEAR(gd.Pdf(1.0), p, 1e-9);
  EXPECT_NEAR(gd.Cdf(3.0), 1.0 - std::pow(1.0 - p, 3), 1e-9);

  EXPECT_NEAR(gd.TheoreticalMean(), 1.0 / p, 1e-9);
  EXPECT_NEAR(gd.TheoreticalVariance(), (1.0 - p) / (p * p), 1e-9);
}

TEST(DistributionTest, PoissonDistributionBasic) {
  using namespace ptm;

  double lambda = 3.0;
  PoissonDistribution pd(lambda);

  EXPECT_NEAR(pd.Pdf(0.0), std::exp(-lambda), 1e-9);

  EXPECT_NEAR(pd.TheoreticalMean(), lambda, 1e-9);
  EXPECT_NEAR(pd.TheoreticalVariance(), lambda, 1e-9);

  double cdf1 = pd.Cdf(1.0);
  double p0 = pd.Pdf(0.0);
  double p1 = pd.Pdf(1.0);
  EXPECT_NEAR(cdf1, p0 + p1, 1e-6);
}

TEST(DistributionTest, CauchyDistributionBasic) {
  using namespace ptm;

  CauchyDistribution cd(0.0, 1.0);
  EXPECT_NEAR(cd.Pdf(0.0), 1.0 / (std::numbers::pi * 1.0), 1e-9);
  EXPECT_NEAR(cd.Cdf(0.0), 0.5, 1e-9);
}

TEST(DistributionTest, LaplaceDistributionBasic) {
  using namespace ptm;

  LaplaceDistribution ld(0.0, 1.0);
  EXPECT_NEAR(ld.Pdf(0.0), 0.5, 1e-9);
  EXPECT_NEAR(ld.Cdf(0.0), 0.5, 1e-9);

  EXPECT_NEAR(ld.TheoreticalMean(), 0.0, 1e-9);
  EXPECT_NEAR(ld.TheoreticalVariance(), 2.0, 1e-9);
}

TEST(DistributionExperimentTest, BinomialEmpiricalMean) {
  using namespace ptm;

  std::mt19937 rng(777);
  auto dist = std::make_shared<BinomialDistribution>(20, 0.3);
  DistributionExperiment experiment(dist, 50000);

  auto stats = experiment.Run(rng);
  EXPECT_NEAR(stats.empirical_mean, dist->TheoreticalMean(), 0.2);
  EXPECT_NEAR(stats.empirical_variance, dist->TheoreticalVariance(), 0.5);
}

TEST(DistributionExperimentTest, KolmogorovDistanceZero) {
  using namespace ptm;

  std::mt19937 rng(1);
  auto dist = std::make_shared<BernoulliDistribution>(1.0);
  DistributionExperiment experiment(dist, 1000);
  std::vector<double> grid = {0.0, 1.0};

  auto empirical_cdf = experiment.EmpiricalCdf(grid, rng, 1000);
  double d = experiment.KolmogorovDistance(grid, empirical_cdf);

  EXPECT_NEAR(d, 0.0, 1e-12);
}

TEST(DistributionExperimentTest, EmpiricalCdfDoNotBreakPoint) {
  using namespace ptm;

  std::mt19937 rng(42);
  auto dist = std::make_shared<NormalDistribution>(0.0, 1.0);
  DistributionExperiment experiment(dist, 10000);

  std::vector<double> grid;
  for (int i = 0; i <= 100; ++i) {
    grid.push_back(i);
  }

  auto cdf = experiment.EmpiricalCdf(grid, rng, 10000);

  for (size_t i = 1; i < cdf.size(); ++i) {
    EXPECT_LE(cdf[i - 1], cdf[i]);
  }

  for (double v : cdf) {
    EXPECT_GE(v, 0.0);
    EXPECT_LE(v, 1.0);
  }
}

TEST(DistributionExperimentTest, KolmogorovDistanceDecreasesWithSampleSize) {
  using namespace ptm;

  std::mt19937 rng1(100);
  std::mt19937 rng2(100);
  auto dist = std::make_shared<NormalDistribution>(0.0, 1.0);
  DistributionExperiment small(dist, 1000);
  DistributionExperiment large(dist, 10000);
  std::vector<double> grid = {-2, -1, 0, 1, 2};

  double d1 = small.KolmogorovDistance(grid, small.EmpiricalCdf(grid, rng1, 1000));
  double d2 = large.KolmogorovDistance(grid, large.EmpiricalCdf(grid, rng2, 10000));

  EXPECT_GE(d1, d2);
}

TEST(DistributionExperimentTest, EmpiricalCdfCloseToTheoretical) {
  using namespace ptm;

  std::mt19937 rng(321);
  auto dist = std::make_shared<UniformDistribution>(0.0, 1.0);
  DistributionExperiment experiment(dist, 20000);
  std::vector<double> grid = {0.25, 0.5, 0.75};
  
  auto cdf = experiment.EmpiricalCdf(grid, rng, 20000);

  for (size_t i = 0; i < grid.size(); ++i) {
    EXPECT_NEAR(cdf[i], dist->Cdf(grid[i]), 0.02);
  }
}
