#include <sstream>

#include <gtest/gtest.h>
#include "lib/sigma-algebra/DiscreteRandomVariable.hpp"
#include "lib/sigma-algebra/Event.hpp"
#include "lib/sigma-algebra/OutcomeSpace.hpp"
#include "lib/sigma-algebra/ProbabilityMeasure.hpp"
#include "lib/sigma-algebra/SigmaAlgebra.hpp"

TEST(SigmaAlgebraTest, ProbabilityMeasureAndExpectation) {
  using namespace ptm;

  OutcomeSpace omega;
  auto w0 = omega.AddOutcome("1");
  auto w1 = omega.AddOutcome("2");
  auto w2 = omega.AddOutcome("3");

  ProbabilityMeasure P(omega);
  P.SetAtomicProbability(w0, 0.2);
  P.SetAtomicProbability(w1, 0.3);
  P.SetAtomicProbability(w2, 0.5);

  EXPECT_TRUE(P.IsValid(1e-9));

  // событие A = {1,3}
  std::vector<bool> mask(omega.GetSize(), false);
  mask[w0] = true;
  mask[w2] = true;
  Event A(mask);

  double pA = P.Probability(A);
  EXPECT_NEAR(pA, 0.7, 1e-9);

  // X(1)=1, X(2)=2, X(3)=3
  std::vector<double> X_values = {1.0, 2.0, 3.0};
  DiscreteRandomVariable X(omega, P, X_values);

  double EX = X.ExpectedValue();
  // E[X] = 1*0.2 + 2*0.3 + 3*0.5 = 2.3
  EXPECT_NEAR(EX, 2.3, 1e-9);
}

TEST(SigmaAlgebraTest, EventOperations) {
  using namespace ptm;

  OutcomeSpace omega;
  auto a = omega.AddOutcome("a");
  auto b = omega.AddOutcome("b");
  auto c = omega.AddOutcome("c");

  auto E1 = ptm::Event::Empty(omega.GetSize());
  auto E2 = ptm::Event::Full(omega.GetSize());

  EXPECT_FALSE(E1.Contains(a));
  EXPECT_TRUE(E2.Contains(a));
  EXPECT_TRUE(E2.Contains(b));
  EXPECT_TRUE(E2.Contains(c));

  auto E3 = Event::Complement(E2);
  EXPECT_FALSE(E3.Contains(a));
  EXPECT_FALSE(E3.Contains(b));
  EXPECT_FALSE(E3.Contains(c));
}

// Add your tests...

TEST(SigmaAlgebraTest, IsSigmaAlgebra_Trivial) {
  using namespace ptm;

  OutcomeSpace omega;
  omega.AddOutcome("1");
  omega.AddOutcome("2");
  omega.AddOutcome("3");

  auto empty = Event::Empty(omega.GetSize());
  auto full  = Event::Full(omega.GetSize());

  SigmaAlgebra F(omega, {empty, full});
  EXPECT_TRUE(F.IsSigmaAlgebra());
}


TEST(SigmaAlgebraTest, IsSigmaAlgebra_FailsWithoutComplement) {
  using namespace ptm;

  OutcomeSpace omega;
  auto w0 = omega.AddOutcome("1");
  auto w1 = omega.AddOutcome("2");
  auto w2 = omega.AddOutcome("3");

  // A = {1,3}
  std::vector<bool> mask(omega.GetSize(), false);
  mask[w0] = true;
  mask[w2] = true;
  Event A(mask);

  auto empty = Event::Empty(omega.GetSize());
  auto full  = Event::Full(omega.GetSize());

  SigmaAlgebra F(omega, {empty, full, A});
  EXPECT_FALSE(F.IsSigmaAlgebra());
}

TEST(SigmaAlgebraTest, IsSigmaAlgebra_FailsWithoutUnionClosure) {
  using namespace ptm;

  OutcomeSpace omega;
  auto w0 = omega.AddOutcome("1");
  auto w1 = omega.AddOutcome("2");
  auto w2 = omega.AddOutcome("3");

  auto empty = Event::Empty(omega.GetSize());
  auto full  = Event::Full(omega.GetSize());

  std::vector<bool> mA(omega.GetSize(), false), mB(omega.GetSize(), false);
  mA[w0] = true;
  mB[w1] = true;
  Event A(mA);
  Event B(mB);

  Event Ac = Event::Complement(A);
  Event Bc = Event::Complement(B);

  SigmaAlgebra F(omega, {empty, full, A, Ac, B, Bc});
  EXPECT_FALSE(F.IsSigmaAlgebra());
}

TEST(SigmaAlgebraTest, Generate_FromSingleGenerator) {
  using namespace ptm;

  OutcomeSpace omega;
  auto w0 = omega.AddOutcome("1");
  auto w1 = omega.AddOutcome("2");
  auto w2 = omega.AddOutcome("3");

  // A = {1,3}
  std::vector<bool> mask(omega.GetSize(), false);
  mask[w0] = true;
  mask[w2] = true;
  Event A(mask);

  auto F = SigmaAlgebra::Generate(omega, {A});

  EXPECT_TRUE(F.IsSigmaAlgebra());

  EXPECT_EQ(F.GetEvents().size(), 4u);

  Event Ac = Event::Complement(A);
  bool hasA = false, hasAc = false;
  for (const auto& e : F.GetEvents()) {
    if (e.GetMask() == A.GetMask())  hasA = true;
    if (e.GetMask() == Ac.GetMask()) hasAc = true;
  }
  EXPECT_TRUE(hasA);
  EXPECT_TRUE(hasAc);
}