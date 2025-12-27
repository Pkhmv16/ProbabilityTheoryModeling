#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>

#include "../lib/markov-chain/MarkovChain.hpp"
#include "../lib/markov-chain/MarkovTextModel.hpp"

TEST(MarkovChainTest, SimpleCountsAndProbabilities) {
  using namespace ptm;

  MarkovChain chain;
  std::vector<std::string> seq = {"A", "B", "A", "B", "A"};
  chain.Train(seq);

  EXPECT_NEAR(chain.TransitionProbability("A", "B"), 1.0, 1e-9);
  EXPECT_NEAR(chain.TransitionProbability("B", "A"), 1.0, 1e-9);
  EXPECT_NEAR(chain.TransitionProbability("A", "A"), 0.0, 1e-9);
}

TEST(MarkovChainTest, IncrementalTraining) {
  using namespace ptm;

  MarkovChain chain;
  chain.Train({"A", "B"});
  double p_ab1 = chain.TransitionProbability("A", "B");

  chain.Train({"A", "C"});
  double p_ab2 = chain.TransitionProbability("A", "B");
  double p_ac2 = chain.TransitionProbability("A", "C");

  EXPECT_NEAR(p_ab2, 0.5, 1e-9);
  EXPECT_NEAR(p_ac2, 0.5, 1e-9);
  EXPECT_NEAR(p_ab1, 1.0, 1e-9);
}

TEST(MarkovTextModelTest, WordLevelGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  std::string text = "hello world hello world hello";
  model.TrainFromText(text);

  std::mt19937 rng(123);

  auto& chain = model.Chain();
  double p = chain.TransitionProbability("hello", "world");
  EXPECT_NEAR(p, 1.0, 1e-9);

  std::string generated = model.GenerateText(5, rng, "hello");
  EXPECT_FALSE(generated.empty());
}

TEST(MarkovTextModelTest, CharacterLevelGeneration) {
  using namespace ptm;

  MarkovTextModel model(MarkovTextModel::TokenLevel::Character);
  std::string text = "ababa";
  model.TrainFromText(text);

  std::mt19937 rng(321);

  double p_ab = model.Chain().TransitionProbability("a", "b");
  double p_ba = model.Chain().TransitionProbability("b", "a");

  EXPECT_NEAR(p_ab, 1.0, 1e-9);
  EXPECT_NEAR(p_ba, 1.0, 1e-9);

  std::string generated = model.GenerateText(4, rng, "a");
  EXPECT_EQ(generated.size(), 4u);
}

TEST(MarkovTextModelTest, TrainOnWarAndPeaceWordLevel) {
  using namespace ptm;

  std::ifstream in("war_and_peace.txt");
  ASSERT_TRUE(in.good()) << "Couldn't open the file tests/war_and_peace.txt";

  std::stringstream buffer;
  buffer << in.rdbuf();
  std::string text = buffer.str();
  ASSERT_FALSE(text.empty()) << "The war and peace file is empty";

  MarkovTextModel model(MarkovTextModel::TokenLevel::Word);
  model.TrainFromText(text);

  const auto& chain = model.Chain();
  auto states = chain.States();

  EXPECT_GT(states.size(), 5000u) << "The dictionary is too small, and the text seems to be truncated.";

  auto has_token = [&](const std::string& token) { return std::ranges::find(states, token) != states.end(); };

  EXPECT_TRUE(has_token("the")) << "The word \"the\" is not found in the dictionary";
  EXPECT_TRUE(has_token("and")) << "The word \"and\" is not found in the dictionary";
  EXPECT_TRUE(has_token("to")) << "The word \"to\" is not found in the dictionary";

  std::mt19937 rng(123);

  std::string generated = model.GenerateText(50, rng, "the");
  EXPECT_FALSE(generated.empty()) << "The generated text is empty";

  std::size_t space_count = std::ranges::count(generated, ' ');
  EXPECT_GT(space_count, 5u);
}

TEST(MarkovChainTest, NextDistributionSumsToOneAndMatchesProbabilities) {
  using namespace ptm;

  MarkovChain chain;
  chain.Train({"A", "B", "A", "C"}); 

  auto dist = chain.NextDistribution("A");

  double sum = 0.0;
  for (const auto& [tok, p] : dist) sum += p;

  EXPECT_NEAR(sum, 1.0, 1e-9);
  EXPECT_NEAR(dist["B"], 0.5, 1e-9);
  EXPECT_NEAR(dist["C"], 0.5, 1e-9);

  
  EXPECT_NEAR(chain.TransitionProbability("A", "B"), dist["B"], 1e-9);
  EXPECT_NEAR(chain.TransitionProbability("A", "C"), dist["C"], 1e-9);
}

TEST(MarkovChainTest, UnknownStateGivesZeroProbabilityAndEmptyDistribution) {
  using namespace ptm;

  MarkovChain chain;
  chain.Train({"A", "B"});

  EXPECT_NEAR(chain.TransitionProbability("X", "Y"), 0.0, 1e-9);
  EXPECT_NEAR(chain.TransitionProbability("A", "Y"), 0.0, 1e-9);

  auto dist = chain.NextDistribution("X");
  EXPECT_TRUE(dist.empty());
}

TEST(MarkovChainTest, SampleNextOnDeadEndReturnsNullopt) {
  using namespace ptm;

  MarkovChain chain;
  chain.Train({"A", "B"}); 

  std::mt19937 rng(123);
  auto next = chain.SampleNext("B", rng);

  EXPECT_FALSE(next.has_value());
}

TEST(MarkovChainTest, GenerateLengthOneReturnsOnlyStart) {
  using namespace ptm;

  MarkovChain chain;
  chain.Train({"A", "B", "C"});

  std::mt19937 rng(1);
  auto seq = chain.Generate("A", 1, rng);

  ASSERT_EQ(seq.size(), 1u);
  EXPECT_EQ(seq[0], "A");
}

// Add your tests...
