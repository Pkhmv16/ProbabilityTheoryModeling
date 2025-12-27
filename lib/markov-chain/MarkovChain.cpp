#include "MarkovChain.hpp"

#include <random>

namespace ptm {

size_t MarkovChain::ensureState(const State& s) {
    auto it = state_to_index_.find(s);
    if (it != state_to_index_.end()) {
        return it->second;
    }

    const size_t new_index = index_to_state_.size();
    state_to_index_[s] = new_index;
    index_to_state_.push_back(s);

    for (auto& row : counts_) {
        row.push_back(0);
    }
    counts_.push_back(std::vector<size_t>(new_index + 1, 0));
    row_sums_.push_back(0);

    return new_index;
}

void MarkovChain::Train(const std::vector<State>& sequence) {
    if (sequence.empty()) return;

    ensureState(sequence.front());
    if (sequence.size() < 2) return;

    for (size_t i = 0; i + 1 < sequence.size(); ++i) {
        const size_t from = ensureState(sequence[i]);
        const size_t to = ensureState(sequence[i + 1]);

        counts_[from][to] += 1;
        row_sums_[from] += 1;
    }
}

std::unordered_map<MarkovChain::State, double>
MarkovChain::NextDistribution(const State& current) const {
    std::unordered_map<State, double> result;

    auto it = state_to_index_.find(current);
    if (it == state_to_index_.end()) {
        return result;
    }

    const size_t i = it->second;
    const size_t sum = row_sums_[i];
    if (sum == 0) {
        return result;
    }

    for (size_t j = 0; j < index_to_state_.size(); ++j) {
        const size_t c = counts_[i][j];
        if (c == 0) {
            continue;
        }
        result[index_to_state_[j]] = static_cast<double>(c) / static_cast<double>(sum);
    }
    return result;
}

double MarkovChain::TransitionProbability(const State& from, const State& to) const {
    auto it_from = state_to_index_.find(from);
    if (it_from == state_to_index_.end()) {
        return 0.0;
    }

    auto it_to = state_to_index_.find(to);
    if (it_to == state_to_index_.end()) {
        return 0.0;
    }

    size_t i = it_from->second;
    const size_t j = it_to->second;

    const size_t sum = row_sums_[i];
    if (sum == 0) {
        return 0.0;
    }

    return static_cast<double>(counts_[i][j]) / static_cast<double>(sum);
}

std::optional<MarkovChain::State> MarkovChain::SampleNext(const State& current, std::mt19937& rng) const {
    auto it = state_to_index_.find(current);
    if (it == state_to_index_.end()) {
        return std::nullopt;
    }

    const size_t i = it->second;
    if (row_sums_[i] == 0) {
        return std::nullopt;
    }

    std::discrete_distribution<size_t> dist(counts_[i].begin(), counts_[i].end());
    const size_t j = dist(rng);
    return index_to_state_[j];
}

std::vector<MarkovChain::State> MarkovChain::Generate(const State& start, size_t length, std::mt19937& rng) const {
    std::vector<State> out;
    if (length == 0) {
        return out;
    }

    if (state_to_index_.find(start) == state_to_index_.end()) {
        return out;
    }

    out.reserve(length);
    out.push_back(start);

    State cur = start;
    while (out.size() < length) {
        auto next = SampleNext(cur, rng);
        if (!next.has_value()) {
            break;
        }
        out.push_back(*next);
        cur = *next;
    }
    return out;
}

std::vector<MarkovChain::State> MarkovChain::States() const {
    return index_to_state_;
}

} // namespace ptm
