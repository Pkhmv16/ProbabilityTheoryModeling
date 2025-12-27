#include "MarkovTextModel.hpp"

#include <cctype>
#include <string>
#include <vector>

namespace ptm {

MarkovTextModel::MarkovTextModel(TokenLevel level) : level_(level), chain_() {}

const MarkovChain& MarkovTextModel::Chain() const noexcept { return chain_; }

static bool IsWordChar(unsigned char c) {
    return std::isalnum(c) != 0 || c == '\'';
}

static bool IsPunctChar(unsigned char c) {
    const std::string punct = ".,!?;:()[]{}\"";
    return punct.find(static_cast<char>(c)) != std::string::npos;
}

static bool NoSpaceBefore(const std::string& tok) {
    if (tok.size() != 1) {
        return false;
    }

    const char c = tok[0];
    const std::string set = ".,!?;:)]}\"";
    return set.find(c) != std::string::npos;
}

static bool NoSpaceAfterPrev(const std::string& prev) {
    if (prev.size() != 1) {
        return false;
    }
    const char c = prev[0];
    const std::string set = "([{\"";
    return set.find(c) != std::string::npos;
}

void MarkovTextModel::TrainFromText(const std::string& text) {
    auto tokens = Tokenize(text);
    chain_.Train(tokens);
}

std::vector<std::string> MarkovTextModel::Tokenize(const std::string& text_in) const {
    std::string text = text_in;

    std::vector<std::string> tokens;

    if (level_ == TokenLevel::Character) {
        tokens.reserve(text.size());
        for (unsigned char c : text) {
            tokens.emplace_back(1, static_cast<char>(c));
        }
        return tokens;
    }

    std::string cur;

    auto flush_word = [&]() {
        if (cur.empty()) return;
        for (char& ch : cur) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
        tokens.push_back(cur);
        cur.clear();
    };

    for (unsigned char uc : text) {
        if (std::isspace(uc) != 0) {
            flush_word();
            continue;
        }

        if (IsWordChar(uc)) {
            cur.push_back(static_cast<char>(uc));
            continue;
        }

    flush_word();

    tokens.emplace_back(1, static_cast<char>(uc));
    }

    flush_word();
    return tokens;
}

std::string MarkovTextModel::Detokenize(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return {};

    if (level_ == TokenLevel::Character) {
        std::string out;
        out.reserve(tokens.size());
        for (const auto& t : tokens) {
            out += t;
        }
        return out;
    }

    std::string out;
    out.reserve(tokens.size() * 4);

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& tok = tokens[i];
        if (tok.empty()) continue;

        if (i == 0) {
            out += tok;
            continue;
        }

        const auto& prev = tokens[i - 1];

        if (tok.size() == 1 && IsPunctChar(static_cast<unsigned char>(tok[0])) && NoSpaceBefore(tok)) {
            out += tok;
            continue;
        }

        if (NoSpaceAfterPrev(prev)) {
            out += tok;
            continue;
        }

        out += ' ';
        out += tok;
    }

    return out;
}

std::string MarkovTextModel::GenerateText(std::size_t num_tokens,
                                         std::mt19937& rng,
                                         const std::string& start_token) const {
    if (num_tokens == 0) return {};

    auto states = chain_.States();
    if (states.empty()) return {};

    std::string start = start_token;
    if (level_ == TokenLevel::Word) {
        for (char& ch : start) {
            ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        }
    }

    if (start.empty() || std::ranges::find(states, start) == states.end()) {
        start = states.front();
    }

    auto generated_tokens = chain_.Generate(start, num_tokens, rng);
    return Detokenize(generated_tokens);
}

} // namespace ptm
