#include "Event.hpp"

namespace ptm {

Event::Event(std::vector<bool> mask) : mask_(std::move(mask)) {
}

size_t Event::GetSize() const noexcept {
    return mask_.size();
}

bool Event::Contains(OutcomeSpace::OutcomeId id) const {
    return id < mask_.size() && mask_[id];
}

const std::vector<bool>& Event::GetMask() const noexcept {
    return mask_;
}

Event Event::Empty(std::size_t n) {
    return Event(std::vector<bool>(n, false));
}

Event Event::Full(std::size_t n) {
    return Event(std::vector<bool>(n, true));
}

Event Event::Complement(const Event& e) {
    std::vector<bool> res(e.mask_.size());
    for (std::size_t i = 0; i < e.mask_.size(); ++i) {
        res[i] = !e.mask_[i];
    }
    return Event(std::move(res));
}

Event Event::Unite(const Event& a, const Event& b) {
    std::size_t n = std::min(a.mask_.size(), b.mask_.size());
    std::vector<bool> res(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        res[i] = a.mask_[i] || b.mask_[i];
    }
    return Event(std::move(res));
}

Event Event::Intersect(const Event& a, const Event& b) {
    std::size_t n = std::min(a.mask_.size(), b.mask_.size());
    std::vector<bool> res(n, false);
    for (std::size_t i = 0; i < n; ++i) {
        res[i] = a.mask_[i] && b.mask_[i];
    }
    return Event(std::move(res));
}

} // namespace ptm
