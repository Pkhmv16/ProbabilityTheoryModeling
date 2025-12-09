#include "SigmaAlgebra.hpp"

namespace ptm {

SigmaAlgebra::SigmaAlgebra(const OutcomeSpace& omega, std::vector<Event> events)
    : omega_(omega),
      events_(std::move(events)) {}

const OutcomeSpace& SigmaAlgebra::GetOutcomeSpace() const noexcept {
    return omega_;
}

const std::vector<Event>& SigmaAlgebra::GetEvents() const noexcept {
  return events_;
}

static bool EventsEqual(const Event& a, const Event& b) {
  return a.GetMask() == b.GetMask();
}






} // namesapce ptm