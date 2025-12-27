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

bool EventsEqual(const Event& a, const Event& b) {
  return a.GetMask() == b.GetMask();
}

bool ContainsEvent(const std::vector<Event>& events, const Event& target) {
  return std::any_of(events.begin(), events.end(), [&](const Event& e) { return EventsEqual(e, target); });
}

std::vector<Event> DeduplicateEvents(const std::vector<Event>& events) {
  std::vector<Event> out;
  out.reserve(events.size());
  for (const auto& e : events) {
    if (!ContainsEvent(out, e)) {
      out.push_back(e);
    }
  }
  return out;
}

bool SigmaAlgebra::IsSigmaAlgebra() const {
  const std::size_t n = omega_.GetSize();

  for (const auto& e : events_) {
    if (e.GetSize() != n) {
      return false;
    }
  }

  const Event empty = Event::Empty(n);
  const Event full = Event::Full(n);

  if (!ContainsEvent(events_, empty)) return false;
  if (!ContainsEvent(events_, full)) return false;

  for (const auto& e: events_) {
    const Event comp = Event::Complement(e);
    if (!ContainsEvent(events_, comp)) {
      return false;
    }
  }

  for (std::size_t i = 0; i < events_.size(); ++i) {
    for (std::size_t j = 0; j < events_.size(); ++j) {
      const Event u = Event::Unite(events_[i], events_[j]);
      if (!ContainsEvent(events_, u)) {
        return false;
      }
    }
  }

  return true;
}

std::string SignatureForOutcome(OutcomeSpace::OutcomeId id, const std::vector<Event>& generators) {
  std::string sig;
  sig.reserve(generators.size());
  for (const auto& g : generators) {
    sig.push_back(g.Contains(id) ? '1' : '0');
  }
  return sig;
}

SigmaAlgebra SigmaAlgebra::Generate(const OutcomeSpace& omega, const std::vector<Event>& generators) {
  const std::size_t n = omega.GetSize();

  if (generators.empty()) {
    return SigmaAlgebra(omega, {Event::Empty(n), Event::Full(n)});
  }

  std::unordered_map<std::string, std::vector<bool>> atoms_map;
  atoms_map.reserve(n * 2);

  for (std::size_t outcome = 0; outcome < n; ++outcome) {
    const std::string sig = SignatureForOutcome(outcome, generators);

    auto it = atoms_map.find(sig);
    if (it == atoms_map.end()) {
      std::vector<bool> mask(n, false);
      mask[outcome] = true;
      atoms_map.emplace(sig, std::move(mask));
    } else {
      it->second[outcome] = true;
    }
  }

  std::vector<Event> atoms;
  atoms.reserve(atoms_map.size());
  for (auto& kv : atoms_map) {
    atoms.emplace_back(std::move(kv.second));
  }

  const std::size_t m = atoms.size();

  if (m >= 63) {
    throw std::runtime_error("Too many atoms to generate full sigma-algebra");
  }

  const std::uint64_t total = (m == 64) ? 0 : (1ULL << m);

  std::vector<Event> events;
  events.reserve(static_cast<std::size_t>(total));

  for (std::uint64_t mask_atoms = 0; mask_atoms < total; ++mask_atoms) {
    std::vector<bool> mask_event(n, false);

    for (std::size_t i = 0; i < m; ++i) {
      if (mask_atoms & (1ULL << i)) {
        const auto& am = atoms[i].GetMask();
        for (std::size_t j = 0; j < n; ++j) {
          mask_event[j] = mask_event[j] || am[j];
        }
      }
    }

    events.emplace_back(std::move(mask_event));
  }

  events = DeduplicateEvents(events);
  return SigmaAlgebra(omega, std::move(events));
}

} // namesapce ptm