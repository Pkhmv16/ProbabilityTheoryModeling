#ifndef PTM_SIGMAALGEBRA_HPP_
#define PTM_SIGMAALGEBRA_HPP_

#include "Event.hpp"
#include "OutcomeSpace.hpp"

#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <cstdint>
#include <cstddef>

namespace ptm {

class SigmaAlgebra {
public:
  SigmaAlgebra(const OutcomeSpace& omega, std::vector<Event> events);

  [[nodiscard]] const OutcomeSpace& GetOutcomeSpace() const noexcept;
  [[nodiscard]] const std::vector<Event>& GetEvents() const noexcept;

  [[nodiscard]] bool IsSigmaAlgebra() const;

  // Построение сигма-алгебры из множества генераторов
  static SigmaAlgebra Generate(const OutcomeSpace& omega, const std::vector<Event>& generators);

private:
  const OutcomeSpace& omega_;
  std::vector<Event> events_;
};

} // namespace ptm

#endif // PTM_SIGMAALGEBRA_HPP_
