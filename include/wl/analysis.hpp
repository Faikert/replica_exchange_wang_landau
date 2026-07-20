#pragma once

#include "wl/wang_landau.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace wl {

struct DosFragment {
    EnergyWindow window;
    std::vector<double> log_g;
    std::vector<std::uint64_t> histogram;
    std::vector<double> standard_error;
    std::vector<std::uint8_t> valid;
};

struct DensityOfStates {
    EnergyGrid grid;
    std::vector<double> log_g;
    std::vector<std::uint64_t> histogram;
    std::vector<double> standard_error;
    std::vector<std::uint8_t> valid;
    std::vector<std::size_t> join_bins;
    bool fully_normalized{false};
};

struct ThermodynamicPoint {
    double temperature{};
    double log_partition{};
    double internal_energy{};
    double heat_capacity{};
    double free_energy{};
    double entropy{};
};

[[nodiscard]] DensityOfStates stitch_dos(EnergyGrid grid,
                                         std::span<const DosFragment> fragments,
                                         bool complete_range, std::size_t spin_count);
[[nodiscard]] std::vector<ThermodynamicPoint> thermodynamics(
    const DensityOfStates& dos, std::span<const double> temperatures,
    double boltzmann_constant = 1.0);
[[nodiscard]] DosFragment exact_enumeration(const Couplings& couplings, EnergyGrid grid,
                                            std::size_t max_spins = 26);

} // namespace wl
