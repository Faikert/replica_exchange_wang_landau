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

struct WindowSamplingStatistics {
    EnergyWindow window;
    std::vector<double> squared_energy_displacement;
    std::vector<std::uint64_t> displacement_samples;
    std::uint64_t round_trips{};
    std::uint64_t walkers{};
};

struct AdaptiveWindowParameters {
    bool enabled{false};
    std::size_t iterations{2};
    double pilot_mcs{10'000.0};
    double smoothing_width{}; // energy units; zero selects an automatic physical width
    double minimum_width{};   // energy units; zero selects one quarter of the mean core width
    double diffusivity_floor_fraction{0.05};
    double curvature_weight{0.25};
    double round_trip_target{2.0};
    double maximum_round_trip_penalty{3.0};
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
[[nodiscard]] std::vector<EnergyWindow> adapt_energy_windows(
    EnergyGrid grid, std::span<const DosFragment> fragments,
    std::span<const WindowSamplingStatistics> sampling, std::size_t window_count,
    double overlap, const AdaptiveWindowParameters& parameters);
[[nodiscard]] std::vector<std::vector<std::int8_t>> select_adaptive_initial_configurations(
    const EnergyGrid& grid, std::span<const EnergyWindow> windows,
    std::span<const EnergyRepresentative> representatives, int mpi_size,
    std::size_t walkers_per_rank, std::size_t& missing,
    std::size_t& external_warm_starts);

} // namespace wl
