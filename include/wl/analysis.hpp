#pragma once

#include "wl/wang_landau.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string>
#include <vector>

namespace wl {

struct DosFragment {
    EnergyWindow window;
    std::vector<double> log_g;
    std::vector<std::uint64_t> histogram;
    std::vector<double> standard_error;
    std::vector<std::uint8_t> valid;
    DosGrid grid{};
    std::vector<std::uint32_t> contributors;
    std::vector<std::int32_t> support_component;
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

struct JointDensityOfStates {
    DosGrid grid;
    double normalization{};
    std::vector<double> log_g;
    std::vector<std::uint64_t> histogram;
    std::vector<double> standard_error;
    std::vector<std::uint8_t> valid;
    std::vector<std::uint32_t> contributors;
    std::vector<std::int32_t> support_component;
    bool fully_normalized{false};
};

class DisconnectedSupportError : public std::runtime_error {
public:
    explicit DisconnectedSupportError(std::size_t components);
    [[nodiscard]] std::size_t components() const noexcept { return components_; }
private:
    std::size_t components_{};
};

class InsufficientSupportError : public std::runtime_error {
public:
    explicit InsufficientSupportError(const std::string& message)
        : std::runtime_error(message) {}
};

struct OrderParameterThermodynamicPoint {
    double temperature{};
    double mean_q{},mean_abs_q{},mean_q2{},mean_q4{};
    double mean_Q{},mean_abs_Q{},mean_Q2{},mean_Q4{};
    double susceptibility{},binder_cumulant{};
};

struct OrderParameterDistributionPoint {
    double temperature{};
    std::size_t q_bin{};
    double Q{},q{},probability{},log_probability{},relative_free_energy{};
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
[[nodiscard]] DosFragment exact_enumeration(const Couplings& couplings, DosGrid grid,
    const WeightedOrderParameter& order_parameter, std::size_t max_spins = 26);
[[nodiscard]] JointDensityOfStates stitch_joint_dos(
    DosGrid grid, std::span<const DosFragment> fragments, bool complete_range,
    bool converged, std::size_t spin_count, double normalization);
[[nodiscard]] DensityOfStates marginalize(const JointDensityOfStates& joint);
[[nodiscard]] DosFragment marginalize_fragment(const DosFragment& joint);
[[nodiscard]] std::size_t support_component_count(const DosFragment& fragment);
[[nodiscard]] std::vector<OrderParameterThermodynamicPoint> order_parameter_thermodynamics(
    const JointDensityOfStates& dos, std::span<const double> temperatures,
    std::size_t spin_count, double boltzmann_constant = 1.0);
[[nodiscard]] std::vector<OrderParameterDistributionPoint> order_parameter_distribution(
    const JointDensityOfStates& dos, std::span<const double> temperatures,
    double boltzmann_constant = 1.0);
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
