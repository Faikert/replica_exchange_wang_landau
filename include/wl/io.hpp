#pragma once

#include "wl/analysis.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace wl {

struct RunConfig {
    std::size_t nx{2}, ny{2}, nz{2};
    double spacing{1.0};
    Vec3 axis{0.0, 0.0, 1.0};
    bool periodic{true};
    std::string geometry_file;
    std::string config_file;
    Vec3 box_lengths{2.0, 2.0, 2.0};
    double coupling_scale{1.0};
    double cutoff{0.0};
    EnergyGrid grid{-20.0, 20.0, 0.1};
    bool energy_grid_explicit{false};
    std::string order_parameter_mode{"none"};
    double q_bin_width{};
    std::shared_ptr<const WeightedOrderParameter> order_parameter;
    std::size_t windows{1};
    std::size_t walkers_per_rank{1};
    double overlap{0.75};
    std::vector<EnergyWindow> explicit_windows;
    std::vector<std::vector<std::int8_t>> initial_spins_by_walker;
    AdaptiveWindowParameters adaptive_windows;
    std::uint64_t seed{1};
    bool seed_explicit{false};
    WlParameters wl;
    std::uint64_t exchange_interval_attempts{1000};
    std::uint64_t max_attempts{10'000'000};
    std::uint64_t checkpoint_interval_attempts{1'000'000};
    double exchange_interval_mcs{10.0};
    double check_interval_mcs{100.0};
    double force_accept_after_mcs{};
    double max_mcs{1'000'000.0};
    double checkpoint_interval_mcs{10'000.0};
    bool exchange_interval_uses_mcs{true};
    bool check_interval_uses_mcs{true};
    bool force_accept_after_uses_mcs{true};
    bool max_limit_uses_mcs{true};
    bool checkpoint_interval_uses_mcs{true};
    std::size_t resolved_spin_count{};
    bool complete_range{false};
    std::string output_prefix{"wl_output"};
    std::string checkpoint_path;
    std::vector<double> temperatures{0.5, 1.0, 2.0, 5.0};
    bool smoke_test{false};
    bool pilot{false};
    double progress_interval_seconds{};

    void resolve_mcs(std::size_t spin_count);
    void resolve_order_parameter(const Geometry& geometry);
    [[nodiscard]] DosGrid dos_grid() const;
    [[nodiscard]] bool uses_legacy_attempt_units() const noexcept;
    void validate(bool require_explicit_grid = true) const;
};

struct WalkerStatistics {
    std::uint64_t walker_id{};
    std::uint64_t window_id{};
    int mpi_rank{};
    std::uint64_t attempted{};
    std::uint64_t accepted{};
    std::uint64_t forced_accepted{};
    std::uint64_t attempts_since_last_accepted{};
    double energy{};
    double factor{1.0};
    std::size_t active_bins{};
    std::uint64_t minimum_histogram{};
    double mean_histogram{};
    double min_over_mean{};
    std::uint64_t round_trips{};
};

[[nodiscard]] RunConfig parse_arguments(int argc, char** argv);
[[nodiscard]] std::string usage(std::string_view program);
void write_dos_csv(const std::string& path, const DensityOfStates& dos);
void write_joint_dos_csv(const std::string& path, const JointDensityOfStates& dos);
void write_fragment_csv(const std::string& path, EnergyGrid grid, const DosFragment& fragment,
                        std::size_t window_id);
void write_joint_fragment_csv(const std::string& path,const DosFragment& fragment,
                              double normalization,std::size_t window_id);
void write_thermodynamics_csv(const std::string& path,
                              std::span<const ThermodynamicPoint> points);
void write_order_parameter_thermodynamics_csv(
    const std::string& path,std::span<const OrderParameterThermodynamicPoint> points);
void write_order_parameter_distribution_csv(
    const std::string& path,std::span<const OrderParameterDistributionPoint> points);
void write_workers_stat_csv(const std::string& path,
                            std::span<const WalkerStatistics> statistics, std::size_t spin_count);
void write_metadata_json(const std::string& path, const RunConfig& config,
                         const Couplings& couplings, std::uint64_t attempted,
                         std::uint64_t accepted, std::uint64_t forced_accepted,
                         std::uint64_t exchange_attempted,
                         std::uint64_t exchange_accepted, bool converged, int mpi_size,
                         int openmp_threads,
                         std::string_view postprocessing_status = "complete",
                         std::span<const DosFragment> fragments = {},
                         std::size_t support_components = 0);
void save_checkpoint(const std::string& path, const WalkerSnapshot& snapshot);
[[nodiscard]] WalkerSnapshot load_checkpoint(const std::string& path);

} // namespace wl
