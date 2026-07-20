#pragma once

#include "wl/physics.hpp"
#include "wl/rng.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace wl {

struct EnergyGrid {
    double minimum{};
    double maximum{};
    double width{};

    void validate() const;
    [[nodiscard]] std::size_t bins() const;
    [[nodiscard]] std::optional<std::size_t> index(double energy) const noexcept;
    [[nodiscard]] double center(std::size_t index) const noexcept;
};

struct EnergyWindow {
    std::size_t begin{};
    std::size_t end{}; // exclusive
    [[nodiscard]] bool contains(std::size_t index) const noexcept {
        return index >= begin && index < end;
    }
};

[[nodiscard]] std::vector<EnergyWindow> partition_windows(std::size_t bins,
                                                           std::size_t count,
                                                           double overlap);

struct WlParameters {
    double flatness{0.8};
    std::uint64_t minimum_visits{100};
    double final_factor{1e-8};
    std::uint64_t check_interval_attempts{10'000};
    std::uint64_t force_accept_after_attempts{}; // 0 disables non-standard forced acceptance
    bool inverse_time_enabled{true};
    std::uint64_t initialization_max_attempts{1'000'000};
    double initialization_target_fraction{0.5};
    double initialization_temperature_fraction{0.05};
    std::uint64_t initialization_stall_attempts_per_spin{1'000};
    double initialization_temperature_multiplier{2.0};
    double initialization_max_temperature_fraction{0.5};
};

enum class RefinementStage : std::uint8_t { wang_landau, inverse_time, frozen };

struct HistogramStatistics {
    std::size_t active_bins{};
    std::uint64_t minimum{};
    double mean{};
    double min_over_mean{};
};

struct WalkerSnapshot {
    std::uint64_t walker_id{};
    std::vector<std::int8_t> spins;
    std::vector<double> fields;
    double energy{};
    std::vector<double> log_g;
    std::vector<std::uint64_t> histogram;
    std::vector<std::uint8_t> active;
    double factor{1.0};
    std::uint64_t attempted{};
    std::uint64_t accepted{};
    std::uint64_t forced_accepted{};
    std::uint64_t last_accepted_attempt{};
    RefinementStage stage{RefinementStage::wang_landau};
    std::array<std::uint64_t, 4> rng_state{};
};

class WangLandauWalker {
public:
    WangLandauWalker(std::uint64_t walker_id, std::shared_ptr<const Couplings> couplings,
                     EnergyGrid grid, EnergyWindow window, WlParameters parameters,
                     std::uint64_t master_seed,
                     std::vector<std::int8_t> initial_spins = {});

    bool attempt_flip();
    void run_attempts(std::uint64_t count);
    [[nodiscard]] bool flat() const;
    [[nodiscard]] bool ready_for_iteration() const;
    void begin_next_iteration();
    void freeze_if_finished();
    [[nodiscard]] HistogramStatistics histogram_statistics() const noexcept;

    [[nodiscard]] WalkerSnapshot snapshot() const;
    void restore(const WalkerSnapshot& snapshot);
    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    [[nodiscard]] double energy() const noexcept { return energy_; }
    [[nodiscard]] std::optional<std::size_t> energy_bin() const noexcept { return grid_.index(energy_); }
    [[nodiscard]] RefinementStage stage() const noexcept { return stage_; }
    [[nodiscard]] double factor() const noexcept { return factor_; }
    [[nodiscard]] const EnergyWindow& window() const noexcept { return window_; }
    [[nodiscard]] const std::vector<double>& log_g() const noexcept { return log_g_; }
    [[nodiscard]] const std::vector<std::uint64_t>& histogram() const noexcept { return histogram_; }
    [[nodiscard]] const std::vector<std::uint8_t>& active_mask() const noexcept { return active_; }
    [[nodiscard]] std::uint64_t attempted() const noexcept { return attempted_; }
    [[nodiscard]] std::uint64_t accepted() const noexcept { return accepted_; }
    [[nodiscard]] std::uint64_t forced_accepted() const noexcept { return forced_accepted_; }
    [[nodiscard]] std::uint64_t last_accepted_attempt() const noexcept {
        return last_accepted_attempt_;
    }

    void replace_configuration(std::span<const std::int8_t> spins,
                               std::span<const double> fields, double energy);
    void swap_configuration(WangLandauWalker& other);
    [[nodiscard]] double exchange_log_probability(const WangLandauWalker& other) const;

private:
    std::uint64_t id_{};
    std::shared_ptr<const Couplings> couplings_;
    EnergyGrid grid_;
    EnergyWindow window_;
    WlParameters parameters_;
    Xoshiro256StarStar rng_;
    std::vector<std::int8_t> spins_;
    std::vector<double> fields_;
    double energy_{};
    std::vector<double> log_g_;
    std::vector<std::uint64_t> histogram_;
    std::vector<std::uint8_t> active_;
    std::size_t active_bin_count_{};
    double factor_{1.0};
    std::uint64_t attempted_{};
    std::uint64_t accepted_{};
    std::uint64_t forced_accepted_{};
    std::uint64_t last_accepted_attempt_{};
    RefinementStage stage_{RefinementStage::wang_landau};

    void update_current_bin();
    [[nodiscard]] std::size_t active_bins() const noexcept;
    void update_inverse_time_factor();
};

} // namespace wl
