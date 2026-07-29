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

struct OrderParameterGrid {
    double minimum{};
    double maximum{};
    double width{};

    void validate() const;
    [[nodiscard]] std::size_t bins() const;
    [[nodiscard]] std::optional<std::size_t> index(double value) const noexcept;
    [[nodiscard]] double center(std::size_t index) const noexcept;
};

struct DosGrid {
    EnergyGrid energy;
    std::optional<OrderParameterGrid> order_parameter;

    void validate() const;
    [[nodiscard]] bool joint() const noexcept { return order_parameter.has_value(); }
    [[nodiscard]] std::size_t energy_bins() const { return energy.bins(); }
    [[nodiscard]] std::size_t q_bins() const { return order_parameter ? order_parameter->bins() : 1; }
    [[nodiscard]] std::size_t cells() const;
    [[nodiscard]] std::optional<std::size_t> index(double energy_value,
                                                   double q_value = 0.0) const noexcept;
    [[nodiscard]] std::size_t flatten(std::size_t energy_bin,
                                      std::size_t q_bin = 0) const noexcept;
    [[nodiscard]] std::size_t energy_bin(std::size_t cell) const noexcept;
    [[nodiscard]] std::size_t q_bin(std::size_t cell) const noexcept;
};

struct WeightedOrderParameter {
    std::vector<double> weights;
    double normalization{};
    OrderParameterGrid grid;

    static WeightedOrderParameter create(std::vector<double> weights, double bin_width);
    [[nodiscard]] double evaluate(std::span<const std::int8_t> spins) const;
    [[nodiscard]] double flip_delta(std::size_t index, std::int8_t old_spin) const noexcept {
        return -2.0 * weights[index] * static_cast<double>(old_spin);
    }
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
    bool collect_window_statistics{false};
    double round_trip_margin_fraction{0.1};
    bool nalivaiko_mod{false};
    std::size_t support_stability_checks{10};
    bool return_mode{false};
};

enum class RefinementStage : std::uint8_t { wang_landau, inverse_time, frozen };

struct HistogramStatistics {
    std::size_t active_bins{};
    std::size_t covered_bins{};
    std::uint64_t minimum{};
    double mean{};
    double min_over_mean{};
    double coverage{};
};

struct EnergyRepresentative {
    double energy{};
    std::vector<std::int8_t> spins;
};

struct WalkerSnapshot {
    std::uint8_t format_version{5};
    std::uint64_t walker_id{};
    EnergyGrid energy_grid{};
    EnergyWindow energy_window{};
    std::vector<std::int8_t> spins;
    std::vector<double> fields;
    double energy{};
    double order_parameter{};
    bool joint_dos{false};
    OrderParameterGrid order_grid{};
    double order_normalization{};
    std::vector<double> order_weights;
    std::vector<double> log_g;
    std::vector<std::uint64_t> histogram;
    std::vector<std::uint8_t> active;
    double factor{1.0};
    std::uint64_t attempted{};
    std::uint64_t accepted{};
    std::uint64_t forced_accepted{};
    std::uint64_t last_accepted_attempt{};
    std::uint64_t last_new_cell_attempt{};
    RefinementStage stage{RefinementStage::wang_landau};
    std::array<std::uint64_t, 4> rng_state{};
};

class WangLandauWalker {
public:
    WangLandauWalker(std::uint64_t walker_id, std::shared_ptr<const Couplings> couplings,
                     EnergyGrid grid, EnergyWindow window, WlParameters parameters,
                     std::uint64_t master_seed,
                     std::vector<std::int8_t> initial_spins = {},
                     std::vector<std::int8_t> return_spins = {});
    WangLandauWalker(std::uint64_t walker_id, std::shared_ptr<const Couplings> couplings,
                     DosGrid grid, EnergyWindow window, WlParameters parameters,
                     std::uint64_t master_seed,
                     std::shared_ptr<const WeightedOrderParameter> order_parameter,
                     std::vector<std::int8_t> initial_spins = {},
                     std::vector<std::int8_t> return_spins = {});

    bool attempt_flip();
    void run_attempts(std::uint64_t count);
    [[nodiscard]] bool flat() const;
    [[nodiscard]] bool covered() const;
    [[nodiscard]] bool ready_for_iteration() const;
    void begin_next_iteration();
    void freeze_if_finished();
    [[nodiscard]] HistogramStatistics histogram_statistics() const noexcept;

    [[nodiscard]] WalkerSnapshot snapshot() const;
    void restore(const WalkerSnapshot& snapshot);
    [[nodiscard]] std::uint64_t id() const noexcept { return id_; }
    [[nodiscard]] double energy() const noexcept { return energy_; }
    [[nodiscard]] double order_parameter() const noexcept { return order_parameter_value_; }
    [[nodiscard]] const DosGrid& dos_grid() const noexcept { return dos_grid_; }
    [[nodiscard]] std::optional<std::size_t> energy_bin() const noexcept {
        return current_location_valid_?std::optional<std::size_t>(current_energy_bin_):std::nullopt;
    }
    [[nodiscard]] RefinementStage stage() const noexcept { return stage_; }
    [[nodiscard]] double factor() const noexcept { return factor_; }
    [[nodiscard]] const EnergyWindow& window() const noexcept { return window_; }
    [[nodiscard]] const std::vector<double>& log_g() const noexcept { return log_g_; }
    [[nodiscard]] const std::vector<std::uint64_t>& histogram() const noexcept { return histogram_; }
    [[nodiscard]] const std::vector<std::uint8_t>& active_mask() const noexcept { return active_; }
    [[nodiscard]] const std::vector<double>& squared_energy_displacement() const noexcept {
        return squared_energy_displacement_;
    }
    [[nodiscard]] const std::vector<std::uint64_t>& displacement_samples() const noexcept {
        return displacement_samples_;
    }
    [[nodiscard]] std::uint64_t round_trips() const noexcept { return round_trips_; }
    [[nodiscard]] const std::vector<EnergyRepresentative>& representatives() const noexcept {
        return representatives_;
    }
    [[nodiscard]] std::uint64_t attempted() const noexcept { return attempted_; }
    [[nodiscard]] std::uint64_t accepted() const noexcept { return accepted_; }
    [[nodiscard]] std::uint64_t forced_accepted() const noexcept { return forced_accepted_; }
    [[nodiscard]] std::uint64_t last_accepted_attempt() const noexcept {
        return last_accepted_attempt_;
    }
    [[nodiscard]] std::span<const std::int8_t> spins() const noexcept { return spins_; }
    [[nodiscard]] std::span<const double> fields() const noexcept { return fields_; }

    void replace_configuration(std::span<const std::int8_t> spins,
                               std::span<const double> fields, double energy,
                               double order_parameter = 0.0);
    void swap_configuration_buffers(std::vector<std::int8_t>& spins,
                                    std::vector<double>& fields, double energy,
                                    double order_parameter = 0.0);
    void swap_configuration(WangLandauWalker& other);
    [[nodiscard]] double exchange_log_probability(const WangLandauWalker& other) const;

private:
    std::uint64_t id_{};
    std::shared_ptr<const Couplings> couplings_;
    EnergyGrid grid_;
    DosGrid dos_grid_;
    std::shared_ptr<const WeightedOrderParameter> order_parameter_;
    EnergyWindow window_;
    WlParameters parameters_;
    Xoshiro256StarStar rng_;
    std::vector<std::int8_t> spins_;
    std::vector<double> fields_;
    double energy_{};
    double order_parameter_value_{};
    std::vector<std::int8_t> return_spins_;
    std::vector<double> return_fields_;
    double return_energy_{};
    double return_order_parameter_{};
    std::vector<double> log_g_;
    std::vector<std::uint64_t> histogram_;
    std::vector<std::uint8_t> active_;
    std::vector<std::size_t> active_cells_;
    std::vector<std::size_t> iteration_cells_;
    std::uint64_t histogram_sum_{};
    std::uint64_t active_histogram_sum_{};
    std::vector<double> squared_energy_displacement_;
    std::vector<std::uint64_t> displacement_samples_;
    std::size_t energy_bin_count_{};
    std::size_t q_bin_count_{1};
    double inverse_energy_width_{};
    double inverse_q_width_{};
    std::size_t current_energy_bin_{};
    std::size_t current_q_bin_{};
    std::size_t current_cell_{};
    bool current_location_valid_{false};
    std::size_t active_bin_count_{};
    double factor_{1.0};
    std::uint64_t attempted_{};
    std::uint64_t accepted_{};
    std::uint64_t forced_accepted_{};
    std::uint64_t last_accepted_attempt_{};
    std::uint64_t last_new_cell_attempt_{};
    RefinementStage stage_{RefinementStage::wang_landau};
    std::uint8_t round_trip_state_{};
    std::uint64_t round_trips_{};
    std::vector<double> representative_targets_;
    std::vector<double> representative_distances_;
    std::vector<EnergyRepresentative> representatives_;

    struct StateLocation {
        std::size_t energy_bin{};
        std::size_t q_bin{};
        std::size_t cell{};
    };
    [[nodiscard]] std::optional<StateLocation> locate_state(
        double energy, double order_parameter) const noexcept;
    void set_current_location(const StateLocation& location) noexcept;
    void refresh_current_location();
    void update_current_bin();
    [[nodiscard]] std::optional<std::size_t> current_cell() const noexcept;
    [[nodiscard]] std::size_t active_bins() const noexcept;
    void update_inverse_time_factor();
    void update_round_trip_state() noexcept;
    void update_representatives();
    void initialize_in_window(bool supplied_initial_configuration);
    void return_to_reference_configuration();
};

} // namespace wl
