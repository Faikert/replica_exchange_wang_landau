#include "wl/wang_landau.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace wl {

void EnergyGrid::validate() const {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) ||
        !std::isfinite(width) || !(maximum > minimum) || !(width > 0.0))
        throw std::invalid_argument("Invalid energy grid");
    const auto count = (maximum - minimum) / width;
    if (std::abs(count - std::round(count)) > 1e-9 * std::max(1.0, count))
        throw std::invalid_argument("Energy range must be an integer multiple of bin width");
}

std::size_t EnergyGrid::bins() const {
    validate();
    return static_cast<std::size_t>(std::llround((maximum - minimum) / width));
}

std::optional<std::size_t> EnergyGrid::index(double energy) const noexcept {
    if (!std::isfinite(energy) || energy < minimum || energy > maximum) return std::nullopt;
    auto raw = static_cast<std::size_t>(std::floor((energy - minimum) / width));
    const auto count = static_cast<std::size_t>(std::llround((maximum - minimum) / width));
    if (raw == count && energy == maximum) raw = count - 1;
    if (raw >= count) return std::nullopt;
    return raw;
}

double EnergyGrid::center(std::size_t index_value) const noexcept {
    return minimum + (static_cast<double>(index_value) + 0.5) * width;
}

std::vector<EnergyWindow> partition_windows(std::size_t bins, std::size_t count,
                                            double overlap) {
    if (bins == 0 || count == 0 || count > bins || overlap < 0.0 || overlap >= 1.0)
        throw std::invalid_argument("Invalid energy-window partition");
    if (count == 1) return {{0, bins}};
    const auto denominator = 1.0 + static_cast<double>(count - 1) * (1.0 - overlap);
    auto window_size = static_cast<std::size_t>(std::ceil(static_cast<double>(bins) / denominator));
    window_size = std::max<std::size_t>(2, std::min(window_size, bins));
    const auto span = bins - window_size;
    std::vector<EnergyWindow> result;
    result.reserve(count);
    for (std::size_t w = 0; w < count; ++w) {
        const auto begin = static_cast<std::size_t>(std::llround(
            static_cast<double>(w * span) / static_cast<double>(count - 1)));
        result.push_back({begin, begin + window_size});
    }
    result.back().end = bins;
    for (std::size_t w = 1; w < result.size(); ++w)
        if (result[w].begin >= result[w-1].end)
            throw std::invalid_argument("Energy windows do not overlap; increase overlap or bins");
    return result;
}

WangLandauWalker::WangLandauWalker(std::uint64_t walker_id,
                                   std::shared_ptr<const Couplings> couplings,
                                   EnergyGrid grid, EnergyWindow window,
                                   WlParameters parameters, std::uint64_t master_seed,
                                   std::vector<std::int8_t> initial_spins)
    : id_(walker_id), couplings_(std::move(couplings)), grid_(grid), window_(window),
      parameters_(parameters), rng_(master_seed, walker_id) {
    if (!couplings_) throw std::invalid_argument("Null couplings");
    grid_.validate();
    if (window_.begin >= window_.end || window_.end > grid_.bins())
        throw std::invalid_argument("Invalid walker energy window");
    if (!(parameters_.flatness > 0.0 && parameters_.flatness <= 1.0) ||
        !(parameters_.final_factor > 0.0) || parameters_.check_interval_attempts == 0 ||
        parameters_.initialization_max_attempts == 0 ||
        !(parameters_.initialization_target_fraction > 0.0 &&
          parameters_.initialization_target_fraction <= 1.0) ||
        !std::isfinite(parameters_.initialization_temperature_fraction) ||
        !(parameters_.initialization_temperature_fraction > 0.0) ||
        parameters_.initialization_stall_attempts_per_spin == 0 ||
        !std::isfinite(parameters_.initialization_temperature_multiplier) ||
        !(parameters_.initialization_temperature_multiplier > 1.0) ||
        !std::isfinite(parameters_.initialization_max_temperature_fraction) ||
        parameters_.initialization_max_temperature_fraction <
            parameters_.initialization_temperature_fraction ||
        !std::isfinite(parameters_.round_trip_margin_fraction) ||
        parameters_.round_trip_margin_fraction < 0.0 ||
        parameters_.round_trip_margin_fraction >= 0.5)
        throw std::invalid_argument("Invalid WL parameters");
    const bool supplied_initial_configuration=!initial_spins.empty();
    if (!supplied_initial_configuration) initial_spins.assign(couplings_->size(), 1);
    if (initial_spins.size() != couplings_->size())
        throw std::invalid_argument("Initial spin count mismatch");
    for (const auto spin : initial_spins)
        if (spin != -1 && spin != 1) throw std::invalid_argument("Spins must be +/-1");
    spins_ = std::move(initial_spins);
    fields_ = local_fields(*couplings_, spins_);
    energy_ = total_energy(*couplings_, spins_);
    log_g_.assign(grid_.bins(), 0.0);
    histogram_.assign(grid_.bins(), 0);
    active_.assign(grid_.bins(), 0);
    if(parameters_.collect_window_statistics) {
        squared_energy_displacement_.assign(grid_.bins(),0.0);
        displacement_samples_.assign(grid_.bins(),0);
    }

    // An explicitly supplied in-window configuration is authoritative. Otherwise sample
    // pi(E) proportional to exp(-abs(E-E_target)/T_search). Raise T_search after stalls
    // and reproducibly randomize the spins after a stall at the maximum temperature.
    const auto initial_bin=grid_.index(energy_);
    if(supplied_initial_configuration && initial_bin && window_.contains(*initial_bin)) {
        update_current_bin();
        update_round_trip_state();
        return;
    }
    const auto lower=grid_.minimum+static_cast<double>(window_.begin)*grid_.width;
    const auto upper=grid_.minimum+static_cast<double>(window_.end)*grid_.width;
    const auto span=upper-lower;
    const auto target=0.5*(lower+upper);
    const auto target_half_width=0.5*parameters_.initialization_target_fraction*span;
    const auto initial_search_temperature=std::max(
        grid_.width,parameters_.initialization_temperature_fraction*span);
    const auto maximum_search_temperature=std::max(
        initial_search_temperature,parameters_.initialization_max_temperature_fraction*span);
    auto search_temperature=initial_search_temperature;
    const auto maximum_attempt_count=std::numeric_limits<std::uint64_t>::max();
    const auto spin_count=static_cast<std::uint64_t>(spins_.size());
    const auto stall_attempts=parameters_.initialization_stall_attempts_per_spin>
            maximum_attempt_count/spin_count?
        maximum_attempt_count:
        parameters_.initialization_stall_attempts_per_spin*spin_count;
    std::uint64_t attempts_since_improvement=0;
    std::uint64_t temperature_increases=0;
    std::uint64_t random_restarts=0;
    double highest_search_temperature=search_temperature;
    auto minimum_reached=energy_;
    auto maximum_reached=energy_;
    auto closest_reached=energy_;
    auto closest_distance=std::abs(energy_-target);
    const auto record_reached_energy = [&](double candidate) {
        minimum_reached=std::min(minimum_reached,candidate);
        maximum_reached=std::max(maximum_reached,candidate);
        const auto distance=std::abs(candidate-target);
        if(distance<closest_distance) {
            closest_distance=distance;
            closest_reached=candidate;
            return true;
        }
        return false;
    };
    const auto inside_target_band = [&](double candidate) {
        const auto bin=grid_.index(candidate);
        return bin && window_.contains(*bin) &&
               std::abs(candidate-target)<=target_half_width;
    };
    for (std::uint64_t step = 0; step < parameters_.initialization_max_attempts; ++step) {
        if (inside_target_band(energy_)) {
            update_current_bin(); update_round_trip_state(); return;
        }
        if(attempts_since_improvement>=stall_attempts) {
            if(search_temperature<maximum_search_temperature) {
                search_temperature=std::min(
                    maximum_search_temperature,
                    search_temperature*parameters_.initialization_temperature_multiplier);
                highest_search_temperature=std::max(highest_search_temperature,search_temperature);
                ++temperature_increases;
            } else {
                for(auto& spin:spins_)
                    spin=rng_.bounded(2)==0?std::int8_t{-1}:std::int8_t{1};
                fields_=local_fields(*couplings_,spins_);
                energy_=total_energy(*couplings_,spins_);
                record_reached_energy(energy_);
                search_temperature=initial_search_temperature;
                ++random_restarts;
            }
            attempts_since_improvement=0;
            if(inside_target_band(energy_)) {
                update_current_bin(); update_round_trip_state(); return;
            }
        }
        const auto i = static_cast<std::size_t>(rng_.bounded(spins_.size()));
        const auto old = spins_[i];
        const auto delta = flip_delta(i, spins_, fields_);
        const auto proposed = energy_ + delta;
        const auto log_acceptance=(std::abs(energy_-target)-std::abs(proposed-target))/
                                  search_temperature;
        bool improved=false;
        if (log_acceptance>=0.0 ||
            std::log(std::max(rng_.uniform(),0x1.0p-53))<log_acceptance) {
            couplings_->add_flip_delta(i, old, fields_);
            spins_[i] = static_cast<std::int8_t>(-old);
            energy_ = proposed;
            improved=record_reached_energy(energy_);
        }
        if(improved) attempts_since_improvement=0;
        else if(attempts_since_improvement<maximum_attempt_count)
            ++attempts_since_improvement;
    }
    if (inside_target_band(energy_)) {
        update_current_bin(); update_round_trip_state(); return;
    }
    std::ostringstream message;
    message<<std::setprecision(17)
           <<"Target Metropolis initialization did not reach target band ["
           <<target-target_half_width<<", "<<target+target_half_width
           <<"] for energy window bins ["<<window_.begin<<", "<<window_.end
           <<"), energy bounds ["<<lower<<", "<<upper
           <<(window_.end==grid_.bins()?']':')')<<" after "
           <<parameters_.initialization_max_attempts
           <<" attempts; reached energy range ["<<minimum_reached<<", "
           <<maximum_reached<<"], closest energy "<<closest_reached;
    if(const auto closest_bin=grid_.index(closest_reached))
        message<<" (bin "<<*closest_bin<<')';
    else
        message<<" (outside the energy grid)";
    message<<", distance to target band "
           <<std::max(0.0,closest_distance-target_half_width)
           <<"; adaptive search used "<<temperature_increases
           <<" temperature increases up to "<<highest_search_temperature
           <<" and "<<random_restarts<<" random restarts"
           <<" (stall interval "<<stall_attempts<<" attempts)";
    throw std::runtime_error(message.str());
}

bool WangLandauWalker::attempt_flip() {
    const auto i = static_cast<std::size_t>(rng_.bounded(spins_.size()));
    const auto old_bin = grid_.index(energy_);
    if (!old_bin || !window_.contains(*old_bin))
        throw std::runtime_error("Walker escaped its energy window");
    const auto old_spin = spins_[i];
    const auto delta = flip_delta(i, spins_, fields_);
    const auto proposed_energy = energy_ + delta;
    const auto new_bin = grid_.index(proposed_energy);
    bool accepted = false;
    bool forced = false;
    if (new_bin && window_.contains(*new_bin)) {
        const auto log_ratio = log_g_[*old_bin] - log_g_[*new_bin];
        const bool force_due = parameters_.force_accept_after_attempts != 0 &&
            attempted_ - last_accepted_attempt_ >= parameters_.force_accept_after_attempts;
        if (log_ratio >= 0.0) accepted = true;
        else if (force_due) accepted = forced = true;
        else accepted = std::log(std::max(rng_.uniform(), 0x1.0p-53)) < log_ratio;
    }
    ++attempted_;
    if (accepted) {
        couplings_->add_flip_delta(i, old_spin, fields_);
        spins_[i] = static_cast<std::int8_t>(-old_spin);
        energy_ = proposed_energy;
        ++accepted_;
        if (forced) ++forced_accepted_;
        last_accepted_attempt_ = attempted_;
    }
    if(parameters_.collect_window_statistics) {
        const auto displacement=accepted?delta:0.0;
        squared_energy_displacement_[*old_bin]+=displacement*displacement;
        ++displacement_samples_[*old_bin];
    }
    update_inverse_time_factor();
    update_current_bin();
    update_round_trip_state();
    return accepted;
}

void WangLandauWalker::run_attempts(std::uint64_t count) {
    for (std::uint64_t i = 0; i < count; ++i) attempt_flip();
}

void WangLandauWalker::update_current_bin() {
    const auto bin = grid_.index(energy_);
    if (!bin || !window_.contains(*bin)) throw std::runtime_error("Invalid current energy bin");
    ++histogram_[*bin];
    if (active_[*bin] == 0) {
        active_[*bin] = 1;
        ++active_bin_count_;
    }
    if (stage_ != RefinementStage::frozen) log_g_[*bin] += factor_;
}

std::size_t WangLandauWalker::active_bins() const noexcept {
    return active_bin_count_;
}

void WangLandauWalker::update_inverse_time_factor() {
    if (stage_ != RefinementStage::inverse_time) return;
    const auto count = active_bins();
    if (count != 0 && attempted_ != 0)
        factor_ = static_cast<double>(count) / static_cast<double>(attempted_);
    if (factor_ <= parameters_.final_factor) stage_ = RefinementStage::frozen;
}

void WangLandauWalker::update_round_trip_state() noexcept {
    if(!parameters_.collect_window_statistics) return;
    const auto lower=grid_.minimum+static_cast<double>(window_.begin)*grid_.width;
    const auto upper=grid_.minimum+static_cast<double>(window_.end)*grid_.width;
    const auto margin=parameters_.round_trip_margin_fraction*(upper-lower);
    if(energy_<=lower+margin) {
        if(round_trip_state_==2) {
            ++round_trips_;
            round_trip_state_=1;
        } else if(round_trip_state_==0) round_trip_state_=1;
    } else if(energy_>=upper-margin && round_trip_state_==1) {
        round_trip_state_=2;
    }
}

HistogramStatistics WangLandauWalker::histogram_statistics() const noexcept {
    HistogramStatistics statistics;
    long double total = 0.0L;
    statistics.minimum = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = window_.begin; i < window_.end; ++i) {
        if (!active_[i]) continue;
        ++statistics.active_bins;
        statistics.minimum = std::min(statistics.minimum, histogram_[i]);
        total += static_cast<long double>(histogram_[i]);
    }
    if (statistics.active_bins == 0) {
        statistics.minimum = 0;
        return statistics;
    }
    statistics.mean = static_cast<double>(total / static_cast<long double>(statistics.active_bins));
    statistics.min_over_mean = statistics.mean > 0.0 ?
        static_cast<double>(statistics.minimum) / statistics.mean : 0.0;
    return statistics;
}

bool WangLandauWalker::flat() const {
    if (stage_ != RefinementStage::wang_landau) return true;
    const auto statistics = histogram_statistics();
    return statistics.active_bins != 0 &&
           statistics.minimum >= parameters_.minimum_visits &&
           statistics.min_over_mean >= parameters_.flatness;
}

bool WangLandauWalker::ready_for_iteration() const {
    return stage_ == RefinementStage::wang_landau && flat();
}

void WangLandauWalker::begin_next_iteration() {
    if (stage_ != RefinementStage::wang_landau || !flat())
        throw std::logic_error("WL iteration is not complete");
    factor_ *= 0.5;
    std::fill(histogram_.begin() + static_cast<std::ptrdiff_t>(window_.begin),
              histogram_.begin() + static_cast<std::ptrdiff_t>(window_.end), 0);
    if(parameters_.inverse_time_enabled) {
        const auto inverse_time = attempted_ == 0 ? 1.0 :
            static_cast<double>(active_bins()) / static_cast<double>(attempted_);
        if (factor_ <= inverse_time) {
            stage_ = RefinementStage::inverse_time;
            factor_ = inverse_time;
        }
    }
    freeze_if_finished();
}

void WangLandauWalker::freeze_if_finished() {
    if (factor_ <= parameters_.final_factor) stage_ = RefinementStage::frozen;
}

WalkerSnapshot WangLandauWalker::snapshot() const {
    return {id_, spins_, fields_, energy_, log_g_, histogram_, active_, factor_, attempted_, accepted_,
            forced_accepted_, last_accepted_attempt_, stage_, rng_.state()};
}

void WangLandauWalker::restore(const WalkerSnapshot& s) {
    if (s.walker_id != id_ || s.spins.size() != couplings_->size() ||
        s.fields.size() != couplings_->size() || s.log_g.size() != grid_.bins() ||
        s.histogram.size() != grid_.bins() || s.active.size() != grid_.bins())
        throw std::invalid_argument("Checkpoint is incompatible with walker");
    if (s.last_accepted_attempt > s.attempted)
        throw std::invalid_argument("Checkpoint has an invalid last accepted attempt");
    if (s.forced_accepted > s.accepted)
        throw std::invalid_argument("Checkpoint has an invalid forced acceptance count");
    if (!parameters_.inverse_time_enabled && s.stage == RefinementStage::inverse_time)
        throw std::invalid_argument("Checkpoint uses disabled inverse-time refinement");
    spins_ = s.spins; fields_ = s.fields; energy_ = s.energy; log_g_ = s.log_g;
    histogram_ = s.histogram; active_ = s.active; factor_ = s.factor;
    attempted_ = s.attempted; accepted_ = s.accepted;
    active_bin_count_=0;
    for(std::size_t i=window_.begin;i<window_.end;++i) {
        active_[i]=active_[i]!=0?1:0;
        active_bin_count_+=active_[i];
    }
    forced_accepted_ = s.forced_accepted; last_accepted_attempt_ = s.last_accepted_attempt; stage_ = s.stage;
    rng_.set_state(s.rng_state);
    if(parameters_.collect_window_statistics) {
        std::fill(squared_energy_displacement_.begin(),squared_energy_displacement_.end(),0.0);
        std::fill(displacement_samples_.begin(),displacement_samples_.end(),0);
        round_trip_state_=0;
        round_trips_=0;
        update_round_trip_state();
    }
    const auto exact = total_energy(*couplings_, spins_);
    if (std::abs(exact - energy_) > 1e-9 * std::max(1.0, std::abs(exact)))
        throw std::runtime_error("Checkpoint energy does not match spin configuration");
    const auto exact_fields = local_fields(*couplings_, spins_);
    for (std::size_t i = 0; i < fields_.size(); ++i)
        if (std::abs(exact_fields[i] - fields_[i]) > 1e-9 * std::max(1.0, std::abs(exact_fields[i])))
            throw std::runtime_error("Checkpoint local fields do not match spin configuration");
}

void WangLandauWalker::replace_configuration(std::span<const std::int8_t> spins,
                                              std::span<const double> fields,
                                              double energy) {
    if (spins.size() != spins_.size() || fields.size() != fields_.size())
        throw std::invalid_argument("Replica configuration size mismatch");
    const auto bin = grid_.index(energy);
    if (!bin || !window_.contains(*bin))
        throw std::invalid_argument("Replica configuration is outside the walker window");
    std::copy(spins.begin(), spins.end(), spins_.begin());
    std::copy(fields.begin(), fields.end(), fields_.begin());
    energy_ = energy;
    update_round_trip_state();
}

void WangLandauWalker::swap_configuration(WangLandauWalker& other) {
    const auto this_bin = grid_.index(other.energy_);
    const auto other_bin = other.grid_.index(energy_);
    if (!this_bin || !other_bin || !window_.contains(*this_bin) || !other.window_.contains(*other_bin))
        throw std::invalid_argument("Replica exchange would violate an energy window");
    spins_.swap(other.spins_);
    fields_.swap(other.fields_);
    std::swap(energy_, other.energy_);
    update_round_trip_state();
    other.update_round_trip_state();
}

double WangLandauWalker::exchange_log_probability(const WangLandauWalker& other) const {
    const auto x_here = grid_.index(energy_);
    const auto y_here = grid_.index(other.energy_);
    const auto x_there = other.grid_.index(energy_);
    const auto y_there = other.grid_.index(other.energy_);
    if (!x_here || !y_here || !x_there || !y_there ||
        !window_.contains(*x_here) || !window_.contains(*y_here) ||
        !other.window_.contains(*x_there) || !other.window_.contains(*y_there))
        return -std::numeric_limits<double>::infinity();
    return log_g_[*x_here] - log_g_[*y_here] +
           other.log_g_[*y_there] - other.log_g_[*x_there];
}

} // namespace wl
