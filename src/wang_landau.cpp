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
    const auto maximum_count=static_cast<long double>(std::min(
        std::numeric_limits<std::size_t>::max(),
        static_cast<std::size_t>(std::numeric_limits<long long>::max())));
    if(!(count>=1.0) || static_cast<long double>(count)>maximum_count)
        throw std::overflow_error("Energy grid has too many bins");
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

void OrderParameterGrid::validate() const {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || !std::isfinite(width) ||
        !(maximum > minimum) || !(width > 0.0))
        throw std::invalid_argument("Invalid order-parameter grid");
    const auto count=(maximum-minimum)/width;
    const auto maximum_count=static_cast<long double>(std::min(
        std::numeric_limits<std::size_t>::max(),
        static_cast<std::size_t>(std::numeric_limits<long long>::max())));
    if(!(count>=1.0) || static_cast<long double>(count)>maximum_count)
        throw std::overflow_error("Order-parameter grid has too many bins");
    if(std::abs(count-std::round(count))>1e-9*std::max(1.0,count))
        throw std::invalid_argument("Order-parameter range must be an integer multiple of bin width");
}

std::size_t OrderParameterGrid::bins() const {
    validate();
    return static_cast<std::size_t>(std::llround((maximum-minimum)/width));
}

std::optional<std::size_t> OrderParameterGrid::index(double value) const noexcept {
    if(!std::isfinite(value)||value<minimum||value>maximum) return std::nullopt;
    const auto count=static_cast<std::size_t>(std::llround((maximum-minimum)/width));
    auto raw=static_cast<std::size_t>(std::floor((value-minimum)/width));
    if(raw==count&&value==maximum) raw=count-1;
    if(raw>=count) return std::nullopt;
    return raw;
}

double OrderParameterGrid::center(std::size_t index_value) const noexcept {
    return minimum+(static_cast<double>(index_value)+0.5)*width;
}

void DosGrid::validate() const {
    energy.validate();
    if(order_parameter) order_parameter->validate();
    (void)cells();
}

std::size_t DosGrid::cells() const {
    const auto e=energy.bins(),q=q_bins();
    if(q!=0&&e>std::numeric_limits<std::size_t>::max()/q)
        throw std::overflow_error("DOS grid cell count overflows size_t");
    return e*q;
}

std::optional<std::size_t> DosGrid::index(double energy_value,double q_value) const noexcept {
    const auto e=energy.index(energy_value);
    if(!e) return std::nullopt;
    if(!order_parameter) return *e;
    const auto q=order_parameter->index(q_value);
    if(!q) return std::nullopt;
    return flatten(*e,*q);
}

std::size_t DosGrid::flatten(std::size_t e,std::size_t q) const noexcept {
    const auto count=order_parameter?static_cast<std::size_t>(std::llround(
        (order_parameter->maximum-order_parameter->minimum)/order_parameter->width)):1;
    return e*count+q;
}

std::size_t DosGrid::energy_bin(std::size_t cell) const noexcept {
    const auto count=order_parameter?static_cast<std::size_t>(std::llround(
        (order_parameter->maximum-order_parameter->minimum)/order_parameter->width)):1;
    return cell/count;
}
std::size_t DosGrid::q_bin(std::size_t cell) const noexcept {
    const auto count=order_parameter?static_cast<std::size_t>(std::llround(
        (order_parameter->maximum-order_parameter->minimum)/order_parameter->width)):1;
    return cell%count;
}

WeightedOrderParameter WeightedOrderParameter::create(std::vector<double> weights,
                                                       double bin_width) {
    if(weights.empty()||!std::isfinite(bin_width)||!(bin_width>0.0))
        throw std::invalid_argument("Weighted order parameter requires weights and a positive bin width");
    long double total=0.0L;
    for(const auto weight:weights) {
        if(!std::isfinite(weight)) throw std::invalid_argument("Order-parameter weights must be finite");
        total+=std::abs(static_cast<long double>(weight));
    }
    const auto normalization=static_cast<double>(total);
    if(!(normalization>0.0)||!std::isfinite(normalization))
        throw std::invalid_argument("Order-parameter normalization must be finite and positive");
    const auto ratio=total/static_cast<long double>(bin_width);
    const auto required=std::max(0.0L,std::ceil(ratio-0.5L));
    const auto maximum_half=(std::numeric_limits<std::size_t>::max()-1)/2;
    if(required>static_cast<long double>(maximum_half))
        throw std::overflow_error("Order-parameter grid has too many bins");
    auto half_bins=static_cast<std::size_t>(required);
    auto bin_count=2*half_bins+1;
    auto extent=0.5*static_cast<double>(bin_count)*bin_width;
    if(!std::isfinite(extent)) throw std::overflow_error("Order-parameter grid extent overflows");
    if(extent<normalization) {
        if(half_bins==maximum_half) throw std::overflow_error("Order-parameter grid has too many bins");
        ++half_bins;
        bin_count=2*half_bins+1;
        extent=0.5*static_cast<double>(bin_count)*bin_width;
    }
    if(!std::isfinite(extent)||extent<normalization)
        throw std::overflow_error("Order-parameter grid cannot represent its full support");
    return {std::move(weights),normalization,{-extent,extent,bin_width}};
}

double WeightedOrderParameter::evaluate(std::span<const std::int8_t> spins) const {
    if(spins.size()!=weights.size()) throw std::invalid_argument("Order-parameter spin count mismatch");
    long double value=0.0L;
    for(std::size_t i=0;i<spins.size();++i) value+=static_cast<long double>(weights[i])*spins[i];
    return static_cast<double>(value);
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
    : WangLandauWalker(walker_id,std::move(couplings),DosGrid{grid,std::nullopt},window,
                       parameters,master_seed,nullptr,std::move(initial_spins)) {}

WangLandauWalker::WangLandauWalker(std::uint64_t walker_id,
                                   std::shared_ptr<const Couplings> couplings,
                                   DosGrid dos_grid, EnergyWindow window,
                                   WlParameters parameters, std::uint64_t master_seed,
                                   std::shared_ptr<const WeightedOrderParameter> order_parameter,
                                   std::vector<std::int8_t> initial_spins)
    : id_(walker_id), couplings_(std::move(couplings)), grid_(dos_grid.energy),
      dos_grid_(std::move(dos_grid)), order_parameter_(std::move(order_parameter)),
      window_(window), parameters_(parameters), rng_(master_seed, walker_id) {
    if (!couplings_) throw std::invalid_argument("Null couplings");
    dos_grid_.validate();
    if(dos_grid_.joint()!=static_cast<bool>(order_parameter_))
        throw std::invalid_argument("DOS grid and order-parameter definition disagree");
    if(order_parameter_) {
        if(order_parameter_->weights.size()!=couplings_->size())
            throw std::invalid_argument("Order-parameter weight count mismatch");
        const auto& expected=order_parameter_->grid;
        const auto& actual=*dos_grid_.order_parameter;
        if(expected.minimum!=actual.minimum||expected.maximum!=actual.maximum||
           expected.width!=actual.width)
            throw std::invalid_argument("DOS grid uses a different order-parameter axis");
    }
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
        parameters_.round_trip_margin_fraction >= 0.5 ||
        (dos_grid_.joint() && parameters_.support_stability_checks==0))
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
    if(order_parameter_) order_parameter_value_=order_parameter_->evaluate(spins_);
    log_g_.assign(dos_grid_.cells(), 0.0);
    histogram_.assign(dos_grid_.cells(), 0);
    active_.assign(dos_grid_.cells(), 0);
    if(parameters_.nalivaiko_mod) refinement_active_.assign(dos_grid_.cells(), 0);
    if(parameters_.collect_window_statistics) {
        squared_energy_displacement_.assign(grid_.bins(),0.0);
        displacement_samples_.assign(grid_.bins(),0);
        constexpr std::size_t representative_count=9;
        const auto lower=grid_.minimum+static_cast<double>(window_.begin)*grid_.width;
        const auto upper=grid_.minimum+static_cast<double>(window_.end)*grid_.width;
        representative_targets_.reserve(representative_count);
        representative_targets_.push_back(lower);
        for(std::size_t r=1;r<representative_count;++r)
            representative_targets_.push_back(lower+(upper-lower)*
                (static_cast<double>(r)-0.5)/static_cast<double>(representative_count-1));
        representative_distances_.assign(representative_count,
                                          std::numeric_limits<double>::infinity());
        representatives_.resize(representative_count);
    }

    // An explicitly supplied in-window configuration is authoritative. Otherwise sample
    // pi(E) proportional to exp(-abs(E-E_target)/T_search). Raise T_search after stalls
    // and reproducibly randomize the spins after a stall at the maximum temperature.
    const auto initial_bin=grid_.index(energy_);
    if(supplied_initial_configuration && initial_bin && window_.contains(*initial_bin)) {
        update_current_bin();
        update_round_trip_state();
        update_representatives();
        return;
    }
    const auto lower=grid_.minimum+static_cast<double>(window_.begin)*grid_.width;
    const auto upper=grid_.minimum+static_cast<double>(window_.end)*grid_.width;
    const auto span=upper-lower;
    const auto target=0.5*(lower+upper);
    // A supplied configuration outside the window is an adaptive warm start. It only
    // needs to enter the window; an automatically generated start still targets the
    // configured central fraction.
    const auto target_fraction=supplied_initial_configuration?1.0:
        parameters_.initialization_target_fraction;
    const auto target_half_width=0.5*target_fraction*span;
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
            update_current_bin(); update_round_trip_state(); update_representatives(); return;
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
                if(order_parameter_) order_parameter_value_=order_parameter_->evaluate(spins_);
                record_reached_energy(energy_);
                search_temperature=initial_search_temperature;
                ++random_restarts;
            }
            attempts_since_improvement=0;
            if(inside_target_band(energy_)) {
                update_current_bin(); update_round_trip_state(); update_representatives(); return;
            }
        }
        const auto i = static_cast<std::size_t>(rng_.bounded(spins_.size()));
        const auto old = spins_[i];
        const auto delta = flip_delta(i, spins_, fields_);
        const auto q_delta=order_parameter_?order_parameter_->flip_delta(i,old):0.0;
        const auto proposed = energy_ + delta;
        const auto log_acceptance=(std::abs(energy_-target)-std::abs(proposed-target))/
                                  search_temperature;
        bool improved=false;
        if (log_acceptance>=0.0 ||
            std::log(std::max(rng_.uniform(),0x1.0p-53))<log_acceptance) {
            couplings_->add_flip_delta(i, old, fields_);
            spins_[i] = static_cast<std::int8_t>(-old);
            energy_ = proposed;
            order_parameter_value_+=q_delta;
            improved=record_reached_energy(energy_);
        }
        if(improved) attempts_since_improvement=0;
        else if(attempts_since_improvement<maximum_attempt_count)
            ++attempts_since_improvement;
    }
    if (inside_target_band(energy_)) {
        update_current_bin(); update_round_trip_state(); update_representatives(); return;
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
    const auto q_delta=order_parameter_?order_parameter_->flip_delta(i,old_spin):0.0;
    const auto proposed_energy = energy_ + delta;
    const auto new_bin = grid_.index(proposed_energy);
    const auto old_cell=dos_grid_.index(energy_,order_parameter_value_);
    const auto new_cell=dos_grid_.index(proposed_energy,order_parameter_value_+q_delta);
    if(!old_cell) throw std::runtime_error("Current state is outside the DOS grid");
    bool accepted = false;
    bool forced = false;
    if (new_bin && new_cell && window_.contains(*new_bin)) {
        const auto log_ratio = log_g_[*old_cell] - log_g_[*new_cell];
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
        order_parameter_value_ += q_delta;
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
    if(accepted) {
        update_round_trip_state();
        update_representatives();
    }
    return accepted;
}

void WangLandauWalker::run_attempts(std::uint64_t count) {
    for (std::uint64_t i = 0; i < count; ++i) attempt_flip();
}

void WangLandauWalker::update_current_bin() {
    const auto energy_bin=grid_.index(energy_);
    const auto cell=current_cell();
    if (!energy_bin || !cell || !window_.contains(*energy_bin))
        throw std::runtime_error("Invalid current DOS cell");
    ++histogram_[*cell];
    if (active_[*cell] == 0) {
        active_[*cell] = 1;
        ++active_bin_count_;
        last_new_cell_attempt_=attempted_;
    }
    if (parameters_.nalivaiko_mod && refinement_active_[*cell] == 0) {
        refinement_active_[*cell] = 1;
        ++refinement_active_bin_count_;
    }
    if (stage_ != RefinementStage::frozen) log_g_[*cell] += factor_;
}

std::optional<std::size_t> WangLandauWalker::current_cell() const noexcept {
    return dos_grid_.index(energy_,order_parameter_value_);
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

void WangLandauWalker::update_representatives() {
    if(!parameters_.collect_window_statistics) return;
    for(std::size_t r=0;r<representative_targets_.size();++r) {
        const auto distance=std::abs(energy_-representative_targets_[r]);
        if(distance<representative_distances_[r]) {
            representative_distances_[r]=distance;
            representatives_[r].energy=energy_;
            representatives_[r].spins=spins_;
        }
    }
}

HistogramStatistics WangLandauWalker::histogram_statistics() const noexcept {
    HistogramStatistics statistics;
    const auto& refinement_mask=parameters_.nalivaiko_mod?refinement_active_:active_;
    statistics.active_bins=parameters_.nalivaiko_mod?
        refinement_active_bin_count_:active_bin_count_;
    long double total = 0.0L;
    statistics.minimum = std::numeric_limits<std::uint64_t>::max();
    const auto begin=dos_grid_.flatten(window_.begin);
    const auto end=dos_grid_.flatten(window_.end);
    for (std::size_t i = begin; i < end; ++i) {
        if (!refinement_mask[i]) continue;
        if(histogram_[i]!=0) ++statistics.covered_bins;
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
    statistics.coverage=static_cast<double>(statistics.covered_bins)/
                        static_cast<double>(statistics.active_bins);
    return statistics;
}

bool WangLandauWalker::flat() const {
    if (stage_ != RefinementStage::wang_landau) return true;
    const auto statistics = histogram_statistics();
    return statistics.active_bins != 0 &&
           statistics.minimum >= parameters_.minimum_visits &&
           statistics.min_over_mean >= parameters_.flatness;
}

bool WangLandauWalker::covered() const {
    if(stage_!=RefinementStage::wang_landau) return true;
    const auto statistics=histogram_statistics();
    return statistics.active_bins!=0 &&
           statistics.covered_bins==statistics.active_bins;
}

bool WangLandauWalker::ready_for_iteration() const {
    if(stage_!=RefinementStage::wang_landau ||
       !(parameters_.inverse_time_enabled?covered():flat())) return false;
    if(!dos_grid_.joint()) return true;
    const auto checks=parameters_.support_stability_checks;
    const auto interval=parameters_.check_interval_attempts;
    const auto required=checks>std::numeric_limits<std::uint64_t>::max()/interval?
        std::numeric_limits<std::uint64_t>::max():static_cast<std::uint64_t>(checks)*interval;
    return attempted_-last_new_cell_attempt_>=required;
}

void WangLandauWalker::begin_next_iteration() {
    if(!ready_for_iteration())
        throw std::logic_error("WL iteration is not complete");
    factor_ *= 0.5;
    const auto begin=dos_grid_.flatten(window_.begin);
    const auto end=dos_grid_.flatten(window_.end);
    std::fill(histogram_.begin() + static_cast<std::ptrdiff_t>(begin),
              histogram_.begin() + static_cast<std::ptrdiff_t>(end), 0);
    if(parameters_.nalivaiko_mod) {
        std::fill(refinement_active_.begin() + static_cast<std::ptrdiff_t>(begin),
                  refinement_active_.begin() + static_cast<std::ptrdiff_t>(end), 0);
        refinement_active_bin_count_=0;
    }
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
    WalkerSnapshot result;
    result.format_version=5; result.walker_id=id_; result.energy_grid=grid_;
    result.energy_window=window_; result.spins=spins_; result.fields=fields_; result.energy=energy_;
    result.order_parameter=order_parameter_value_; result.joint_dos=dos_grid_.joint();
    if(dos_grid_.order_parameter) result.order_grid=*dos_grid_.order_parameter;
    if(order_parameter_) {
        result.order_normalization=order_parameter_->normalization;
        result.order_weights=order_parameter_->weights;
    }
    result.log_g=log_g_; result.histogram=histogram_; result.active=active_;
    result.factor=factor_; result.attempted=attempted_; result.accepted=accepted_;
    result.forced_accepted=forced_accepted_; result.last_accepted_attempt=last_accepted_attempt_;
    result.last_new_cell_attempt=last_new_cell_attempt_; result.stage=stage_;
    result.rng_state=rng_.state();
    return result;
}

void WangLandauWalker::restore(const WalkerSnapshot& s) {
    if (s.walker_id != id_ || s.spins.size() != couplings_->size() ||
        s.fields.size() != couplings_->size() || s.log_g.size() != dos_grid_.cells() ||
        s.histogram.size() != dos_grid_.cells() || s.active.size() != dos_grid_.cells() ||
        s.joint_dos!=dos_grid_.joint())
        throw std::invalid_argument("Checkpoint is incompatible with walker");
    if(s.format_version==0||s.format_version>5)
        throw std::invalid_argument("Checkpoint has an unsupported version");
    if(s.format_version<5&&dos_grid_.joint())
        throw std::invalid_argument("Legacy checkpoints cannot be restored in joint-DOS mode");
    if(s.format_version>=5 &&
       (s.energy_grid.minimum!=grid_.minimum||s.energy_grid.maximum!=grid_.maximum||
        s.energy_grid.width!=grid_.width||s.energy_window.begin!=window_.begin||
        s.energy_window.end!=window_.end))
        throw std::invalid_argument("Checkpoint has an incompatible energy layout");
    if(dos_grid_.joint()) {
        const auto& grid=*dos_grid_.order_parameter;
        if(s.order_grid.minimum!=grid.minimum||s.order_grid.maximum!=grid.maximum||
           s.order_grid.width!=grid.width||!order_parameter_||
           s.order_normalization!=order_parameter_->normalization||
           s.order_weights!=order_parameter_->weights)
            throw std::invalid_argument("Checkpoint has an incompatible order-parameter grid");
    }
    if(s.accepted>s.attempted||s.last_accepted_attempt>s.attempted||
       s.last_new_cell_attempt>s.attempted)
        throw std::invalid_argument("Checkpoint has invalid attempt counters");
    if(s.forced_accepted>s.accepted)
        throw std::invalid_argument("Checkpoint has an invalid forced acceptance count");
    if(s.stage!=RefinementStage::wang_landau&&s.stage!=RefinementStage::inverse_time&&
       s.stage!=RefinementStage::frozen)
        throw std::invalid_argument("Checkpoint has an invalid refinement stage");
    if(!std::isfinite(s.factor)||!(s.factor>0.0))
        throw std::invalid_argument("Checkpoint has an invalid modification factor");
    if (!parameters_.inverse_time_enabled && s.stage == RefinementStage::inverse_time)
        throw std::invalid_argument("Checkpoint uses disabled inverse-time refinement");
    if(!std::isfinite(s.energy)||!std::isfinite(s.order_parameter)||
       std::any_of(s.log_g.begin(),s.log_g.end(),[](double value){return !std::isfinite(value);})||
       std::all_of(s.rng_state.begin(),s.rng_state.end(),[](auto word){return word==0;}))
        throw std::invalid_argument("Checkpoint contains non-finite state or an invalid RNG state");
    for(const auto spin:s.spins)
        if(spin!=-1&&spin!=1) throw std::invalid_argument("Checkpoint contains an invalid spin");
    for(const auto value:s.active)
        if(value>1) throw std::invalid_argument("Checkpoint contains an invalid active mask");

    const auto exact=total_energy(*couplings_,s.spins);
    if(std::abs(exact-s.energy)>1e-9*std::max(1.0,std::abs(exact)))
        throw std::runtime_error("Checkpoint energy does not match spin configuration");
    const auto exact_fields=local_fields(*couplings_,s.spins);
    for(std::size_t i=0;i<s.fields.size();++i) {
        if(!std::isfinite(s.fields[i])||
           std::abs(exact_fields[i]-s.fields[i])>1e-9*std::max(1.0,std::abs(exact_fields[i])))
            throw std::runtime_error("Checkpoint local fields do not match spin configuration");
    }
    if(order_parameter_) {
        const auto exact_q=order_parameter_->evaluate(s.spins);
        if(std::abs(exact_q-s.order_parameter)>1e-10*std::max(1.0,std::abs(exact_q)))
            throw std::runtime_error("Checkpoint order parameter does not match spin configuration");
    }
    const auto state_bin=grid_.index(s.energy);
    const auto state_cell=dos_grid_.index(s.energy,s.order_parameter);
    if(!state_bin||!state_cell||!window_.contains(*state_bin))
        throw std::runtime_error("Checkpoint state is outside the walker DOS window");

    std::size_t active_count=0,refinement_count=0;
    std::vector<std::uint8_t> refinement;
    if(parameters_.nalivaiko_mod) refinement.assign(dos_grid_.cells(),0);
    const auto begin=dos_grid_.flatten(window_.begin),end=dos_grid_.flatten(window_.end);
    for(std::size_t i=begin;i<end;++i) {
        active_count+=s.active[i];
        if(parameters_.nalivaiko_mod) {
            refinement[i]=static_cast<std::uint8_t>(s.histogram[i]!=0);
            refinement_count+=refinement[i];
        }
    }
    for(std::size_t i=0;i<begin;++i)
        if(s.active[i]!=0) throw std::invalid_argument("Checkpoint has active cells outside its window");
    for(std::size_t i=end;i<s.active.size();++i)
        if(s.active[i]!=0) throw std::invalid_argument("Checkpoint has active cells outside its window");

    // Commit only after every compatibility and consistency check succeeds.
    spins_=s.spins; fields_=s.fields; energy_=s.energy;
    order_parameter_value_=s.order_parameter; log_g_=s.log_g;
    histogram_=s.histogram; active_=s.active; factor_=s.factor;
    attempted_=s.attempted; accepted_=s.accepted;
    active_bin_count_=active_count;
    refinement_active_=std::move(refinement);
    refinement_active_bin_count_=refinement_count;
    forced_accepted_ = s.forced_accepted; last_accepted_attempt_ = s.last_accepted_attempt;
    last_new_cell_attempt_=s.last_new_cell_attempt; stage_ = s.stage;
    rng_.set_state(s.rng_state);
    if(parameters_.collect_window_statistics) {
        std::fill(squared_energy_displacement_.begin(),squared_energy_displacement_.end(),0.0);
        std::fill(displacement_samples_.begin(),displacement_samples_.end(),0);
        round_trip_state_=0;
        round_trips_=0;
        update_round_trip_state();
        std::fill(representative_distances_.begin(),representative_distances_.end(),
                  std::numeric_limits<double>::infinity());
        for(auto& representative:representatives_) representative.spins.clear();
        update_representatives();
    }
}

void WangLandauWalker::replace_configuration(std::span<const std::int8_t> spins,
                                              std::span<const double> fields,
                                              double energy,double order_parameter) {
    if (spins.size() != spins_.size() || fields.size() != fields_.size())
        throw std::invalid_argument("Replica configuration size mismatch");
    const auto bin = grid_.index(energy);
    if (!bin || !window_.contains(*bin))
        throw std::invalid_argument("Replica configuration is outside the walker window");
    if(order_parameter_&&!dos_grid_.index(energy,order_parameter))
        throw std::invalid_argument("Replica configuration is outside the order-parameter grid");
    std::copy(spins.begin(), spins.end(), spins_.begin());
    std::copy(fields.begin(), fields.end(), fields_.begin());
    energy_ = energy;
    order_parameter_value_=order_parameter_?order_parameter:0.0;
    update_round_trip_state();
    update_representatives();
}

void WangLandauWalker::swap_configuration(WangLandauWalker& other) {
    const auto this_bin = grid_.index(other.energy_);
    const auto other_bin = other.grid_.index(energy_);
    if (!this_bin || !other_bin || !window_.contains(*this_bin) || !other.window_.contains(*other_bin))
        throw std::invalid_argument("Replica exchange would violate an energy window");
    spins_.swap(other.spins_);
    fields_.swap(other.fields_);
    std::swap(energy_, other.energy_);
    std::swap(order_parameter_value_,other.order_parameter_value_);
    update_round_trip_state();
    other.update_round_trip_state();
    update_representatives();
    other.update_representatives();
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
    const auto x_cell=dos_grid_.index(energy_,order_parameter_value_);
    const auto y_cell=dos_grid_.index(other.energy_,other.order_parameter_value_);
    const auto x_other=other.dos_grid_.index(energy_,order_parameter_value_);
    const auto y_other=other.dos_grid_.index(other.energy_,other.order_parameter_value_);
    if(!x_cell||!y_cell||!x_other||!y_other)
        return -std::numeric_limits<double>::infinity();
    return log_g_[*x_cell] - log_g_[*y_cell] +
           other.log_g_[*y_other] - other.log_g_[*x_other];
}

} // namespace wl
