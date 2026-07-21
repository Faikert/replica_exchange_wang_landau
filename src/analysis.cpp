#include "wl/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace wl {
namespace {

double log_sum_exp(std::span<const double> values) {
    auto maximum=-std::numeric_limits<double>::infinity();
    for(const auto value:values) if(std::isfinite(value)) maximum=std::max(maximum,value);
    if (!std::isfinite(maximum)) return maximum;
    double sum = 0.0;
    for (const auto value : values)
        if(std::isfinite(value)) sum += std::exp(value - maximum);
    return maximum + std::log(sum);
}

double local_slope(const std::vector<double>& values,const std::vector<std::uint8_t>& valid,
                   std::size_t center,
                   std::size_t begin, std::size_t end, double width) {
    const auto lo = center > 2 ? std::max(begin, center - 2) : begin;
    const auto hi = std::min(end, center + 3);
    std::size_t count=0;
    double mean_x = 0.0, mean_y = 0.0;
    for (auto i = lo; i < hi; ++i) if(valid[i]!=0&&std::isfinite(values[i])) {
        mean_x += static_cast<double>(i); mean_y += values[i]; ++count;
    }
    if(count<2) return std::numeric_limits<double>::quiet_NaN();
    mean_x /= static_cast<double>(count); mean_y /= static_cast<double>(count);
    double numerator = 0.0, denominator = 0.0;
    for (auto i = lo; i < hi; ++i) if(valid[i]!=0&&std::isfinite(values[i])) {
        const auto dx = static_cast<double>(i) - mean_x;
        numerator += dx * (values[i] - mean_y); denominator += dx*dx;
    }
    return denominator>0.0?numerator/denominator/width:
        std::numeric_limits<double>::quiet_NaN();
}

} // namespace

DensityOfStates stitch_dos(EnergyGrid grid, std::span<const DosFragment> input,
                           bool complete_range, std::size_t spin_count) {
    grid.validate();
    if (input.empty()) throw std::invalid_argument("No DOS fragments supplied");
    std::vector<DosFragment> fragments(input.begin(), input.end());
    std::sort(fragments.begin(), fragments.end(), [](const auto& a, const auto& b) {
        return a.window.begin < b.window.begin;
    });
    const auto bins = grid.bins();
    DensityOfStates result{grid, std::vector<double>(bins, std::numeric_limits<double>::quiet_NaN()),
                           std::vector<std::uint64_t>(bins, 0),
                           std::vector<double>(bins,std::numeric_limits<double>::quiet_NaN()),
                           std::vector<std::uint8_t>(bins,0),{},false};
    auto first = fragments.front();
    if (first.log_g.size()!=bins||first.histogram.size()!=bins||
        first.standard_error.size()!=bins||first.valid.size()!=bins)
        throw std::invalid_argument("DOS fragment size mismatch");
    for (std::size_t i = first.window.begin; i < first.window.end; ++i) {
        result.histogram[i]=first.histogram[i];
        result.valid[i]=first.valid[i];
        if(first.valid[i]!=0) {
            result.log_g[i]=first.log_g[i];
            result.standard_error[i]=first.standard_error[i];
        }
    }
    auto covered_end = first.window.end;
    for (std::size_t f = 1; f < fragments.size(); ++f) {
        auto& right = fragments[f];
        if (right.log_g.size()!=bins||right.histogram.size()!=bins||
            right.standard_error.size()!=bins||right.valid.size()!=bins||
            right.window.begin>=covered_end)
            throw std::invalid_argument("DOS fragments must overlap and match the global grid");
        const auto overlap_begin = right.window.begin;
        const auto overlap_end = std::min(covered_end, right.window.end);
        const auto midpoint=overlap_begin+(overlap_end-overlap_begin)/2;
        auto join=overlap_end;
        auto midpoint_distance=std::numeric_limits<std::size_t>::max();
        for(auto i=overlap_begin;i<overlap_end;++i)
            if(result.valid[i]!=0&&right.valid[i]!=0&&std::isfinite(result.log_g[i])&&
               std::isfinite(right.log_g[i])) {
                const auto distance=i>midpoint?i-midpoint:midpoint-i;
                if(distance<midpoint_distance) { midpoint_distance=distance; join=i; }
            }
        if(join==overlap_end)
            throw std::runtime_error("Adjacent DOS fragments have no common valid overlap bin");
        auto best = std::numeric_limits<double>::infinity();
        for (auto i = overlap_begin + 2; i + 2 < overlap_end; ++i) {
            if(result.valid[i]==0||right.valid[i]==0) continue;
            const auto left_beta=local_slope(result.log_g,result.valid,i,overlap_begin,
                                             overlap_end,grid.width);
            const auto right_beta=local_slope(right.log_g,right.valid,i,overlap_begin,
                                              overlap_end,grid.width);
            const auto difference = std::abs(left_beta-right_beta);
            if (std::isfinite(difference) && difference < best) { best = difference; join = i; }
        }
        const auto shift = result.log_g[join] - right.log_g[join];
        result.join_bins.push_back(join);
        for (std::size_t i = join; i < right.window.end; ++i) {
            result.histogram[i]=right.histogram[i];
            result.valid[i]=right.valid[i];
            if(right.valid[i]!=0) {
                result.log_g[i]=right.log_g[i]+shift;
                result.standard_error[i]=right.standard_error[i];
            } else {
                result.log_g[i]=std::numeric_limits<double>::quiet_NaN();
                result.standard_error[i]=std::numeric_limits<double>::quiet_NaN();
            }
        }
        covered_end = std::max(covered_end, right.window.end);
    }
    if (complete_range) {
        if(std::any_of(result.valid.begin(),result.valid.end(),[](const auto value){return value==0;}))
            throw std::runtime_error("Complete-range normalization requires every DOS bin to be valid");
        const auto normalization = static_cast<double>(spin_count) * std::log(2.0) -
                                   log_sum_exp(result.log_g);
        for(std::size_t i=0;i<bins;++i) if(result.valid[i]!=0) result.log_g[i]+=normalization;
        result.fully_normalized = true;
    } else {
        const auto maximum=log_sum_exp(result.log_g);
        if(!std::isfinite(maximum)) throw std::runtime_error("Stitched DOS has no valid bins");
        auto peak=-std::numeric_limits<double>::infinity();
        for(std::size_t i=0;i<bins;++i) if(result.valid[i]!=0) peak=std::max(peak,result.log_g[i]);
        for(std::size_t i=0;i<bins;++i) if(result.valid[i]!=0) result.log_g[i]-=peak;
    }
    return result;
}

std::vector<ThermodynamicPoint> thermodynamics(const DensityOfStates& dos,
                                               std::span<const double> temperatures,
                                               double k_b) {
    if (!(k_b > 0.0)) throw std::invalid_argument("Boltzmann constant must be positive");
    if(dos.valid.size()!=dos.log_g.size()) throw std::invalid_argument("DOS valid-mask size mismatch");
    std::vector<ThermodynamicPoint> result;
    result.reserve(temperatures.size());
    std::vector<double> weights(dos.log_g.size());
    for (const auto temperature : temperatures) {
        if (!(temperature > 0.0)) throw std::invalid_argument("Temperature must be positive");
        const auto beta = 1.0 / (k_b*temperature);
        for (std::size_t i = 0; i < weights.size(); ++i)
            weights[i]=dos.valid[i]!=0?dos.log_g[i]-beta*dos.grid.center(i):
                -std::numeric_limits<double>::infinity();
        const auto log_z = log_sum_exp(weights);
        if(!std::isfinite(log_z)) throw std::runtime_error("DOS has no valid bins for thermodynamics");
        double e1 = 0.0, e2 = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            if(dos.valid[i]==0) continue;
            const auto probability = std::exp(weights[i]-log_z);
            const auto energy = dos.grid.center(i);
            e1 += probability*energy; e2 += probability*energy*energy;
        }
        const auto heat = (e2-e1*e1)/(k_b*temperature*temperature);
        const auto free = -k_b*temperature*log_z;
        result.push_back({temperature, log_z, e1, heat, free, (e1-free)/temperature});
    }
    return result;
}

DosFragment exact_enumeration(const Couplings& couplings, EnergyGrid grid,
                              std::size_t max_spins) {
    const auto n = couplings.size();
    if (n > max_spins || n >= 63) throw std::invalid_argument("System too large for exact enumeration");
    std::vector<std::uint64_t> counts(grid.bins(), 0);
    std::vector<std::int8_t> spins(n, -1);
    const auto states = std::uint64_t{1} << n;
    for (std::uint64_t state = 0; state < states; ++state) {
        for (std::size_t i = 0; i < n; ++i) spins[i] = (state & (std::uint64_t{1} << i)) ? 1 : -1;
        const auto bin = grid.index(total_energy(couplings, spins));
        if (bin) ++counts[*bin];
    }
    std::vector<double> log_g(grid.bins(), -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < counts.size(); ++i)
        if (counts[i]) log_g[i] = std::log(static_cast<double>(counts[i]));
    // Exact enumeration confirms both occupied bins and exact zero-DOS bins.
    std::vector<std::uint8_t> valid(grid.bins(),1);
    return {{0, grid.bins()}, std::move(log_g), std::move(counts),
            std::vector<double>(grid.bins(), 0.0),std::move(valid)};
}

std::vector<EnergyWindow> adapt_energy_windows(
    EnergyGrid grid, std::span<const DosFragment> fragments,
    std::span<const WindowSamplingStatistics> sampling, std::size_t window_count,
    double overlap, const AdaptiveWindowParameters& p) {
    grid.validate();
    const auto bins=grid.bins();
    if(window_count==0 || window_count>bins || fragments.size()!=window_count ||
       sampling.size()!=window_count || overlap<0.0 || overlap>=1.0 ||
       !std::isfinite(p.diffusivity_floor_fraction) ||
       !(p.diffusivity_floor_fraction>0.0) ||
       !std::isfinite(p.curvature_weight) || p.curvature_weight<0.0 ||
       !std::isfinite(p.round_trip_target) || p.round_trip_target<0.0 ||
       !std::isfinite(p.maximum_round_trip_penalty) ||
       p.maximum_round_trip_penalty<1.0 || !std::isfinite(p.smoothing_width) ||
       p.smoothing_width<0.0 || !std::isfinite(p.minimum_width) || p.minimum_width<0.0)
        throw std::invalid_argument("Invalid adaptive-window parameters");
    if(window_count==1) return {{0,bins}};

    std::vector<double> displacement_sum(bins,0.0),curvature_sum(bins,0.0);
    std::vector<std::uint64_t> samples(bins,0),curvature_samples(bins,0);
    std::vector<double> round_trip_penalty(bins,1.0);
    const auto automatic_smoothing=(grid.maximum-grid.minimum)/
        (10.0*static_cast<double>(window_count));
    const auto smoothing=std::max(grid.width,p.smoothing_width>0.0?
        p.smoothing_width:automatic_smoothing);
    const auto radius=std::max<std::size_t>(1,static_cast<std::size_t>(
        std::llround(smoothing/grid.width)));

    for(std::size_t w=0;w<window_count;++w) {
        const auto& stats=sampling[w];
        const auto& fragment=fragments[w];
        if(stats.window.begin!=fragment.window.begin || stats.window.end!=fragment.window.end ||
           stats.squared_energy_displacement.size()!=bins ||
           stats.displacement_samples.size()!=bins || fragment.log_g.size()!=bins ||
           fragment.valid.size()!=bins)
            throw std::invalid_argument("Adaptive-window pilot fragments are incompatible");
        for(std::size_t i=stats.window.begin;i<stats.window.end;++i) {
            displacement_sum[i]+=stats.squared_energy_displacement[i];
            samples[i]+=stats.displacement_samples[i];
        }
        const auto expected=p.round_trip_target*static_cast<double>(stats.walkers);
        const auto observed=static_cast<double>(stats.round_trips);
        const auto penalty=expected<=0.0?1.0:std::min(p.maximum_round_trip_penalty,
            std::sqrt(expected/std::max(1.0,observed)));
        for(std::size_t i=stats.window.begin;i<stats.window.end;++i)
            round_trip_penalty[i]=std::max(round_trip_penalty[i],penalty);

        for(std::size_t i=fragment.window.begin+radius;
            i+radius<fragment.window.end;++i) {
            if(fragment.valid[i-radius]==0 || fragment.valid[i]==0 ||
               fragment.valid[i+radius]==0) continue;
            const auto scale=static_cast<double>(radius)*grid.width;
            const auto curvature=std::abs(fragment.log_g[i+radius]-2.0*fragment.log_g[i]+
                                          fragment.log_g[i-radius])/(scale*scale);
            if(std::isfinite(curvature)) {
                curvature_sum[i]+=curvature;
                ++curvature_samples[i];
            }
        }
    }

    std::vector<long double> prefix_displacement(bins+1,0.0L);
    std::vector<std::uint64_t> prefix_samples(bins+1,0);
    for(std::size_t i=0;i<bins;++i) {
        prefix_displacement[i+1]=prefix_displacement[i]+displacement_sum[i];
        prefix_samples[i+1]=prefix_samples[i]+samples[i];
    }
    std::vector<double> diffusivity(bins,0.0),positive_diffusivity;
    positive_diffusivity.reserve(bins);
    for(std::size_t i=0;i<bins;++i) {
        const auto lo=i>radius?i-radius:0;
        const auto hi=std::min(bins,i+radius+1);
        const auto count=prefix_samples[hi]-prefix_samples[lo];
        if(count!=0) diffusivity[i]=static_cast<double>(
            (prefix_displacement[hi]-prefix_displacement[lo])/static_cast<long double>(count));
        if(diffusivity[i]>0.0&&std::isfinite(diffusivity[i]))
            positive_diffusivity.push_back(diffusivity[i]);
    }
    double typical_diffusivity=1.0;
    if(!positive_diffusivity.empty()) {
        const auto middle=positive_diffusivity.begin()+
            static_cast<std::ptrdiff_t>(positive_diffusivity.size()/2);
        std::nth_element(positive_diffusivity.begin(),middle,positive_diffusivity.end());
        typical_diffusivity=*middle;
    }
    const auto diffusivity_floor=std::max(std::numeric_limits<double>::min(),
        p.diffusivity_floor_fraction*typical_diffusivity);

    std::vector<double> curvature;
    curvature.reserve(bins);
    for(std::size_t i=0;i<bins;++i) if(curvature_samples[i]!=0) {
        curvature_sum[i]/=static_cast<double>(curvature_samples[i]);
        if(curvature_sum[i]>0.0&&std::isfinite(curvature_sum[i]))
            curvature.push_back(curvature_sum[i]);
    }
    double curvature_scale=1.0;
    if(!curvature.empty()) {
        const auto position=std::min(curvature.size()-1,
            static_cast<std::size_t>(0.9*static_cast<double>(curvature.size())));
        const auto percentile=curvature.begin()+static_cast<std::ptrdiff_t>(position);
        std::nth_element(curvature.begin(),percentile,curvature.end());
        curvature_scale=std::max(*percentile,std::numeric_limits<double>::min());
    }

    std::vector<long double> cumulative_weight(bins+1,0.0L);
    for(std::size_t i=0;i<bins;++i) {
        const auto normalized_curvature=std::min(1.0,curvature_sum[i]/curvature_scale);
        const auto weight=round_trip_penalty[i]*
            (1.0+p.curvature_weight*normalized_curvature)/
            std::sqrt(std::max(diffusivity[i],diffusivity_floor));
        cumulative_weight[i+1]=cumulative_weight[i]+
            static_cast<long double>(weight*grid.width);
    }
    const auto total_weight=cumulative_weight.back();
    if(!(total_weight>0.0L)) throw std::runtime_error("Adaptive-window weight is empty");

    const auto energy_span=grid.maximum-grid.minimum;
    const auto minimum_width=p.minimum_width>0.0?p.minimum_width:
        energy_span/(4.0*static_cast<double>(window_count));
    if(!(minimum_width>0.0) || minimum_width*static_cast<double>(window_count)>energy_span)
        throw std::invalid_argument("Adaptive minimum window width is too large");
    const auto minimum_cells=std::max<std::size_t>(1,static_cast<std::size_t>(
        std::ceil(minimum_width/grid.width)));

    std::vector<std::size_t> core(window_count+1,0);
    core.back()=bins;
    for(std::size_t k=1;k<window_count;++k) {
        const auto target=total_weight*static_cast<long double>(k)/
                          static_cast<long double>(window_count);
        auto candidate=static_cast<std::size_t>(std::lower_bound(
            cumulative_weight.begin(),cumulative_weight.end(),target)-cumulative_weight.begin());
        const auto lowest=core[k-1]+minimum_cells;
        const auto highest=bins-(window_count-k)*minimum_cells;
        candidate=std::clamp(candidate,lowest,highest);
        core[k]=candidate;
    }

    const auto extension=overlap==0.0?0.0:overlap/(2.0*(1.0-overlap));
    std::vector<EnergyWindow> result;
    result.reserve(window_count);
    for(std::size_t k=0;k<window_count;++k) {
        const auto core_width=core[k+1]-core[k];
        const auto extra=static_cast<std::size_t>(std::ceil(extension*
            static_cast<double>(core_width)));
        const auto begin=k==0?std::size_t{0}:(core[k]>extra?core[k]-extra:0);
        const auto end=k+1==window_count?bins:
            std::min(bins-(window_count-1-k),core[k+1]+extra);
        result.push_back({begin,end});
    }
    for(std::size_t k=1;k<result.size();++k) {
        result[k].begin=std::max(result[k].begin,result[k-1].begin+1);
        result[k].end=std::max(result[k].end,result[k-1].end+1);
        if(result[k].begin>=result[k-1].end || result[k].begin>=result[k].end)
            throw std::runtime_error("Adaptive energy windows do not overlap");
    }
    return result;
}

std::vector<std::vector<std::int8_t>> select_adaptive_initial_configurations(
    const EnergyGrid& grid, std::span<const EnergyWindow> windows,
    std::span<const EnergyRepresentative> representatives, int mpi_size,
    std::size_t walkers_per_rank, std::size_t& missing,
    std::size_t& external_warm_starts) {
    if(windows.empty() || mpi_size<=0 || walkers_per_rank==0)
        throw std::invalid_argument("Invalid adaptive initial-configuration layout");
    const bool distributed=mpi_size>1;
    if(distributed && static_cast<std::size_t>(mpi_size)%windows.size()!=0)
        throw std::invalid_argument("MPI size must be divisible by adaptive window count");
    const auto shards=distributed?static_cast<std::size_t>(mpi_size)/windows.size():1;
    const auto owner_count=distributed?static_cast<std::size_t>(mpi_size):windows.size();
    std::vector<std::vector<std::int8_t>> result(owner_count*walkers_per_rank);
    missing=0;
    external_warm_starts=0;

    const EnergyRepresentative* minimum_representative=nullptr;
    for(const auto& representative:representatives)
        if(!representative.spins.empty() &&
           (minimum_representative==nullptr ||
            representative.energy<minimum_representative->energy))
            minimum_representative=&representative;

    for(std::size_t owner=0;owner<owner_count;++owner) {
        const auto window_id=distributed?owner/shards:owner;
        const auto& window=windows[window_id];
        const auto center=0.5*(grid.minimum+static_cast<double>(window.begin)*grid.width+
                               grid.minimum+static_cast<double>(window.end)*grid.width);
        const auto lower=grid.minimum+static_cast<double>(window.begin)*grid.width;
        const auto upper=grid.minimum+static_cast<double>(window.end)*grid.width;

        // Every walker in the lower edge window starts from the lowest-energy pilot
        // configuration. Distinct RNG streams make the trajectories independent after
        // startup, while all local DOS estimates retain coverage of the rare low edge.
        if(window_id==0 && minimum_representative!=nullptr) {
            const auto minimum_bin=grid.index(minimum_representative->energy);
            const bool external=!minimum_bin || !window.contains(*minimum_bin);
            for(std::size_t local=0;local<walkers_per_rank;++local) {
                const auto id=owner*walkers_per_rank+local;
                result[id]=minimum_representative->spins;
                if(external) ++external_warm_starts;
            }
            continue;
        }

        std::vector<const EnergyRepresentative*> candidates;
        for(const auto& representative:representatives) {
            const auto energy_bin=grid.index(representative.energy);
            if(!representative.spins.empty()&&energy_bin&&window.contains(*energy_bin))
                candidates.push_back(&representative);
        }
        const bool external=candidates.empty();
        if(external)
            for(const auto& representative:representatives)
                if(!representative.spins.empty()) candidates.push_back(&representative);
        const auto interval_distance=[lower,upper](double energy) {
            if(energy<lower) return lower-energy;
            if(energy>=upper) return energy-upper;
            return 0.0;
        };
        std::sort(candidates.begin(),candidates.end(),[&](const auto* a,const auto* b) {
            const auto ia=interval_distance(a->energy),ib=interval_distance(b->energy);
            if(ia!=ib) return ia<ib;
            const auto da=std::abs(a->energy-center),db=std::abs(b->energy-center);
            return da==db?a->energy<b->energy:da<db;
        });
        const auto shard=distributed?owner%shards:0;
        for(std::size_t local=0;local<walkers_per_rank;++local) {
            const auto id=owner*walkers_per_rank+local;
            if(candidates.empty()) {
                ++missing;
                continue;
            }
            const auto choice=(shard*walkers_per_rank+local)%candidates.size();
            result[id]=candidates[choice]->spins;
            if(external) ++external_warm_starts;
        }
    }
    return result;
}

} // namespace wl
