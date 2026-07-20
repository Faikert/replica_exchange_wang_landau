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

} // namespace wl
