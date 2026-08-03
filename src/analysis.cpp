#include "wl/analysis.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <queue>
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

} // namespace

DisconnectedSupportError::DisconnectedSupportError(std::size_t components)
    : std::runtime_error("DOS support has "+std::to_string(components)+
                         " disconnected components"),components_(components) {}

DensityOfStates stitch_dos(EnergyGrid grid, std::span<const DosFragment> input,
                           bool complete_range, std::size_t spin_count) {
    const auto stitched=stitch_joint_dos(DosGrid{grid,std::nullopt},input,complete_range,
                                         true,spin_count,1.0);
    return {grid,stitched.log_g,stitched.histogram,stitched.standard_error,stitched.valid,
            stitched.contributors,stitched.support_component,{},stitched.fully_normalized};
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
    std::uint64_t in_grid=0;
    for (std::uint64_t state = 0; state < states; ++state) {
        for (std::size_t i = 0; i < n; ++i) spins[i] = (state & (std::uint64_t{1} << i)) ? 1 : -1;
        const auto bin = grid.index(total_energy(couplings, spins));
        if (bin) { ++counts[*bin]; ++in_grid; }
    }
    if(in_grid!=states)
        throw std::runtime_error("Exact enumeration lost "+std::to_string(states-in_grid)+
                                 " states outside the energy grid");
    std::vector<double> log_g(grid.bins(), -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < counts.size(); ++i)
        if (counts[i]) log_g[i] = std::log(static_cast<double>(counts[i]));
    // Exact enumeration confirms both occupied bins and exact zero-DOS bins.
    std::vector<std::uint8_t> valid(grid.bins(),1);
    return {{0, grid.bins()}, std::move(log_g), std::move(counts),
            std::vector<double>(grid.bins(), 0.0),std::move(valid),
            DosGrid{grid,std::nullopt},std::vector<std::uint32_t>(grid.bins(),1),
            std::vector<std::int32_t>(grid.bins(),0)};
}

struct StitchNode {
    std::size_t fragment{};
    std::int32_t component{};
};

struct StitchEdge {
    std::size_t first{},second{};
    long double weight{},difference{};
};

std::pair<std::vector<double>,std::size_t> solve_stitch_offsets(
    std::size_t node_count,std::span<const StitchEdge> edges) {
    std::vector<std::vector<std::size_t>> adjacency(node_count);
    for(const auto& edge:edges) {
        adjacency[edge.first].push_back(edge.second);
        adjacency[edge.second].push_back(edge.first);
    }
    std::vector<std::int32_t> component(node_count,-1);
    std::vector<double> shifts(node_count,0.0);
    std::size_t components=0;
    for(std::size_t seed=0;seed<node_count;++seed) if(component[seed]<0) {
        const auto label=static_cast<std::int32_t>(components++);
        std::vector<std::size_t> nodes;
        std::queue<std::size_t> pending; pending.push(seed); component[seed]=label;
        while(!pending.empty()) {
            const auto node=pending.front(); pending.pop(); nodes.push_back(node);
            for(const auto other:adjacency[node]) if(component[other]<0) {
                component[other]=label; pending.push(other);
            }
        }
        if(nodes.size()==1) continue;
        std::vector<std::size_t> equation(node_count,std::numeric_limits<std::size_t>::max());
        for(std::size_t i=1;i<nodes.size();++i) equation[nodes[i]]=i-1;
        const auto unknowns=nodes.size()-1;
        std::vector<long double> matrix(unknowns*(unknowns+1),0.0L);
        const auto add=[&](std::size_t row,std::size_t column,long double value) {
            if(row!=std::numeric_limits<std::size_t>::max()&&
               column!=std::numeric_limits<std::size_t>::max())
                matrix[row*(unknowns+1)+column]+=value;
        };
        for(const auto& edge:edges) if(component[edge.first]==label&&component[edge.second]==label) {
            const auto a=equation[edge.first],b=equation[edge.second];
            add(a,a,edge.weight); add(b,b,edge.weight);
            add(a,b,-edge.weight); add(b,a,-edge.weight);
            if(a!=std::numeric_limits<std::size_t>::max())
                matrix[a*(unknowns+1)+unknowns]-=edge.weight*edge.difference;
            if(b!=std::numeric_limits<std::size_t>::max())
                matrix[b*(unknowns+1)+unknowns]+=edge.weight*edge.difference;
        }
        for(std::size_t column=0;column<unknowns;++column) {
            auto pivot=column;
            for(std::size_t row=column+1;row<unknowns;++row)
                if(std::abs(matrix[row*(unknowns+1)+column])>
                   std::abs(matrix[pivot*(unknowns+1)+column])) pivot=row;
            if(!(std::abs(matrix[pivot*(unknowns+1)+column])>0.0L))
                throw std::runtime_error("Singular joint-DOS stitching system");
            if(pivot!=column) for(std::size_t j=column;j<=unknowns;++j)
                std::swap(matrix[pivot*(unknowns+1)+j],matrix[column*(unknowns+1)+j]);
            const auto divisor=matrix[column*(unknowns+1)+column];
            for(std::size_t j=column;j<=unknowns;++j)
                matrix[column*(unknowns+1)+j]/=divisor;
            for(std::size_t row=0;row<unknowns;++row) if(row!=column) {
                const auto multiplier=matrix[row*(unknowns+1)+column];
                for(std::size_t j=column;j<=unknowns;++j)
                    matrix[row*(unknowns+1)+j]-=multiplier*matrix[column*(unknowns+1)+j];
            }
        }
        for(std::size_t i=1;i<nodes.size();++i)
            shifts[nodes[i]]=static_cast<double>(matrix[(i-1)*(unknowns+1)+unknowns]);
    }
    return {std::move(shifts),components};
}

DosFragment exact_enumeration(const Couplings& couplings,DosGrid grid,
                              const WeightedOrderParameter& order_parameter,
                              std::size_t max_spins) {
    grid.validate();
    if(!grid.joint()) throw std::invalid_argument("Joint exact enumeration requires a Q grid");
    const auto n=couplings.size();
    if(n>max_spins||n>=63) throw std::invalid_argument("System too large for exact enumeration");
    if(order_parameter.weights.size()!=n)
        throw std::invalid_argument("Order-parameter weight count mismatch");
    std::vector<std::uint64_t> counts(grid.cells(),0);
    std::vector<std::int8_t> spins(n,-1);
    const auto states=std::uint64_t{1}<<n;
    std::uint64_t in_grid=0;
    for(std::uint64_t state=0;state<states;++state) {
        for(std::size_t i=0;i<n;++i)
            spins[i]=(state&(std::uint64_t{1}<<i))?1:-1;
        const auto cell=grid.index(total_energy(couplings,spins),order_parameter.evaluate(spins));
        if(cell) { ++counts[*cell]; ++in_grid; }
    }
    if(in_grid!=states)
        throw std::runtime_error("Exact enumeration lost "+std::to_string(states-in_grid)+
                                 " states outside the joint DOS grid");
    std::vector<double> log_g(grid.cells(),-std::numeric_limits<double>::infinity());
    for(std::size_t i=0;i<counts.size();++i) if(counts[i]) log_g[i]=std::log(static_cast<double>(counts[i]));
    DosFragment result{{0,grid.energy_bins()},std::move(log_g),std::move(counts),
        std::vector<double>(grid.cells(),0.0),std::vector<std::uint8_t>(grid.cells(),1),
        grid,{},{}};
    result.contributors.assign(grid.cells(),1);
    result.support_component.assign(grid.cells(),0);
    return result;
}

DosFragment marginalize_fragment(const DosFragment& joint) {
    if(!joint.grid.joint()) return joint;
    const auto& layout=joint.grid;
    const auto e_bins=layout.energy_bins(),q_bins=layout.q_bins();
    if(joint.window.begin>=joint.window.end||joint.window.end>e_bins||
       joint.log_g.size()!=layout.cells()||joint.histogram.size()!=joint.log_g.size()||
       joint.standard_error.size()!=joint.log_g.size()||joint.valid.size()!=joint.log_g.size())
        throw std::invalid_argument("Invalid joint DOS fragment for marginalization");
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    DosFragment result{joint.window,std::vector<double>(e_bins,nan),
        std::vector<std::uint64_t>(e_bins,0),std::vector<double>(e_bins,nan),
        std::vector<std::uint8_t>(e_bins,0),DosGrid{layout.energy,std::nullopt},
        std::vector<std::uint32_t>(e_bins,0),std::vector<std::int32_t>(e_bins,-1)};
    std::vector<double> values;
    for(std::size_t e=joint.window.begin;e<joint.window.end;++e) {
        values.clear();
        bool known=false;
        for(std::size_t q=0;q<q_bins;++q) {
            const auto cell=layout.flatten(e,q);
            result.histogram[e]+=joint.histogram[cell];
            if(joint.valid[cell]!=0) {
                known=true;
                const auto contributors=joint.contributors.empty()?std::uint32_t{1}:
                    joint.contributors[cell];
                result.contributors[e]=std::max(result.contributors[e],contributors);
                result.support_component[e]=0;
                if(std::isfinite(joint.log_g[cell])) values.push_back(joint.log_g[cell]);
            }
        }
        if(!known) continue;
        result.valid[e]=1;
        if(values.empty()) {
            result.log_g[e]=-std::numeric_limits<double>::infinity();
            result.standard_error[e]=0.0;
            continue;
        }
        result.log_g[e]=log_sum_exp(values);
        double variance=0.0; bool finite_sem=true;
        for(std::size_t q=0;q<q_bins;++q) {
            const auto cell=layout.flatten(e,q);
            if(joint.valid[cell]==0||!std::isfinite(joint.log_g[cell])) continue;
            if(!std::isfinite(joint.standard_error[cell])) { finite_sem=false; break; }
            const auto weight=std::exp(joint.log_g[cell]-result.log_g[e]);
            variance+=weight*weight*joint.standard_error[cell]*joint.standard_error[cell];
        }
        if(finite_sem) result.standard_error[e]=std::sqrt(variance);
    }
    return result;
}

std::size_t support_component_count(const DosFragment& fragment) {
    if(fragment.valid.size()!=fragment.grid.cells()||
       (!fragment.support_component.empty()&&
        fragment.support_component.size()!=fragment.valid.size()))
        throw std::invalid_argument("Invalid support-component layout");
    std::vector<std::int32_t> components;
    for(std::size_t cell=0;cell<fragment.valid.size();++cell) if(fragment.valid[cell]) {
        const auto component=fragment.support_component.empty()?0:fragment.support_component[cell];
        if(component<0) throw std::invalid_argument("Valid cell has no support component");
        if(std::find(components.begin(),components.end(),component)==components.end())
            components.push_back(component);
    }
    return components.size();
}

JointDensityOfStates stitch_joint_dos(DosGrid grid,std::span<const DosFragment> input,
                                      bool complete_range,bool converged,
                                      std::size_t spin_count,double normalization) {
    grid.validate();
    if(input.empty()||!(normalization>0.0)||!std::isfinite(normalization))
        throw std::invalid_argument("No valid DOS fragments supplied");
    std::vector<DosFragment> fragments(input.begin(),input.end());
    std::sort(fragments.begin(),fragments.end(),[](const auto& a,const auto& b){
        return a.window.begin<b.window.begin;
    });
    const auto cells=grid.cells(),q_bins=grid.q_bins();
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const auto same_grid=[&](const DosGrid& other) {
        if(other.joint()!=grid.joint()||other.energy.minimum!=grid.energy.minimum||
           other.energy.maximum!=grid.energy.maximum||other.energy.width!=grid.energy.width)
            return false;
        return !grid.joint()||
            (other.order_parameter->minimum==grid.order_parameter->minimum&&
             other.order_parameter->maximum==grid.order_parameter->maximum&&
             other.order_parameter->width==grid.order_parameter->width);
    };
    for(const auto& fragment:fragments) {
        if(!same_grid(fragment.grid)||fragment.window.begin>=fragment.window.end||
           fragment.window.end>grid.energy_bins()||fragment.log_g.size()!=cells||
           fragment.histogram.size()!=cells||fragment.standard_error.size()!=cells||
           fragment.valid.size()!=cells||
           (!fragment.contributors.empty()&&fragment.contributors.size()!=cells)||
           (!fragment.support_component.empty()&&fragment.support_component.size()!=cells))
            throw std::invalid_argument("DOS fragment layout mismatch");
    }
    for(std::size_t f=1;f<fragments.size();++f)
        if(fragments[f].window.begin>=fragments[f-1].window.end)
            throw std::invalid_argument("DOS fragments must overlap");

    const auto component_at=[](const DosFragment& fragment,std::size_t cell) {
        return fragment.support_component.empty()?std::int32_t{0}:
            fragment.support_component[cell];
    };
    const auto contributors_at=[](const DosFragment& fragment,std::size_t cell) {
        return fragment.contributors.empty()?
            static_cast<std::uint32_t>(fragment.valid[cell]!=0):fragment.contributors[cell];
    };
    std::vector<StitchNode> nodes;
    std::vector<std::vector<std::pair<std::int32_t,std::size_t>>> node_lookup(fragments.size());
    for(std::size_t f=0;f<fragments.size();++f) {
        const auto begin=grid.flatten(fragments[f].window.begin);
        const auto end=grid.flatten(fragments[f].window.end);
        for(std::size_t cell=begin;cell<end;++cell) if(fragments[f].valid[cell]) {
            const auto component=component_at(fragments[f],cell);
            if(component<0) throw std::invalid_argument("Valid DOS cell has no support component");
            const auto found=std::find_if(node_lookup[f].begin(),node_lookup[f].end(),
                [component](const auto& item){return item.first==component;});
            if(found==node_lookup[f].end()) {
                node_lookup[f].push_back({component,nodes.size()});
                nodes.push_back({f,component});
            }
        }
    }
    if(nodes.empty()) throw DisconnectedSupportError(0);
    std::vector<StitchEdge> edges;
    for(std::size_t a=0;a<nodes.size();++a) for(std::size_t b=a+1;b<nodes.size();++b) {
        if(nodes[a].fragment==nodes[b].fragment) continue;
        const auto& left=fragments[nodes[a].fragment];
        const auto& right=fragments[nodes[b].fragment];
        const auto overlap_begin=std::max(left.window.begin,right.window.begin);
        const auto overlap_end=std::min(left.window.end,right.window.end);
        if(overlap_begin>=overlap_end) continue;
        long double total_weight=0.0L,weighted_difference=0.0L;
        for(std::size_t e=overlap_begin;e<overlap_end;++e)
            for(std::size_t q=0;q<q_bins;++q) {
                const auto cell=grid.flatten(e,q);
                if(left.valid[cell]&&right.valid[cell]&&
                   component_at(left,cell)==nodes[a].component&&
                   component_at(right,cell)==nodes[b].component&&
                   std::isfinite(left.log_g[cell])&&std::isfinite(right.log_g[cell])) {
                    const auto weight=1.0L+static_cast<long double>(
                        std::min(left.histogram[cell],right.histogram[cell]));
                    total_weight+=weight;
                    weighted_difference+=weight*(left.log_g[cell]-right.log_g[cell]);
                }
            }
        if(total_weight>0.0L)
            edges.push_back({a,b,total_weight,weighted_difference/total_weight});
    }
    auto [node_shifts,support_components]=solve_stitch_offsets(nodes.size(),edges);
    if(support_components!=1) throw DisconnectedSupportError(support_components);
    const auto node_for=[&](std::size_t fragment,std::size_t cell) {
        const auto component=component_at(fragments[fragment],cell);
        const auto found=std::find_if(node_lookup[fragment].begin(),node_lookup[fragment].end(),
            [component](const auto& item){return item.first==component;});
        if(found==node_lookup[fragment].end())
            throw std::logic_error("Missing DOS support node");
        return found->second;
    };

    JointDensityOfStates result{grid,normalization,std::vector<double>(cells,nan),
        std::vector<std::uint64_t>(cells,0),std::vector<double>(cells,nan),
        std::vector<std::uint8_t>(cells,0),std::vector<std::uint32_t>(cells,0),
        std::vector<std::int32_t>(cells,-1),false};
    const auto copy_cell=[&](std::size_t cell,const DosFragment& source,std::size_t fragment) {
        result.histogram[cell]=source.histogram[cell]; result.valid[cell]=source.valid[cell];
        if(source.valid[cell]) {
            result.log_g[cell]=source.log_g[cell]+node_shifts[node_for(fragment,cell)];
            result.standard_error[cell]=source.standard_error[cell];
            result.contributors[cell]=contributors_at(source,cell);
            result.support_component[cell]=0;
        }
    };
    const auto& first=fragments.front();
    for(auto cell=grid.flatten(first.window.begin);cell<grid.flatten(first.window.end);++cell)
        copy_cell(cell,first,0);
    auto covered_end=first.window.end;
    for(std::size_t f=1;f<fragments.size();++f) {
        const auto& right=fragments[f];
        const auto overlap_begin=right.window.begin;
        const auto overlap_end=std::min(covered_end,right.window.end);
        for(std::size_t e=right.window.begin;e<right.window.end;++e)
            for(std::size_t q=0;q<q_bins;++q) {
                const auto cell=grid.flatten(e,q);
                const bool left_valid=result.valid[cell]!=0;
                const bool right_valid=right.valid[cell]!=0;
                if(!right_valid) continue;
                const auto right_value=right.log_g[cell]+node_shifts[node_for(f,cell)];
                if(left_valid&&e<overlap_end) {
                    const auto alpha=std::clamp((static_cast<double>(e-overlap_begin)+0.5)/
                        static_cast<double>(overlap_end-overlap_begin),0.0,1.0);
                    if(std::isfinite(result.log_g[cell])&&std::isfinite(right_value))
                        result.log_g[cell]=(1.0-alpha)*result.log_g[cell]+alpha*right_value;
                    else if(std::isfinite(right_value)) result.log_g[cell]=right_value;
                    const auto a=result.standard_error[cell],b=right.standard_error[cell];
                    result.standard_error[cell]=std::isfinite(a)&&std::isfinite(b)?
                        std::hypot((1.0-alpha)*a,alpha*b):nan;
                    result.histogram[cell]+=right.histogram[cell];
                    result.contributors[cell]+=contributors_at(right,cell);
                } else copy_cell(cell,right,f);
            }
        covered_end=std::max(covered_end,right.window.end);
    }
    if(complete_range&&converged) {
        if(!grid.joint()&&std::any_of(result.valid.begin(),result.valid.end(),
                                     [](const auto value){return value==0;}))
            throw InsufficientSupportError(
                "Complete-range normalization requires every DOS bin to be valid");
        const auto log_total=log_sum_exp(result.log_g);
        if(!std::isfinite(log_total)) throw std::runtime_error("DOS has no finite support");
        const auto shift=static_cast<double>(spin_count)*std::log(2.0)-log_total;
        for(std::size_t i=0;i<cells;++i) if(result.valid[i]) result.log_g[i]+=shift;
        result.fully_normalized=true;
    } else {
        auto peak=-std::numeric_limits<double>::infinity();
        for(std::size_t i=0;i<cells;++i) if(result.valid[i]&&std::isfinite(result.log_g[i])) peak=std::max(peak,result.log_g[i]);
        if(!std::isfinite(peak)) throw std::runtime_error("Stitched DOS has no valid cells");
        for(std::size_t i=0;i<cells;++i) if(result.valid[i]) result.log_g[i]-=peak;
    }
    return result;
}

DensityOfStates marginalize(const JointDensityOfStates& joint) {
    DosFragment fragment{{0,joint.grid.energy_bins()},joint.log_g,joint.histogram,
        joint.standard_error,joint.valid,joint.grid,joint.contributors,joint.support_component};
    const auto marginal=marginalize_fragment(fragment);
    return {joint.grid.energy,marginal.log_g,marginal.histogram,marginal.standard_error,
        marginal.valid,marginal.contributors,marginal.support_component,{},
        joint.fully_normalized};
}

std::vector<OrderParameterDistributionPoint> order_parameter_distribution(
    const JointDensityOfStates& dos,std::span<const double> temperatures,double k_b) {
    if(!(k_b>0.0)||!dos.grid.joint()||!(dos.normalization>0.0))
        throw std::invalid_argument("Invalid joint DOS distribution request");
    if(dos.log_g.size()!=dos.grid.cells()||dos.valid.size()!=dos.log_g.size()||
       dos.histogram.size()!=dos.log_g.size()||dos.standard_error.size()!=dos.log_g.size())
        throw std::invalid_argument("Invalid joint DOS arrays for distribution");
    std::vector<OrderParameterDistributionPoint> result;
    const auto e_bins=dos.grid.energy_bins(),q_bins=dos.grid.q_bins();
    std::vector<double> log_q(q_bins),terms;
    for(const auto temperature:temperatures) {
        if(!(temperature>0.0)) throw std::invalid_argument("Temperature must be positive");
        const auto beta=1.0/(k_b*temperature);
        for(std::size_t q=0;q<q_bins;++q) {
            terms.clear();
            for(std::size_t e=0;e<e_bins;++e) {
                const auto cell=dos.grid.flatten(e,q);
                if(dos.valid[cell]&&std::isfinite(dos.log_g[cell]))
                    terms.push_back(dos.log_g[cell]-beta*dos.grid.energy.center(e));
            }
            log_q[q]=terms.empty()?-std::numeric_limits<double>::infinity():log_sum_exp(terms);
        }
        const auto log_z=log_sum_exp(log_q);
        auto maximum=-std::numeric_limits<double>::infinity();
        for(auto& value:log_q) { value-=log_z; maximum=std::max(maximum,value); }
        for(std::size_t q=0;q<q_bins;++q) {
            const auto Q=dos.grid.order_parameter->center(q);
            const auto probability=std::isfinite(log_q[q])?std::exp(log_q[q]):0.0;
            const auto free=std::isfinite(log_q[q])?-k_b*temperature*(log_q[q]-maximum):
                std::numeric_limits<double>::infinity();
            result.push_back({temperature,q,Q,Q/dos.normalization,probability,log_q[q],free});
        }
    }
    return result;
}

std::vector<OrderParameterThermodynamicPoint> order_parameter_thermodynamics(
    const JointDensityOfStates& dos,std::span<const double> temperatures,
    std::size_t spin_count,double k_b) {
    const auto distribution=order_parameter_distribution(dos,temperatures,k_b);
    const auto q_bins=dos.grid.q_bins();
    std::vector<OrderParameterThermodynamicPoint> result;
    result.reserve(temperatures.size());
    for(std::size_t t=0;t<temperatures.size();++t) {
        OrderParameterThermodynamicPoint point; point.temperature=temperatures[t];
        for(std::size_t b=0;b<q_bins;++b) {
            const auto& x=distribution[t*q_bins+b]; const auto p=x.probability;
            const auto q2=x.q*x.q,Q2=x.Q*x.Q;
            point.mean_q+=p*x.q; point.mean_abs_q+=p*std::abs(x.q);
            point.mean_q2+=p*q2; point.mean_q4+=p*q2*q2;
            point.mean_Q+=p*x.Q; point.mean_abs_Q+=p*std::abs(x.Q);
            point.mean_Q2+=p*Q2; point.mean_Q4+=p*Q2*Q2;
        }
        point.susceptibility=static_cast<double>(spin_count)*
            (point.mean_q2-point.mean_q*point.mean_q)/(k_b*point.temperature);
        point.binder_cumulant=point.mean_q2>0.0?
            1.0-point.mean_q4/(3.0*point.mean_q2*point.mean_q2):
            std::numeric_limits<double>::quiet_NaN();
        result.push_back(point);
    }
    return result;
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
