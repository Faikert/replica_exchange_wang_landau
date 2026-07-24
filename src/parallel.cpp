#include "wl/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <stdexcept>

#ifdef WL_HAS_MPI
#include <mpi.h>
#endif
#ifdef WL_HAS_OPENMP
#include <omp.h>
#endif

namespace wl {
namespace {

struct EstimatorView {
    std::span<const double> log_g;
    std::span<const std::uint64_t> histogram;
    std::span<const std::uint8_t> active;
};

struct AlignmentEdge {
    std::size_t first{},second{};
    long double weight{},difference{};
};

struct AlignmentSolution {
    std::vector<double> shifts;
    std::vector<std::int32_t> component;
    std::size_t components{};
};

AlignmentSolution solve_alignment(std::span<const EstimatorView> estimators,
                                  std::size_t begin,std::size_t end) {
    const auto count=estimators.size();
    std::vector<AlignmentEdge> edges;
    std::vector<std::vector<std::size_t>> adjacency(count);
    for(std::size_t a=0;a<count;++a) for(std::size_t b=a+1;b<count;++b) {
        long double weight=0.0L,difference=0.0L;
        for(std::size_t cell=begin;cell<end;++cell)
            if(estimators[a].active[cell]&&estimators[b].active[cell]&&
               std::isfinite(estimators[a].log_g[cell])&&
               std::isfinite(estimators[b].log_g[cell])) {
                const auto w=1.0L+static_cast<long double>(std::min(
                    estimators[a].histogram[cell],estimators[b].histogram[cell]));
                weight+=w;
                difference+=w*static_cast<long double>(
                    estimators[a].log_g[cell]-estimators[b].log_g[cell]);
            }
        if(weight>0.0L) {
            edges.push_back({a,b,weight,difference/weight});
            adjacency[a].push_back(b); adjacency[b].push_back(a);
        }
    }

    AlignmentSolution result{std::vector<double>(count,0.0),
        std::vector<std::int32_t>(count,-1),0};
    for(std::size_t seed=0;seed<count;++seed) if(result.component[seed]<0) {
        const auto component=static_cast<std::int32_t>(result.components++);
        std::vector<std::size_t> nodes;
        std::queue<std::size_t> pending; pending.push(seed); result.component[seed]=component;
        while(!pending.empty()) {
            const auto node=pending.front(); pending.pop(); nodes.push_back(node);
            for(const auto other:adjacency[node]) if(result.component[other]<0) {
                result.component[other]=component; pending.push(other);
            }
        }
        if(nodes.size()==1) continue;
        std::vector<std::size_t> equation(count,std::numeric_limits<std::size_t>::max());
        for(std::size_t i=1;i<nodes.size();++i) equation[nodes[i]]=i-1;
        const auto unknowns=nodes.size()-1;
        std::vector<long double> matrix(unknowns*(unknowns+1),0.0L);
        const auto add=[&](std::size_t row,std::size_t column,long double value) {
            if(row!=std::numeric_limits<std::size_t>::max()&&
               column!=std::numeric_limits<std::size_t>::max())
                matrix[row*(unknowns+1)+column]+=value;
        };
        for(const auto& edge:edges) if(result.component[edge.first]==component&&
                                        result.component[edge.second]==component) {
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
                throw std::runtime_error("Singular joint-DOS alignment system");
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
            result.shifts[nodes[i]]=static_cast<double>(matrix[(i-1)*(unknowns+1)+unknowns]);
    }
    return result;
}

DosFragment summarize_joint_union(EnergyWindow window,const DosGrid& layout,
                                  std::span<const EstimatorView> estimators) {
    const auto cells=layout.cells();
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    DosFragment result{window,std::vector<double>(cells,nan),
        std::vector<std::uint64_t>(cells,0),std::vector<double>(cells,nan),
        std::vector<std::uint8_t>(cells,0),layout,std::vector<std::uint32_t>(cells,0),
        std::vector<std::int32_t>(cells,-1)};
    const auto begin=layout.flatten(window.begin),end=layout.flatten(window.end);
    const auto alignment=solve_alignment(estimators,begin,end);
    for(std::size_t cell=begin;cell<end;++cell) {
        long double sum=0.0L;
        for(std::size_t w=0;w<estimators.size();++w) if(estimators[w].active[cell]) {
            sum+=static_cast<long double>(estimators[w].log_g[cell]+alignment.shifts[w]);
            result.histogram[cell]+=estimators[w].histogram[cell];
            ++result.contributors[cell];
            result.support_component[cell]=alignment.component[w];
        }
        if(result.contributors[cell]==0) continue;
        result.valid[cell]=1;
        result.log_g[cell]=static_cast<double>(sum/result.contributors[cell]);
        if(result.contributors[cell]>1) {
            long double squared=0.0L;
            for(std::size_t w=0;w<estimators.size();++w) if(estimators[w].active[cell]) {
                const auto delta=estimators[w].log_g[cell]+alignment.shifts[w]-result.log_g[cell];
                squared+=static_cast<long double>(delta)*delta;
            }
            result.standard_error[cell]=std::sqrt(static_cast<double>(squared/
                (static_cast<long double>(result.contributors[cell])*
                 static_cast<long double>(result.contributors[cell]-1))));
        }
    }
    return result;
}

DosFragment summarize_impl(EnergyWindow window,
                           std::span<const WangLandauWalker* const> walkers,std::size_t bins,
                           bool allow_empty_intersection) {
    (void)bins;
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const auto layout=walkers.front()->dos_grid();
    if(layout.joint()) {
        std::vector<EstimatorView> estimators;
        estimators.reserve(walkers.size());
        for(const auto* walker:walkers)
            estimators.push_back({walker->log_g(),walker->histogram(),walker->active_mask()});
        return summarize_joint_union(window,layout,estimators);
    }
    const auto cells=layout.cells();
    DosFragment f{window,std::vector<double>(cells,nan),std::vector<std::uint64_t>(cells,0),
                  std::vector<double>(cells,nan),std::vector<std::uint8_t>(cells,0),
                  layout,{},{}};
    const auto begin=layout.flatten(window.begin),end=layout.flatten(window.end);
    for(std::size_t i=begin;i<end;++i) {
        bool valid=true;
        for(const auto& walker:walkers) {
            valid=valid&&walker->active_mask()[i]!=0;
            f.histogram[i]+=walker->histogram()[i];
        }
        f.valid[i]=valid?1:0;
    }
    const auto reference=std::find(f.valid.begin()+static_cast<std::ptrdiff_t>(begin),
                                   f.valid.begin()+static_cast<std::ptrdiff_t>(end),1);
    if(reference==f.valid.begin()+static_cast<std::ptrdiff_t>(end) && allow_empty_intersection)
        return f;
    if(reference==f.valid.begin()+static_cast<std::ptrdiff_t>(end))
        throw std::runtime_error("Walkers in an energy window have no common active bin");
    const auto reference_bin=static_cast<std::size_t>(reference-f.valid.begin());
    for(std::size_t i=begin;i<end;++i) {
        if(f.valid[i]==0) continue;
        f.log_g[i]=0.0;
        for(const auto& walker:walkers)
            f.log_g[i]+=walker->log_g()[i]-walker->log_g()[reference_bin];
        f.log_g[i]/=static_cast<double>(walkers.size());
        if(walkers.size()>1) {
            double squared_deviation=0.0;
            for(const auto& walker:walkers) {
                const auto value=walker->log_g()[i]-walker->log_g()[reference_bin];
                const auto delta=value-f.log_g[i];
                squared_deviation+=delta*delta;
            }
            f.standard_error[i]=std::sqrt(squared_deviation/
                (static_cast<double>(walkers.size())*static_cast<double>(walkers.size()-1)));
        }
    }
    return f;
}

WindowSamplingStatistics summarize_sampling_impl(
    EnergyWindow window,std::span<const WangLandauWalker* const> walkers,std::size_t bins) {
    WindowSamplingStatistics result{window,std::vector<double>(bins,0.0),
                                    std::vector<std::uint64_t>(bins,0),0,
                                    static_cast<std::uint64_t>(walkers.size())};
    for(const auto* walker:walkers) {
        if(walker->squared_energy_displacement().empty()) continue;
        for(std::size_t i=window.begin;i<window.end;++i) {
            result.squared_energy_displacement[i]+=walker->squared_energy_displacement()[i];
            result.displacement_samples[i]+=walker->displacement_samples()[i];
        }
        result.round_trips+=walker->round_trips();
    }
    return result;
}

#ifdef WL_HAS_MPI
template<class T>
void allreduce_chunks(const T* input,T* output,std::size_t count,MPI_Datatype type,
                      MPI_Op operation,MPI_Comm communicator) {
    constexpr auto maximum=static_cast<std::size_t>(std::numeric_limits<int>::max());
    for(std::size_t offset=0;offset<count;) {
        const auto chunk=std::min(maximum,count-offset);
        MPI_Allreduce(input+offset,output+offset,static_cast<int>(chunk),type,operation,communicator);
        offset+=chunk;
    }
}

template<class T>
void gather_chunks(const T* input,std::vector<T>& output,std::size_t count,
                   MPI_Datatype type,int root,MPI_Comm communicator) {
    int rank=0,size=1;
    MPI_Comm_rank(communicator,&rank);
    MPI_Comm_size(communicator,&size);
    const auto ranks=static_cast<std::size_t>(size);
    if(count!=0&&ranks>std::numeric_limits<std::size_t>::max()/count)
        throw std::overflow_error("MPI gathered DOS size overflows size_t");
    if(rank==root) {
        output.resize(count*ranks);
    }
    constexpr std::size_t maximum=1'000'000;
    for(std::size_t offset=0;offset<count;) {
        const auto chunk=std::min(maximum,count-offset);
        std::vector<T> received;
        if(rank==root) received.resize(chunk*static_cast<std::size_t>(size));
        MPI_Gather(input+offset,static_cast<int>(chunk),type,
                   rank==root?received.data():nullptr,static_cast<int>(chunk),type,
                   root,communicator);
        if(rank==root)
            for(int source=0;source<size;++source)
                std::copy_n(received.begin()+static_cast<std::ptrdiff_t>(
                                static_cast<std::size_t>(source)*chunk),chunk,
                            output.begin()+static_cast<std::ptrdiff_t>(
                                static_cast<std::size_t>(source)*count+offset));
        offset+=chunk;
    }
}

template<class T>
void broadcast_chunks(T* values,std::size_t count,MPI_Datatype type,int root,
                      MPI_Comm communicator) {
    constexpr auto maximum=static_cast<std::size_t>(std::numeric_limits<int>::max());
    for(std::size_t offset=0;offset<count;) {
        const auto chunk=std::min(maximum,count-offset);
        MPI_Bcast(values+offset,static_cast<int>(chunk),type,root,communicator);
        offset+=chunk;
    }
}

template<class T>
void sendrecv_chunks(const T* send,T* receive,std::size_t count,MPI_Datatype type,
                     int partner,int tag,MPI_Comm communicator) {
    constexpr auto maximum=static_cast<std::size_t>(std::numeric_limits<int>::max());
    for(std::size_t offset=0;offset<count;) {
        const auto chunk=std::min(maximum,count-offset);
        MPI_Sendrecv(send+offset,static_cast<int>(chunk),type,partner,tag,
                     receive+offset,static_cast<int>(chunk),type,partner,tag,
                     communicator,MPI_STATUS_IGNORE);
        offset+=chunk;
    }
}

template<class T>
std::vector<std::uint64_t> gather_variable_chunks(
    const T* input,std::size_t count,std::vector<T>& output,MPI_Datatype type,
    int root,int tag,MPI_Comm communicator) {
    int rank=0,size=1;
    MPI_Comm_rank(communicator,&rank); MPI_Comm_size(communicator,&size);
    const auto local_count=static_cast<std::uint64_t>(count);
    std::vector<std::uint64_t> counts;
    if(rank==root) counts.resize(static_cast<std::size_t>(size));
    MPI_Gather(&local_count,1,MPI_UINT64_T,rank==root?counts.data():nullptr,1,
               MPI_UINT64_T,root,communicator);
    std::vector<std::size_t> offsets;
    int valid_layout=1;
    if(rank==root) {
        offsets.resize(static_cast<std::size_t>(size));
        std::size_t total=0;
        for(int source=0;source<size;++source) {
            offsets[static_cast<std::size_t>(source)]=total;
            const auto source_count=counts[static_cast<std::size_t>(source)];
            if(source_count>std::numeric_limits<std::size_t>::max()-total) {
                valid_layout=0; break;
            }
            total+=static_cast<std::size_t>(source_count);
        }
        if(valid_layout) output.resize(total);
    }
    MPI_Bcast(&valid_layout,1,MPI_INT,root,communicator);
    if(!valid_layout) throw std::overflow_error("Variable MPI gather size overflows size_t");
    constexpr auto maximum=static_cast<std::size_t>(std::numeric_limits<int>::max());
    if(rank==root) {
        if(count!=0) std::copy_n(input,count,output.begin()+static_cast<std::ptrdiff_t>(
            offsets[static_cast<std::size_t>(root)]));
        for(int source=0;source<size;++source) if(source!=root) {
            const auto source_count=static_cast<std::size_t>(counts[static_cast<std::size_t>(source)]);
            for(std::size_t offset=0;offset<source_count;) {
                const auto chunk=std::min(maximum,source_count-offset);
                MPI_Recv(output.data()+offsets[static_cast<std::size_t>(source)]+offset,
                         static_cast<int>(chunk),type,source,tag,communicator,MPI_STATUS_IGNORE);
                offset+=chunk;
            }
        }
    } else {
        for(std::size_t offset=0;offset<count;) {
            const auto chunk=std::min(maximum,count-offset);
            MPI_Send(input+offset,static_cast<int>(chunk),type,root,tag,communicator);
            offset+=chunk;
        }
    }
    return counts;
}

DosFragment summarize_distributed(EnergyWindow window,
                                  std::span<const std::unique_ptr<WangLandauWalker>> walkers,
                                  std::size_t bins,MPI_Comm communicator,
                                  bool allow_empty_intersection) {
    (void)bins;
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const auto layout=walkers.front()->dos_grid();
    const auto cells=layout.cells();
    if(layout.joint()) {
        if(walkers.size()!=0&&cells>std::numeric_limits<std::size_t>::max()/walkers.size())
            throw std::overflow_error("MPI local joint-DOS gather size overflows size_t");
        const auto local_cells=walkers.size()*cells;
        std::vector<double> local_log_g(local_cells);
        std::vector<std::uint64_t> local_histogram(local_cells);
        std::vector<std::uint8_t> local_active(local_cells);
        for(std::size_t w=0;w<walkers.size();++w) {
            std::copy(walkers[w]->log_g().begin(),walkers[w]->log_g().end(),
                      local_log_g.begin()+static_cast<std::ptrdiff_t>(w*cells));
            std::copy(walkers[w]->histogram().begin(),walkers[w]->histogram().end(),
                      local_histogram.begin()+static_cast<std::ptrdiff_t>(w*cells));
            std::copy(walkers[w]->active_mask().begin(),walkers[w]->active_mask().end(),
                      local_active.begin()+static_cast<std::ptrdiff_t>(w*cells));
        }
        std::vector<double> gathered_log_g;
        std::vector<std::uint64_t> gathered_histogram;
        std::vector<std::uint8_t> gathered_active;
        gather_chunks(local_log_g.data(),gathered_log_g,local_cells,MPI_DOUBLE,0,communicator);
        gather_chunks(local_histogram.data(),gathered_histogram,local_cells,MPI_UINT64_T,0,communicator);
        gather_chunks(local_active.data(),gathered_active,local_cells,MPI_UNSIGNED_CHAR,0,communicator);
        int communicator_rank=0,communicator_size=1;
        MPI_Comm_rank(communicator,&communicator_rank);
        MPI_Comm_size(communicator,&communicator_size);
        DosFragment result;
        if(communicator_rank==0) {
            const auto estimator_count=static_cast<std::size_t>(communicator_size)*walkers.size();
            std::vector<EstimatorView> estimators;
            estimators.reserve(estimator_count);
            for(std::size_t w=0;w<estimator_count;++w) {
                const auto offset=w*cells;
                estimators.push_back({
                    std::span<const double>(gathered_log_g).subspan(offset,cells),
                    std::span<const std::uint64_t>(gathered_histogram).subspan(offset,cells),
                    std::span<const std::uint8_t>(gathered_active).subspan(offset,cells)});
            }
            result=summarize_joint_union(window,layout,estimators);
        } else {
            result={window,std::vector<double>(cells,nan),std::vector<std::uint64_t>(cells),
                std::vector<double>(cells,nan),std::vector<std::uint8_t>(cells),layout,
                std::vector<std::uint32_t>(cells),std::vector<std::int32_t>(cells,-1)};
        }
        broadcast_chunks(result.log_g.data(),cells,MPI_DOUBLE,0,communicator);
        broadcast_chunks(result.histogram.data(),cells,MPI_UINT64_T,0,communicator);
        broadcast_chunks(result.standard_error.data(),cells,MPI_DOUBLE,0,communicator);
        broadcast_chunks(result.valid.data(),cells,MPI_UNSIGNED_CHAR,0,communicator);
        broadcast_chunks(result.contributors.data(),cells,MPI_UINT32_T,0,communicator);
        broadcast_chunks(result.support_component.data(),cells,MPI_INT32_T,0,communicator);
        return result;
    }
    DosFragment f{window,std::vector<double>(cells,nan),std::vector<std::uint64_t>(cells,0),
                  std::vector<double>(cells,nan),std::vector<std::uint8_t>(cells,0),
                  layout,{},{}};
    std::vector<std::uint64_t> local_histogram(cells,0),global_histogram(cells,0);
    std::vector<std::uint8_t> local_valid(cells,0),global_valid(cells,0);
    const auto begin=layout.flatten(window.begin),end=layout.flatten(window.end);
    for(std::size_t i=begin;i<end;++i) {
        local_valid[i]=1;
        for(const auto& walker:walkers) {
            local_valid[i]=static_cast<std::uint8_t>(local_valid[i]&&walker->active_mask()[i]!=0);
            local_histogram[i]+=walker->histogram()[i];
        }
    }
    allreduce_chunks(local_valid.data(),global_valid.data(),cells,MPI_UNSIGNED_CHAR,MPI_MIN,communicator);
    allreduce_chunks(local_histogram.data(),global_histogram.data(),cells,MPI_UINT64_T,MPI_SUM,communicator);
    f.valid=std::move(global_valid);
    f.histogram=std::move(global_histogram);
    const auto reference=std::find(f.valid.begin()+static_cast<std::ptrdiff_t>(begin),
                                   f.valid.begin()+static_cast<std::ptrdiff_t>(end),1);
    if(reference==f.valid.begin()+static_cast<std::ptrdiff_t>(end) && allow_empty_intersection)
        return f;
    if(reference==f.valid.begin()+static_cast<std::ptrdiff_t>(end))
        throw std::runtime_error("Walkers in an MPI energy window have no common active bin");
    const auto reference_bin=static_cast<std::size_t>(reference-f.valid.begin());
    std::uint64_t local_count=static_cast<std::uint64_t>(walkers.size()),walker_count=0;
    MPI_Allreduce(&local_count,&walker_count,1,MPI_UINT64_T,MPI_SUM,communicator);
    std::vector<double> local_sum(cells,0.0),global_sum(cells,0.0);
    for(const auto& walker:walkers)
        for(std::size_t i=begin;i<end;++i)
            if(f.valid[i]!=0) local_sum[i]+=walker->log_g()[i]-walker->log_g()[reference_bin];
    allreduce_chunks(local_sum.data(),global_sum.data(),cells,MPI_DOUBLE,MPI_SUM,communicator);
    for(std::size_t i=begin;i<end;++i)
        if(f.valid[i]!=0) f.log_g[i]=global_sum[i]/static_cast<double>(walker_count);
    if(walker_count>1) {
        std::vector<double> local_squared_deviation(cells,0.0),global_squared_deviation(cells,0.0);
        for(const auto& walker:walkers)
            for(std::size_t i=begin;i<end;++i) if(f.valid[i]!=0) {
                const auto value=walker->log_g()[i]-walker->log_g()[reference_bin];
                const auto delta=value-f.log_g[i];
                local_squared_deviation[i]+=delta*delta;
            }
        allreduce_chunks(local_squared_deviation.data(),global_squared_deviation.data(),cells,
                         MPI_DOUBLE,MPI_SUM,communicator);
        for(std::size_t i=begin;i<end;++i)
            if(f.valid[i]!=0) f.standard_error[i]=std::sqrt(global_squared_deviation[i]/
                (static_cast<double>(walker_count)*static_cast<double>(walker_count-1)));
    }
    return f;
}

WindowSamplingStatistics summarize_sampling_distributed(
    EnergyWindow window,std::span<const std::unique_ptr<WangLandauWalker>> walkers,
    std::size_t bins,MPI_Comm communicator) {
    std::vector<const WangLandauWalker*> pointers;
    pointers.reserve(walkers.size());
    for(const auto& walker:walkers) pointers.push_back(walker.get());
    auto local=summarize_sampling_impl(window,pointers,bins);
    WindowSamplingStatistics result{window,std::vector<double>(bins,0.0),
                                    std::vector<std::uint64_t>(bins,0)};
    allreduce_chunks(local.squared_energy_displacement.data(),
                     result.squared_energy_displacement.data(),bins,MPI_DOUBLE,
                     MPI_SUM,communicator);
    allreduce_chunks(local.displacement_samples.data(),result.displacement_samples.data(),
                     bins,MPI_UINT64_T,MPI_SUM,communicator);
    MPI_Allreduce(&local.round_trips,&result.round_trips,1,MPI_UINT64_T,MPI_SUM,communicator);
    MPI_Allreduce(&local.walkers,&result.walkers,1,MPI_UINT64_T,MPI_SUM,communicator);
    return result;
}
#endif

bool local_exchange(WangLandauWalker& left, WangLandauWalker& right,
                    std::uint64_t seed, std::uint64_t epoch, std::size_t boundary) {
    const auto log_probability=left.exchange_log_probability(right);
    Xoshiro256StarStar rng(seed ^ (epoch*0x9e3779b97f4a7c15ULL), boundary);
    if (std::isfinite(log_probability) &&
        (log_probability>=0.0 || std::log(std::max(rng.uniform(),0x1.0p-53))<log_probability)) {
        left.swap_configuration(right); return true;
    }
    return false;
}

} // namespace

DosFragment summarize_walkers(EnergyWindow window,
                              std::span<const WangLandauWalker* const> walkers,
                              std::size_t bins,bool allow_empty_intersection) {
    if(walkers.empty()) throw std::invalid_argument("No walkers to summarize");
    if(!walkers.front()) throw std::invalid_argument("Cannot summarize a null walker");
    const auto layout=walkers.front()->dos_grid();
    for(const auto* walker:walkers)
        if(!walker||walker->dos_grid().energy_bins()!=bins||
           walker->log_g().size()!=walker->dos_grid().cells()||walker->window().begin!=window.begin||
           walker->window().end!=window.end||walker->dos_grid().joint()!=layout.joint()||
           walker->dos_grid().q_bins()!=layout.q_bins())
            throw std::invalid_argument("Cannot summarize incompatible walkers");
    return summarize_impl(window,walkers,bins,allow_empty_intersection);
}

ParallelContext::ParallelContext(int& argc,char**& argv) {
#ifdef WL_HAS_MPI
    int initialized=0; MPI_Initialized(&initialized);
    if(!initialized) {
        int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
        if(provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI lacks MPI_THREAD_FUNNELED support");
        owns_mpi_=true;
    }
    mpi_enabled_=true; MPI_Comm_rank(MPI_COMM_WORLD,&rank_); MPI_Comm_size(MPI_COMM_WORLD,&size_);
#else
    (void)argc; (void)argv;
#endif
}

ParallelContext::~ParallelContext() {
#ifdef WL_HAS_MPI
    if(owns_mpi_) { int finalized=0; MPI_Finalized(&finalized); if(!finalized) MPI_Finalize(); }
#endif
}

std::uint64_t ParallelContext::broadcast_seed(std::uint64_t seed) const {
#ifdef WL_HAS_MPI
    MPI_Bcast(&seed,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
#endif
    return seed;
}

void ParallelContext::broadcast_windows(std::vector<EnergyWindow>& windows) const {
#ifdef WL_HAS_MPI
    std::uint64_t count=rank_==0?static_cast<std::uint64_t>(windows.size()):0;
    MPI_Bcast(&count,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(count>std::numeric_limits<std::size_t>::max()/2)
        throw std::overflow_error("MPI energy-window broadcast size overflows size_t");
    std::vector<std::uint64_t> packed(static_cast<std::size_t>(count)*2);
    if(rank_==0) for(std::size_t i=0;i<windows.size();++i) {
        packed[2*i]=static_cast<std::uint64_t>(windows[i].begin);
        packed[2*i+1]=static_cast<std::uint64_t>(windows[i].end);
    }
    broadcast_chunks(packed.data(),packed.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(rank_!=0) {
        windows.resize(static_cast<std::size_t>(count));
        for(std::size_t i=0;i<windows.size();++i)
            windows[i]={static_cast<std::size_t>(packed[2*i]),
                        static_cast<std::size_t>(packed[2*i+1])};
    }
#else
    (void)windows;
#endif
}

void ParallelContext::broadcast_spin_configurations(
    std::vector<std::vector<std::int8_t>>& configurations) const {
#ifdef WL_HAS_MPI
    std::uint64_t count=rank_==0?static_cast<std::uint64_t>(configurations.size()):0;
    MPI_Bcast(&count,1,MPI_UINT64_T,0,MPI_COMM_WORLD);
    if(count>std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("MPI configuration count overflows size_t");
    std::vector<std::uint64_t> sizes(static_cast<std::size_t>(count),0);
    if(rank_==0) for(std::size_t i=0;i<configurations.size();++i) {
        sizes[i]=static_cast<std::uint64_t>(configurations[i].size());
    }
    broadcast_chunks(sizes.data(),sizes.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
    std::size_t total=0;
    for(const auto size:sizes) {
        if(size>std::numeric_limits<std::size_t>::max()-total)
            throw std::overflow_error("Adaptive spin-configuration bank overflows size_t");
        total+=static_cast<std::size_t>(size);
    }
    std::vector<std::int8_t> packed(total);
    if(rank_==0) {
        std::size_t offset=0;
        for(const auto& configuration:configurations) {
            std::copy(configuration.begin(),configuration.end(),packed.begin()+
                      static_cast<std::ptrdiff_t>(offset));
            offset+=configuration.size();
        }
    }
    broadcast_chunks(packed.data(),packed.size(),MPI_BYTE,0,MPI_COMM_WORLD);
    if(rank_!=0) {
        configurations.resize(static_cast<std::size_t>(count));
        std::size_t offset=0;
        for(std::size_t i=0;i<configurations.size();++i) {
            configurations[i].assign(packed.begin()+static_cast<std::ptrdiff_t>(offset),
                                     packed.begin()+static_cast<std::ptrdiff_t>(offset+sizes[i]));
            offset+=static_cast<std::size_t>(sizes[i]);
        }
    }
#else
    (void)configurations;
#endif
}

int maximum_openmp_threads() noexcept {
#ifdef WL_HAS_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

RewlResult run_rewl(const ParallelContext& context, std::shared_ptr<const Couplings> couplings,
                    const RunConfig& c) {
    if(!couplings) throw std::invalid_argument("Null couplings");
    c.validate(!c.smoke_test);
    const auto windows=c.explicit_windows.empty()?
        partition_windows(c.grid.bins(),c.windows,c.overlap):c.explicit_windows;
    const bool distributed=context.size()>1;
    if(distributed&&!c.checkpoint_path.empty())
        throw std::invalid_argument(
            "MPI checkpoint/restart is not supported; checkpoint_path must be empty when MPI size > 1");
    if(!c.checkpoint_path.empty()&&(c.windows!=1||c.walkers_per_rank!=1))
        throw std::invalid_argument(
            "Exact checkpoint/restart requires one window and one walker");
    if(distributed && context.size()%static_cast<int>(c.windows)!=0)
        throw std::invalid_argument("MPI size must be a multiple of the number of energy windows");
    const auto shards=distributed?static_cast<std::size_t>(context.size())/c.windows:1;
    const auto expected_initial_configurations=(distributed?
        static_cast<std::size_t>(context.size()):c.windows)*c.walkers_per_rank;
    if(!c.initial_spins_by_walker.empty()) {
        if(c.initial_spins_by_walker.size()!=expected_initial_configurations)
            throw std::invalid_argument("Adaptive initial-configuration count mismatch");
        for(const auto& spins:c.initial_spins_by_walker)
            if(!spins.empty()&&spins.size()!=couplings->size())
                throw std::invalid_argument("Adaptive initial spin count mismatch");
    }
    const auto window_id=distributed?static_cast<std::size_t>(context.rank())/shards:0;
    const auto first_window=distributed?window_id:0;
    const auto last_window=distributed?window_id+1:c.windows;

#ifdef WL_HAS_MPI
    MPI_Comm window_comm=MPI_COMM_NULL;
    if(distributed) MPI_Comm_split(MPI_COMM_WORLD,static_cast<int>(window_id),context.rank(),&window_comm);
#endif

    std::vector<std::vector<std::unique_ptr<WangLandauWalker>>> groups(c.windows);
    std::string initialization_error;
    std::size_t initialization_window=first_window,initialization_walker=0;
    try {
        for(std::size_t w=first_window;w<last_window;++w) {
            initialization_window=w;
            auto& group=groups[w]; group.reserve(c.walkers_per_rank);
            for(std::size_t local=0;local<c.walkers_per_rank;++local) {
                initialization_walker=local;
                const auto global_id=(distributed?static_cast<std::uint64_t>(context.rank()):w)*
                                     c.walkers_per_rank+local;
                std::vector<std::int8_t> initial_spins;
                if(!c.initial_spins_by_walker.empty())
                    initial_spins=c.initial_spins_by_walker[static_cast<std::size_t>(global_id)];
                group.push_back(std::make_unique<WangLandauWalker>(
                    global_id,couplings,c.dos_grid(),windows[w],c.wl,c.seed,
                    c.order_parameter,std::move(initial_spins)));
            }
        }
    }
    catch(const std::exception& error) {
        std::ostringstream message;
        message<<"window "<<initialization_window<<", local walker "<<initialization_walker
               <<": "<<error.what();
        initialization_error=message.str();
    }

#ifdef WL_HAS_MPI
    if(distributed) {
        const int local_failed=initialization_error.empty()?0:1;
        int any_failed=0;
        MPI_Allreduce(&local_failed,&any_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if(any_failed!=0) {
            const int local_failure_rank=local_failed!=0?context.rank():context.size();
            int failure_rank=context.size();
            MPI_Allreduce(&local_failure_rank,&failure_rank,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
            int message_length=context.rank()==failure_rank?
                static_cast<int>(initialization_error.size()):0;
            MPI_Bcast(&message_length,1,MPI_INT,failure_rank,MPI_COMM_WORLD);
            std::string shared_error(static_cast<std::size_t>(message_length),'\0');
            if(context.rank()==failure_rank) shared_error=initialization_error;
            MPI_Bcast(shared_error.data(),message_length,MPI_CHAR,failure_rank,MPI_COMM_WORLD);
            if(window_comm!=MPI_COMM_NULL) MPI_Comm_free(&window_comm);
            throw std::runtime_error("MPI walker initialization failed on rank "+
                                     std::to_string(failure_rank)+": "+shared_error);
        }
    }
#endif
    if(!initialization_error.empty())
        throw std::runtime_error("Walker initialization failed: "+initialization_error);

    if(!c.checkpoint_path.empty() && c.windows==1 && c.walkers_per_rank==1) {
        try {
            const auto checkpoint=load_checkpoint(c.checkpoint_path);
            groups[first_window][0]->restore(checkpoint);
            if(checkpoint.format_version<5&&context.rank()==0)
                std::cerr<<"warning: restored a legacy 1D checkpoint without full grid metadata\n";
        }
        catch(const std::exception& error) {
            if(context.rank()==0)
                std::cerr<<"warning: checkpoint was not restored: "<<error.what()<<"; starting a new run\n";
        }
    }

    RewlResult result;
    std::uint64_t epoch=0,last_checkpoint=0;
    std::vector<std::vector<std::uint64_t>> last_flatness_check(c.windows);
    std::vector<std::vector<double>> last_flatness(c.windows);
    for(std::size_t w=first_window;w<last_window;++w) {
        last_flatness_check[w].assign(groups[w].size(),0);
        last_flatness[w].assign(groups[w].size(),0.0);
    }
    const auto progress_period=std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(c.progress_interval_seconds));
    auto next_progress=std::chrono::steady_clock::now()+progress_period;
    std::size_t progress_line_width=0;
    const bool unlimited_attempts=c.max_attempts==0;
    bool globally_done=false;
    while(!globally_done) {
        bool limit_reached=!unlimited_attempts;
        for(std::size_t w=first_window;w<last_window;++w) {
            auto& group=groups[w];
            #pragma omp parallel for schedule(static)
            for(std::int64_t li=0;li<static_cast<std::int64_t>(group.size());++li) {
                auto& walker=*group[static_cast<std::size_t>(li)];
                auto batch=c.exchange_interval_attempts;
                if(!unlimited_attempts) {
                    const auto remaining=walker.attempted()<c.max_attempts?
                        c.max_attempts-walker.attempted():0;
                    batch=std::min(batch,remaining);
                }
                walker.run_attempts(batch);
            }
            if(!unlimited_attempts)
                for(const auto& walker:group)
                    if(walker->attempted()<c.max_attempts) limit_reached=false;

            const auto check_interval=std::max<std::uint64_t>(1,c.wl.check_interval_attempts);
            for(std::size_t local=0;local<group.size();++local) {
                auto& walker=*group[local];
                if(walker.stage()!=RefinementStage::wang_landau) continue;
                const auto attempted=walker.attempted();
                const bool at_limit=!unlimited_attempts&&attempted>=c.max_attempts;
                if(attempted-last_flatness_check[w][local]<check_interval&&!at_limit) continue;
                last_flatness_check[w][local]=attempted;
                const auto histogram=walker.histogram_statistics();
                last_flatness[w][local]=c.wl.inverse_time_enabled?
                    histogram.coverage:histogram.min_over_mean;
                if(walker.ready_for_iteration()) walker.begin_next_iteration();
            }
        }

        const auto parity=epoch%2;
        if(!distributed) {
            for(std::size_t boundary=parity;boundary+1<c.windows;boundary+=2) {
                auto& left=groups[boundary]; auto& right=groups[boundary+1];
                const auto pairs=std::min(left.size(),right.size());
                for(std::size_t p=0;p<pairs;++p) {
                    ++result.exchange_attempted;
                    if(local_exchange(*left[p],*right[(p+epoch)%pairs],c.seed,epoch,boundary))
                        ++result.exchange_accepted;
                }
            }
        }
#ifdef WL_HAS_MPI
        else {
            int partner=-1;
            if(window_id%2==parity && window_id+1<c.windows) partner=context.rank()+static_cast<int>(shards);
            else if(window_id>0 && (window_id-1)%2==parity) partner=context.rank()-static_cast<int>(shards);
            if(partner>=0) {
                const auto slot=static_cast<std::size_t>(epoch%c.walkers_per_rank);
                auto& walker=*groups[window_id][slot];
                const auto own_energy=walker.energy();
                const auto own_order_parameter=walker.order_parameter();
                double own_state[2]{own_energy,own_order_parameter},other_state[2]{};
                MPI_Sendrecv(own_state,2,MPI_DOUBLE,partner,10,other_state,2,MPI_DOUBLE,partner,10,
                             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                const auto other_energy=other_state[0];
                const auto other_order_parameter=other_state[1];
                const auto own_bin=c.grid.index(own_energy),other_bin=c.grid.index(other_energy);
                double contribution=-std::numeric_limits<double>::infinity();
                const auto layout=c.dos_grid();
                const auto own_cell=layout.index(own_energy,own_order_parameter);
                const auto other_cell=layout.index(other_energy,other_order_parameter);
                if(own_bin&&other_bin&&own_cell&&other_cell&&windows[window_id].contains(*own_bin)&&
                   windows[window_id].contains(*other_bin))
                    contribution=walker.log_g()[*own_cell]-walker.log_g()[*other_cell];
                double other_contribution{};
                MPI_Sendrecv(&contribution,1,MPI_DOUBLE,partner,11,&other_contribution,1,MPI_DOUBLE,partner,11,
                             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                const auto logp=contribution+other_contribution;
                const auto boundary=std::min(window_id,static_cast<std::size_t>(partner)/shards);
                Xoshiro256StarStar exchange_rng(c.seed^(epoch*0x9e3779b97f4a7c15ULL),boundary);
                const bool accept=std::isfinite(logp)&&(logp>=0.0||
                    std::log(std::max(exchange_rng.uniform(),0x1.0p-53))<logp);
                ++result.exchange_attempted;
                if(accept) {
                    const auto own_spins=walker.spins();
                    const auto own_fields=walker.fields();
                    std::vector<std::int8_t> remote_spins(own_spins.size());
                    std::vector<double> remote_fields(own_fields.size());
                    sendrecv_chunks(own_spins.data(),remote_spins.data(),own_spins.size(),
                                    MPI_BYTE,partner,12,MPI_COMM_WORLD);
                    sendrecv_chunks(own_fields.data(),remote_fields.data(),own_fields.size(),
                                    MPI_DOUBLE,partner,13,MPI_COMM_WORLD);
                    walker.replace_configuration(remote_spins,remote_fields,other_energy,
                                                 other_order_parameter);
                    ++result.exchange_accepted;
                }
            }
        }
#endif
        ++epoch;
        if(c.progress_interval_seconds>0.0) {
            int due=0;
            if(context.rank()==0 && std::chrono::steady_clock::now()>=next_progress) due=1;
#ifdef WL_HAS_MPI
            if(distributed) MPI_Bcast(&due,1,MPI_INT,0,MPI_COMM_WORLD);
#endif
            if(due!=0) {
                std::uint64_t local_progress_attempted=0;
                double local_progress_factor=0.0;
                double local_progress_flatness=1.0;
                std::uint64_t local_stage_counts[3]{};
                for(std::size_t w=first_window;w<last_window;++w) {
                    const auto& group=groups[w];
                    local_progress_attempted=std::max(local_progress_attempted,group.front()->attempted());
                    for(std::size_t local=0;local<group.size();++local) {
                        const auto& walker=group[local];
                        local_progress_factor=std::max(local_progress_factor,walker->factor());
                        const auto stage=walker->stage();
                        ++local_stage_counts[stage==RefinementStage::wang_landau?0:
                                             stage==RefinementStage::inverse_time?1:2];
                        if(stage==RefinementStage::wang_landau)
                            local_progress_flatness=std::min(local_progress_flatness,
                                                            last_flatness[w][local]);
                    }
                }
                auto progress_attempted=local_progress_attempted;
                auto progress_factor=local_progress_factor;
                auto progress_flatness=local_progress_flatness;
                std::uint64_t stage_counts[3]{local_stage_counts[0],local_stage_counts[1],
                                              local_stage_counts[2]};
#ifdef WL_HAS_MPI
                if(distributed) {
                    MPI_Reduce(&local_progress_attempted,&progress_attempted,1,MPI_UINT64_T,MPI_MAX,0,
                               MPI_COMM_WORLD);
                    MPI_Reduce(&local_progress_factor,&progress_factor,1,MPI_DOUBLE,MPI_MAX,0,
                               MPI_COMM_WORLD);
                    MPI_Reduce(&local_progress_flatness,&progress_flatness,1,MPI_DOUBLE,MPI_MIN,0,
                               MPI_COMM_WORLD);
                    MPI_Reduce(local_stage_counts,stage_counts,3,MPI_UINT64_T,MPI_SUM,0,
                               MPI_COMM_WORLD);
                }
#endif
                if(context.rank()==0) {
                    std::ostringstream line;
                    line<<"progress mcs="
                        <<static_cast<double>(progress_attempted)/static_cast<double>(couplings->size())
                        <<" attempted_flips_per_walker="<<progress_attempted
                        <<(c.wl.inverse_time_enabled?" coverage=":" flatness=")
                        <<progress_flatness<<" factor="<<progress_factor
                        <<" stages="<<stage_counts[0]<<'/'<<stage_counts[1]<<'/'<<stage_counts[2];
                    const auto text=line.str();
                    std::cout<<'\r'<<text;
                    if(progress_line_width>text.size())
                        std::cout<<std::string(progress_line_width-text.size(),' ');
                    std::cout<<std::flush;
                    progress_line_width=std::max(progress_line_width,text.size());
                    next_progress=std::chrono::steady_clock::now()+progress_period;
                }
            }
        }
        if(!c.checkpoint_path.empty() && c.windows==1 && c.walkers_per_rank==1 &&
           groups[first_window][0]->attempted()-last_checkpoint>=c.checkpoint_interval_attempts) {
            save_checkpoint(c.checkpoint_path,groups[first_window][0]->snapshot());
            last_checkpoint=groups[first_window][0]->attempted();
        }
        bool local_done=true;
        for(std::size_t w=first_window;w<last_window;++w)
            for(const auto& walker:groups[w]) local_done&=walker->stage()==RefinementStage::frozen;
        globally_done=local_done||limit_reached;
#ifdef WL_HAS_MPI
        if(distributed) { int local=globally_done?1:0,all=0; MPI_Allreduce(&local,&all,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD); globally_done=all!=0; }
#endif
    }

    bool local_converged=true;
    if(c.progress_interval_seconds>0.0&&context.rank()==0&&progress_line_width!=0)
        std::cout<<'\n';
    std::uint64_t local_attempted=0,local_accepted=0;
    std::uint64_t local_forced_accepted=0;
    for(std::size_t w=first_window;w<last_window;++w) {
        for(const auto& walker:groups[w]) {
            local_attempted+=walker->attempted(); local_accepted+=walker->accepted();
            local_forced_accepted+=walker->forced_accepted();
            local_converged&=walker->stage()==RefinementStage::frozen;
            const auto last_accepted=walker->last_accepted_attempt();
            const auto histogram=walker->histogram_statistics();
            result.walker_statistics.push_back({
                walker->id(),static_cast<std::uint64_t>(w),context.rank(),
                walker->attempted(),walker->accepted(),walker->forced_accepted(),
                walker->attempted()-last_accepted,walker->energy(),
                walker->factor(),histogram.active_bins,histogram.minimum,
                histogram.mean,histogram.min_over_mean,walker->round_trips()});
            if(c.wl.collect_window_statistics)
                for(const auto& representative:walker->representatives())
                    if(!representative.spins.empty())
                        result.representatives.push_back(representative);
        }
#ifdef WL_HAS_MPI
        if(distributed) {
            result.fragments.push_back(summarize_distributed(windows[w],groups[w],c.grid.bins(),
                                                             window_comm,
                                                             c.wl.collect_window_statistics));
            if(c.wl.collect_window_statistics)
                result.sampling_statistics.push_back(summarize_sampling_distributed(
                    windows[w],groups[w],c.grid.bins(),window_comm));
        } else
#endif
        {
            std::vector<const WangLandauWalker*> pointers;
            pointers.reserve(groups[w].size());
            for(const auto& walker:groups[w]) pointers.push_back(walker.get());
            result.fragments.push_back(summarize_walkers(windows[w],pointers,c.grid.bins(),
                                                         c.wl.collect_window_statistics));
            if(c.wl.collect_window_statistics)
                result.sampling_statistics.push_back(summarize_sampling_impl(
                    windows[w],pointers,c.grid.bins()));
        }
    }
    result.attempted=local_attempted; result.accepted=local_accepted; result.converged=local_converged;
    result.forced_accepted=local_forced_accepted;

#ifdef WL_HAS_MPI
    if(distributed) {
        std::uint64_t totals[5]{local_attempted,local_accepted,local_forced_accepted,
                                result.exchange_attempted,result.exchange_accepted};
        std::uint64_t reduced[5]{}; MPI_Reduce(totals,reduced,5,MPI_UINT64_T,MPI_SUM,0,MPI_COMM_WORLD);
        int converged=local_converged?1:0,all_converged=0;
        MPI_Reduce(&converged,&all_converged,1,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
        const auto& local_fragment=result.fragments.front();
        const auto bins=c.grid.bins();
        const auto cells=c.dos_grid().cells();
        std::vector<double> all_logg,all_error;
        std::vector<std::uint64_t> all_hist;
        std::vector<std::uint8_t> all_valid;
        std::vector<std::uint32_t> all_contributors;
        std::vector<std::int32_t> all_components;
        gather_chunks(local_fragment.log_g.data(),all_logg,cells,MPI_DOUBLE,0,MPI_COMM_WORLD);
        gather_chunks(local_fragment.standard_error.data(),all_error,cells,MPI_DOUBLE,0,MPI_COMM_WORLD);
        gather_chunks(local_fragment.histogram.data(),all_hist,cells,MPI_UINT64_T,0,MPI_COMM_WORLD);
        gather_chunks(local_fragment.valid.data(),all_valid,cells,MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
        if(c.order_parameter) {
            gather_chunks(local_fragment.contributors.data(),all_contributors,cells,
                          MPI_UINT32_T,0,MPI_COMM_WORLD);
            gather_chunks(local_fragment.support_component.data(),all_components,cells,
                          MPI_INT32_T,0,MPI_COMM_WORLD);
        }
        const auto local_stat_count=result.walker_statistics.size();
        constexpr std::size_t statistic_integer_fields=10;
        constexpr std::size_t statistic_double_fields=4;
        std::vector<std::uint64_t> local_stat_integers(local_stat_count*statistic_integer_fields);
        std::vector<double> local_stat_doubles(local_stat_count*statistic_double_fields);
        for(std::size_t i=0;i<local_stat_count;++i) {
            const auto& statistic=result.walker_statistics[i];
            const auto integer_offset=i*statistic_integer_fields;
            const auto double_offset=i*statistic_double_fields;
            local_stat_integers[integer_offset]=statistic.walker_id;
            local_stat_integers[integer_offset+1]=statistic.window_id;
            local_stat_integers[integer_offset+2]=static_cast<std::uint64_t>(statistic.mpi_rank);
            local_stat_integers[integer_offset+3]=statistic.attempted;
            local_stat_integers[integer_offset+4]=statistic.accepted;
            local_stat_integers[integer_offset+5]=statistic.forced_accepted;
            local_stat_integers[integer_offset+6]=statistic.attempts_since_last_accepted;
            local_stat_integers[integer_offset+7]=static_cast<std::uint64_t>(statistic.active_bins);
            local_stat_integers[integer_offset+8]=statistic.minimum_histogram;
            local_stat_integers[integer_offset+9]=statistic.round_trips;
            local_stat_doubles[double_offset]=statistic.energy;
            local_stat_doubles[double_offset+1]=statistic.factor;
            local_stat_doubles[double_offset+2]=statistic.mean_histogram;
            local_stat_doubles[double_offset+3]=statistic.min_over_mean;
        }
        std::vector<std::uint64_t> all_stat_integers;
        std::vector<double> all_stat_doubles;
        gather_chunks(local_stat_integers.data(),all_stat_integers,
                      local_stat_integers.size(),MPI_UINT64_T,0,MPI_COMM_WORLD);
        gather_chunks(local_stat_doubles.data(),all_stat_doubles,
                      local_stat_doubles.size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        result.fragments.clear();
        std::vector<double> all_displacement;
        std::vector<std::uint64_t> all_displacement_samples;
        std::vector<std::uint64_t> all_sampling_scalars;
        if(c.wl.collect_window_statistics) {
            const auto& local_sampling=result.sampling_statistics.front();
            gather_chunks(local_sampling.squared_energy_displacement.data(),all_displacement,bins,
                          MPI_DOUBLE,0,MPI_COMM_WORLD);
            gather_chunks(local_sampling.displacement_samples.data(),all_displacement_samples,bins,
                          MPI_UINT64_T,0,MPI_COMM_WORLD);
            if(context.rank()==0) all_sampling_scalars.resize(2*context.size());
            const std::uint64_t local_sampling_scalars[2]{local_sampling.round_trips,
                                                           local_sampling.walkers};
            MPI_Gather(local_sampling_scalars,2,MPI_UINT64_T,all_sampling_scalars.data(),2,
                       MPI_UINT64_T,0,MPI_COMM_WORLD);
        }
        std::vector<double> all_representative_energies;
        std::vector<std::int8_t> all_representative_spins;
        if(c.wl.collect_window_statistics) {
            const auto local_representative_count=result.representatives.size();
            const int local_overflow=couplings->size()!=0&&local_representative_count>
                std::numeric_limits<std::size_t>::max()/couplings->size()?1:0;
            int any_overflow=0;
            MPI_Allreduce(&local_overflow,&any_overflow,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
            if(any_overflow)
                throw std::overflow_error("Local representative spin storage overflows size_t");
            std::vector<double> local_energies(local_representative_count);
            std::vector<std::int8_t> local_spins(local_representative_count*couplings->size());
            for(std::size_t i=0;i<local_representative_count;++i) {
                local_energies[i]=result.representatives[i].energy;
                std::copy(result.representatives[i].spins.begin(),
                          result.representatives[i].spins.end(),
                          local_spins.begin()+static_cast<std::ptrdiff_t>(i*couplings->size()));
            }
            (void)gather_variable_chunks(local_energies.data(),local_energies.size(),
                all_representative_energies,MPI_DOUBLE,0,201,MPI_COMM_WORLD);
            (void)gather_variable_chunks(local_spins.data(),local_spins.size(),
                all_representative_spins,MPI_BYTE,0,202,MPI_COMM_WORLD);
            int representative_payload_valid=1;
            if(context.rank()==0) {
                const auto count=all_representative_energies.size();
                representative_payload_valid=couplings->size()==0||
                    (count<=std::numeric_limits<std::size_t>::max()/couplings->size()&&
                     all_representative_spins.size()==count*couplings->size());
            }
            MPI_Bcast(&representative_payload_valid,1,MPI_INT,0,MPI_COMM_WORLD);
            if(!representative_payload_valid)
                throw std::runtime_error("MPI representative payload sizes are inconsistent");
        }
        result.sampling_statistics.clear();
        result.representatives.clear();
        result.walker_statistics.clear();
        if(context.rank()==0) {
            result.attempted=reduced[0]; result.accepted=reduced[1]; result.forced_accepted=reduced[2];
            result.exchange_attempted=reduced[3]/2; result.exchange_accepted=reduced[4]/2;
            result.converged=all_converged!=0;
            const auto total_stat_count=local_stat_count*static_cast<std::size_t>(context.size());
            for(std::size_t i=0;i<total_stat_count;++i) {
                const auto integer_offset=i*statistic_integer_fields;
                const auto double_offset=i*statistic_double_fields;
                result.walker_statistics.push_back({
                    all_stat_integers[integer_offset],all_stat_integers[integer_offset+1],
                    static_cast<int>(all_stat_integers[integer_offset+2]),
                    all_stat_integers[integer_offset+3],all_stat_integers[integer_offset+4],
                    all_stat_integers[integer_offset+5],all_stat_integers[integer_offset+6],
                    all_stat_doubles[double_offset],
                    all_stat_doubles[double_offset+1],
                    static_cast<std::size_t>(all_stat_integers[integer_offset+7]),
                    all_stat_integers[integer_offset+8],all_stat_doubles[double_offset+2],
                    all_stat_doubles[double_offset+3],all_stat_integers[integer_offset+9]});
            }
            for(std::size_t w=0;w<c.windows;++w) {
                const auto leader_rank=w*shards;
                const auto dos_offset=leader_rank*cells;
                DosFragment f{windows[w],std::vector<double>(cells),
                              std::vector<std::uint64_t>(cells),std::vector<double>(cells),
                              std::vector<std::uint8_t>(cells),c.dos_grid(),{}, {}};
                std::copy_n(all_logg.begin()+static_cast<std::ptrdiff_t>(dos_offset),cells,f.log_g.begin());
                std::copy_n(all_hist.begin()+static_cast<std::ptrdiff_t>(dos_offset),cells,f.histogram.begin());
                std::copy_n(all_error.begin()+static_cast<std::ptrdiff_t>(dos_offset),cells,
                             f.standard_error.begin());
                std::copy_n(all_valid.begin()+static_cast<std::ptrdiff_t>(dos_offset),cells,f.valid.begin());
                if(c.order_parameter) {
                    f.contributors.resize(cells);
                    f.support_component.resize(cells);
                    std::copy_n(all_contributors.begin()+static_cast<std::ptrdiff_t>(dos_offset),cells,
                                f.contributors.begin());
                    std::copy_n(all_components.begin()+static_cast<std::ptrdiff_t>(dos_offset),cells,
                                f.support_component.begin());
                }
                result.fragments.push_back(std::move(f));
                if(c.wl.collect_window_statistics) {
                    WindowSamplingStatistics sampling_result{
                        windows[w],std::vector<double>(bins),std::vector<std::uint64_t>(bins),
                        all_sampling_scalars[2*leader_rank],
                        all_sampling_scalars[2*leader_rank+1]};
                    const auto sampling_offset=leader_rank*bins;
                    std::copy_n(all_displacement.begin()+static_cast<std::ptrdiff_t>(sampling_offset),bins,
                                sampling_result.squared_energy_displacement.begin());
                    std::copy_n(all_displacement_samples.begin()+static_cast<std::ptrdiff_t>(sampling_offset),
                                bins,sampling_result.displacement_samples.begin());
                    result.sampling_statistics.push_back(std::move(sampling_result));
                }
            }
            if(c.wl.collect_window_statistics) {
                const auto count=all_representative_energies.size();
                result.representatives.reserve(count);
                for(std::size_t i=0;i<count;++i) {
                    EnergyRepresentative representative;
                    representative.energy=all_representative_energies[i];
                    const auto offset=i*couplings->size();
                    representative.spins.assign(
                        all_representative_spins.begin()+static_cast<std::ptrdiff_t>(offset),
                        all_representative_spins.begin()+
                            static_cast<std::ptrdiff_t>(offset+couplings->size()));
                    result.representatives.push_back(std::move(representative));
                }
            }
        }
        MPI_Comm_free(&window_comm);
    }
#endif
    return result;
}

} // namespace wl
