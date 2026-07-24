#include "wl/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
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

DosFragment summarize_impl(EnergyWindow window,
                           std::span<const WangLandauWalker* const> walkers,std::size_t bins,
                           bool allow_empty_intersection) {
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const auto layout=walkers.front()->dos_grid();
    const auto cells=layout.cells();
    DosFragment f{window,std::vector<double>(cells,nan),std::vector<std::uint64_t>(cells,0),
                  std::vector<double>(cells,nan),std::vector<std::uint8_t>(cells,0)};
    f.grid=layout;
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

DosFragment summarize_distributed(EnergyWindow window,
                                  std::span<const std::unique_ptr<WangLandauWalker>> walkers,
                                  std::size_t bins,MPI_Comm communicator,
                                  bool allow_empty_intersection) {
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const auto layout=walkers.front()->dos_grid();
    const auto cells=layout.cells();
    DosFragment f{window,std::vector<double>(cells,nan),std::vector<std::uint64_t>(cells,0),
                  std::vector<double>(cells,nan),std::vector<std::uint8_t>(cells,0)};
    f.grid=layout;
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
    MPI_Allreduce(local.squared_energy_displacement.data(),
                  result.squared_energy_displacement.data(),static_cast<int>(bins),MPI_DOUBLE,
                  MPI_SUM,communicator);
    MPI_Allreduce(local.displacement_samples.data(),result.displacement_samples.data(),
                  static_cast<int>(bins),MPI_UINT64_T,MPI_SUM,communicator);
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
    std::vector<std::uint64_t> packed(static_cast<std::size_t>(count)*2);
    if(rank_==0) for(std::size_t i=0;i<windows.size();++i) {
        packed[2*i]=static_cast<std::uint64_t>(windows[i].begin);
        packed[2*i+1]=static_cast<std::uint64_t>(windows[i].end);
    }
    MPI_Bcast(packed.data(),static_cast<int>(packed.size()),MPI_UINT64_T,0,MPI_COMM_WORLD);
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
    std::vector<std::uint64_t> sizes(static_cast<std::size_t>(count),0);
    std::uint64_t total=0;
    if(rank_==0) for(std::size_t i=0;i<configurations.size();++i) {
        sizes[i]=static_cast<std::uint64_t>(configurations[i].size());
        total+=sizes[i];
    }
    MPI_Bcast(sizes.data(),static_cast<int>(sizes.size()),MPI_UINT64_T,0,MPI_COMM_WORLD);
    for(const auto size:sizes) total+=rank_==0?0:size;
    if(total>static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Adaptive spin-configuration bank is too large for MPI");
    std::vector<std::int8_t> packed(static_cast<std::size_t>(total));
    if(rank_==0) {
        std::size_t offset=0;
        for(const auto& configuration:configurations) {
            std::copy(configuration.begin(),configuration.end(),packed.begin()+
                      static_cast<std::ptrdiff_t>(offset));
            offset+=configuration.size();
        }
    }
    MPI_Bcast(packed.data(),static_cast<int>(packed.size()),MPI_BYTE,0,MPI_COMM_WORLD);
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
        try { groups[first_window][0]->restore(load_checkpoint(c.checkpoint_path)); }
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
                auto& walker=*groups[window_id][slot]; auto own=walker.snapshot();
                double own_state[2]{own.energy,own.order_parameter},other_state[2]{};
                MPI_Sendrecv(own_state,2,MPI_DOUBLE,partner,10,other_state,2,MPI_DOUBLE,partner,10,
                             MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                const auto other_energy=other_state[0];
                const auto other_order_parameter=other_state[1];
                const auto own_bin=c.grid.index(own.energy),other_bin=c.grid.index(other_energy);
                double contribution=-std::numeric_limits<double>::infinity();
                const auto layout=c.dos_grid();
                const auto own_cell=layout.index(own.energy,own.order_parameter);
                const auto other_cell=layout.index(other_energy,other_order_parameter);
                if(own_bin&&other_bin&&own_cell&&other_cell&&windows[window_id].contains(*own_bin)&&
                   windows[window_id].contains(*other_bin))
                    contribution=own.log_g[*own_cell]-own.log_g[*other_cell];
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
                    std::vector<std::int8_t> remote_spins(own.spins.size());
                    std::vector<double> remote_fields(own.fields.size());
                    MPI_Sendrecv(own.spins.data(),static_cast<int>(own.spins.size()),MPI_BYTE,partner,12,
                                 remote_spins.data(),static_cast<int>(remote_spins.size()),MPI_BYTE,partner,12,
                                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
                    MPI_Sendrecv(own.fields.data(),static_cast<int>(own.fields.size()),MPI_DOUBLE,partner,13,
                                 remote_fields.data(),static_cast<int>(remote_fields.size()),MPI_DOUBLE,partner,13,
                                 MPI_COMM_WORLD,MPI_STATUS_IGNORE);
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
        const auto& local_fragment=result.fragments.front(); const auto bins=c.grid.bins();
        std::vector<double> all_logg,all_error;
        std::vector<std::uint64_t> all_hist;
        std::vector<std::uint8_t> all_valid;
        if(context.rank()==0) {
            all_logg.resize(bins*context.size()); all_error.resize(bins*context.size());
            all_hist.resize(bins*context.size()); all_valid.resize(bins*context.size());
        }
        MPI_Gather(local_fragment.log_g.data(),static_cast<int>(bins),MPI_DOUBLE,all_logg.data(),static_cast<int>(bins),MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Gather(local_fragment.standard_error.data(),static_cast<int>(bins),MPI_DOUBLE,all_error.data(),static_cast<int>(bins),MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Gather(local_fragment.histogram.data(),static_cast<int>(bins),MPI_UINT64_T,all_hist.data(),static_cast<int>(bins),MPI_UINT64_T,0,MPI_COMM_WORLD);
        MPI_Gather(local_fragment.valid.data(),static_cast<int>(bins),MPI_UNSIGNED_CHAR,
                   all_valid.data(),static_cast<int>(bins),MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);
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
        if(context.rank()==0) {
            all_stat_integers.resize(local_stat_integers.size()*context.size());
            all_stat_doubles.resize(local_stat_doubles.size()*context.size());
        }
        MPI_Gather(local_stat_integers.data(),static_cast<int>(local_stat_integers.size()),MPI_UINT64_T,
                   all_stat_integers.data(),static_cast<int>(local_stat_integers.size()),MPI_UINT64_T,0,MPI_COMM_WORLD);
        MPI_Gather(local_stat_doubles.data(),static_cast<int>(local_stat_doubles.size()),MPI_DOUBLE,
                   all_stat_doubles.data(),static_cast<int>(local_stat_doubles.size()),MPI_DOUBLE,0,MPI_COMM_WORLD);
        result.fragments.clear();
        std::vector<double> all_displacement;
        std::vector<std::uint64_t> all_displacement_samples;
        std::vector<std::uint64_t> all_sampling_scalars;
        if(c.wl.collect_window_statistics) {
            const auto& local_sampling=result.sampling_statistics.front();
            if(context.rank()==0) {
                all_displacement.resize(bins*context.size());
                all_displacement_samples.resize(bins*context.size());
                all_sampling_scalars.resize(2*context.size());
            }
            MPI_Gather(local_sampling.squared_energy_displacement.data(),static_cast<int>(bins),
                       MPI_DOUBLE,all_displacement.data(),static_cast<int>(bins),MPI_DOUBLE,0,
                       MPI_COMM_WORLD);
            MPI_Gather(local_sampling.displacement_samples.data(),static_cast<int>(bins),MPI_UINT64_T,
                       all_displacement_samples.data(),static_cast<int>(bins),MPI_UINT64_T,0,
                       MPI_COMM_WORLD);
            const std::uint64_t local_sampling_scalars[2]{local_sampling.round_trips,
                                                           local_sampling.walkers};
            MPI_Gather(local_sampling_scalars,2,MPI_UINT64_T,all_sampling_scalars.data(),2,
                       MPI_UINT64_T,0,MPI_COMM_WORLD);
        }
        std::vector<double> all_representative_energies;
        std::vector<std::int8_t> all_representative_spins;
        if(c.wl.collect_window_statistics) {
            const auto local_representative_count=result.representatives.size();
            std::vector<double> local_energies(local_representative_count);
            std::vector<std::int8_t> local_spins(local_representative_count*couplings->size());
            for(std::size_t i=0;i<local_representative_count;++i) {
                local_energies[i]=result.representatives[i].energy;
                std::copy(result.representatives[i].spins.begin(),
                          result.representatives[i].spins.end(),
                          local_spins.begin()+static_cast<std::ptrdiff_t>(i*couplings->size()));
            }
            if(context.rank()==0) {
                all_representative_energies.resize(local_representative_count*context.size());
                all_representative_spins.resize(local_spins.size()*context.size());
            }
            MPI_Gather(local_energies.data(),static_cast<int>(local_energies.size()),MPI_DOUBLE,
                       all_representative_energies.data(),static_cast<int>(local_energies.size()),
                       MPI_DOUBLE,0,MPI_COMM_WORLD);
            MPI_Gather(local_spins.data(),static_cast<int>(local_spins.size()),MPI_BYTE,
                       all_representative_spins.data(),static_cast<int>(local_spins.size()),
                       MPI_BYTE,0,MPI_COMM_WORLD);
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
                const auto offset=leader_rank*bins;
                DosFragment f{windows[w],std::vector<double>(bins),
                              std::vector<std::uint64_t>(bins),std::vector<double>(bins),
                              std::vector<std::uint8_t>(bins)};
                std::copy_n(all_logg.begin()+static_cast<std::ptrdiff_t>(offset),bins,f.log_g.begin());
                std::copy_n(all_hist.begin()+static_cast<std::ptrdiff_t>(offset),bins,f.histogram.begin());
                std::copy_n(all_error.begin()+static_cast<std::ptrdiff_t>(offset),bins,
                            f.standard_error.begin());
                std::copy_n(all_valid.begin()+static_cast<std::ptrdiff_t>(offset),bins,f.valid.begin());
                result.fragments.push_back(std::move(f));
                if(c.wl.collect_window_statistics) {
                    WindowSamplingStatistics sampling_result{
                        windows[w],std::vector<double>(bins),std::vector<std::uint64_t>(bins),
                        all_sampling_scalars[2*leader_rank],
                        all_sampling_scalars[2*leader_rank+1]};
                    std::copy_n(all_displacement.begin()+static_cast<std::ptrdiff_t>(offset),bins,
                                sampling_result.squared_energy_displacement.begin());
                    std::copy_n(all_displacement_samples.begin()+static_cast<std::ptrdiff_t>(offset),
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
