#include "wl/parallel.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <filesystem>
#include <random>
#include <vector>

namespace {
std::uint64_t mix_seed(std::uint64_t value) noexcept {
    value+=0x9e3779b97f4a7c15ULL;
    value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
    value=(value^(value>>27))*0x94d049bb133111ebULL;
    return value^(value>>31);
}

std::uint64_t random_seed() {
    auto seed=static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    try {
        std::random_device device;
        for(int i=0;i<4;++i) seed=mix_seed(seed^static_cast<std::uint64_t>(device()));
    } catch(const std::exception&) {}
    return mix_seed(seed);
}

}

int main(int argc,char** argv) {
    try {
        wl::ParallelContext parallel(argc,argv);
        for(int i=1;i<argc;++i) if(std::string_view(argv[i])=="--help"||std::string_view(argv[i])=="-h") {
            if(parallel.rank()==0) std::cout<<wl::usage(argv[0]);
            return 0;
        }
        auto config=wl::parse_arguments(argc,argv);
        if(!config.seed_explicit&&parallel.rank()==0) config.seed=random_seed();
        config.seed=parallel.broadcast_seed(config.seed);
        if(parallel.rank()==0)
            std::cout<<"seed="<<config.seed<<" source="<<(config.seed_explicit?"configured":"random")<<'\n';
        wl::Geometry geometry;
        if(config.geometry_file.empty())
            geometry=wl::Geometry::simple_cubic(config.nx,config.ny,config.nz,config.spacing,config.axis,config.periodic);
        else if(std::filesystem::path(config.geometry_file).extension()==".csv")
            geometry=wl::Geometry::load_csv(config.geometry_file,{config.box_lengths,config.periodic});
        else
            geometry=wl::Geometry::load_xyz_axes(config.geometry_file,{config.box_lengths,config.periodic});
        config.resolve_order_parameter(geometry);
        config.resolve_mcs(geometry.size());
        config.validate(!config.smoke_test&&!config.pilot);
        if(config.pilot && config.max_attempts==0)
            throw std::invalid_argument("Pilot mode requires a positive max-mcs value");
        if(parallel.rank()==0) {
            std::cout<<"spins="<<geometry.size()
                     <<" exchange_interval_mcs="<<config.exchange_interval_mcs
                     <<" check_interval_mcs="<<config.check_interval_mcs
                     <<" max_mcs="<<config.max_mcs
                     <<" checkpoint_interval_mcs="<<config.checkpoint_interval_mcs
                     <<" force_accept_after_mcs="<<config.force_accept_after_mcs
                     <<" inverse_time="<<(config.wl.inverse_time_enabled?"yes":"no")
                     <<" initialization_max_attempts="<<config.wl.initialization_max_attempts
                     <<" initialization_target_fraction="<<config.wl.initialization_target_fraction
                     <<" initialization_temperature_fraction="
                     <<config.wl.initialization_temperature_fraction
                     <<" initialization_stall_attempts_per_spin="
                     <<config.wl.initialization_stall_attempts_per_spin
                     <<" initialization_temperature_multiplier="
                     <<config.wl.initialization_temperature_multiplier
                     <<" initialization_max_temperature_fraction="
                     <<config.wl.initialization_max_temperature_fraction
                     <<" progress_interval_seconds="<<config.progress_interval_seconds<<'\n';
            const auto layout=config.dos_grid();
            const auto bytes_per_cell=sizeof(double)+sizeof(std::uint64_t)+
                sizeof(std::uint8_t)*(config.wl.nalivaiko_mod?2:1);
            if(layout.cells()>std::numeric_limits<std::size_t>::max()/bytes_per_cell)
                throw std::overflow_error("DOS memory estimate overflows size_t");
            const auto bytes_per_walker=layout.cells()*bytes_per_cell;
            std::cout<<"dos_dimensions="<<(layout.joint()?2:1)<<" dos_cells="<<layout.cells()
                     <<" approximate_core_bytes_per_walker="<<bytes_per_walker<<'\n'
                     <<"output_prefix_absolute="
                     <<std::filesystem::absolute(config.output_prefix).string()<<'\n';
            if(config.uses_legacy_attempt_units())
                std::cerr<<"warning: legacy attempt-based interval option used; prefer the corresponding *_mcs option\n";
            if(!config.pilot && config.wl.force_accept_after_attempts!=0)
                std::cerr<<"warning: force_accept_after_mcs="<<config.force_accept_after_mcs
                         <<" enables non-standard forced acceptance and may bias the DOS\n";
        }
        std::shared_ptr<const wl::Couplings> couplings;
        if(config.cutoff>0.0) couplings=std::make_shared<wl::CsrCouplings>(geometry,config.coupling_scale,config.cutoff);
        else couplings=std::make_shared<wl::DenseCouplings>(geometry,config.coupling_scale);
        if(config.pilot) {
            if(parallel.rank()==0) {
                std::vector<std::int8_t> spins(geometry.size(),1);
                auto fields=wl::local_fields(*couplings,spins);
                auto energy=wl::total_energy(*couplings,spins);
                auto minimum=energy,maximum=energy; wl::Xoshiro256StarStar rng(config.seed);
                for(std::uint64_t step=0;step<config.max_attempts;++step) {
                    const auto i=static_cast<std::size_t>(rng.bounded(spins.size()));
                    const auto old=spins[i]; energy+=wl::flip_delta(i,spins,fields);
                    couplings->add_flip_delta(i,old,fields); spins[i]=-old;
                    minimum=std::min(minimum,energy); maximum=std::max(maximum,energy);
                }
                const auto padding=std::max(1e-12,0.05*(maximum-minimum));
                std::cout<<"pilot_only=true attempted_flips="<<config.max_attempts
                         <<" mcs="<<config.max_mcs
                         <<" sampled_min="<<minimum<<" sampled_max="<<maximum
                         <<" suggested_emin="<<minimum-padding<<" suggested_emax="<<maximum+padding
                         <<"\nwarning: sampled bounds do not prove the complete energy range\n";
            }
            return 0;
        }
        if(config.adaptive_windows.enabled) {
            auto windows=wl::partition_windows(config.grid.bins(),config.windows,config.overlap);
            std::vector<std::vector<std::int8_t>> initial_configurations;
            const auto adaptation_start=std::chrono::steady_clock::now();
            for(std::size_t iteration=0;iteration<config.adaptive_windows.iterations;++iteration) {
                if(parallel.rank()==0)
                    std::cout<<"adaptive_window_pilot="<<(iteration+1)<<'/'
                             <<config.adaptive_windows.iterations
                             <<" pilot_mcs="<<config.adaptive_windows.pilot_mcs<<'\n';
                auto pilot_config=config;
                pilot_config.adaptive_windows.enabled=false;
                pilot_config.explicit_windows=windows;
                pilot_config.initial_spins_by_walker=initial_configurations;
                pilot_config.wl.collect_window_statistics=true;
                pilot_config.max_mcs=config.adaptive_windows.pilot_mcs;
                pilot_config.max_limit_uses_mcs=true;
                pilot_config.checkpoint_path.clear();
                pilot_config.resolved_spin_count=0;
                pilot_config.seed=config.seed^
                    ((static_cast<std::uint64_t>(iteration)+1)*0x9e3779b97f4a7c15ULL);
                pilot_config.resolve_mcs(geometry.size());
                const auto pilot_result=wl::run_rewl(parallel,couplings,pilot_config);
                std::vector<std::vector<std::int8_t>> next_initial_configurations;
                if(parallel.rank()==0) {
                    auto adaptive_fragments=pilot_result.fragments;
                    if(config.order_parameter)
                        for(auto& fragment:adaptive_fragments) {
                            const auto components=wl::support_component_count(fragment);
                            if(components!=1) throw wl::DisconnectedSupportError(components);
                            fragment=wl::marginalize_fragment(fragment);
                        }
                    windows=wl::adapt_energy_windows(config.grid,adaptive_fragments,
                        pilot_result.sampling_statistics,config.windows,config.overlap,
                        config.adaptive_windows);
                    std::uint64_t round_trips=0;
                    for(const auto& statistic:pilot_result.sampling_statistics)
                        round_trips+=statistic.round_trips;
                    std::size_t missing_configurations=0;
                    std::size_t external_warm_starts=0;
                    auto minimum_seed_energy=std::numeric_limits<double>::infinity();
                    for(const auto& representative:pilot_result.representatives)
                        if(!representative.spins.empty())
                            minimum_seed_energy=std::min(minimum_seed_energy,representative.energy);
                    next_initial_configurations=wl::select_adaptive_initial_configurations(
                        config.grid,windows,pilot_result.representatives,parallel.size(),
                        config.walkers_per_rank,missing_configurations,external_warm_starts);
                    std::cout<<"adaptive_window_round_trips="<<round_trips
                             <<" seeded_walkers="
                             <<next_initial_configurations.size()-missing_configurations<<'/'
                             <<next_initial_configurations.size()
                             <<" external_warm_starts="<<external_warm_starts;
                    if(std::isfinite(minimum_seed_energy))
                        std::cout<<" minimum_seed_energy="<<minimum_seed_energy;
                    std::cout<<" energy_ranges=";
                    for(std::size_t w=0;w<windows.size();++w) {
                        if(w!=0) std::cout<<';';
                        std::cout<<'['
                            <<config.grid.minimum+static_cast<double>(windows[w].begin)*config.grid.width
                            <<','
                            <<config.grid.minimum+static_cast<double>(windows[w].end)*config.grid.width
                            <<')';
                    }
                    std::cout<<'\n';
                }
                parallel.broadcast_windows(windows);
                parallel.broadcast_spin_configurations(next_initial_configurations);
                initial_configurations=std::move(next_initial_configurations);
            }
            config.explicit_windows=std::move(windows);
            config.initial_spins_by_walker=std::move(initial_configurations);
            config.wl.collect_window_statistics=false;
            config.validate(!config.smoke_test);
            if(parallel.rank()==0)
                std::cout<<"adaptive_window_elapsed_seconds="
                         <<std::chrono::duration<double>(
                               std::chrono::steady_clock::now()-adaptation_start).count()<<'\n';
        }
        const auto start=std::chrono::steady_clock::now();
        const auto result=wl::run_rewl(parallel,couplings,config);
        const auto elapsed_seconds=
            std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        bool postprocessing_failed=false;
        if(parallel.rank()==0) {
            std::string postprocessing_status=result.converged?"complete":"nonconverged";
            std::optional<wl::JointDensityOfStates> joint;
            std::optional<wl::DensityOfStates> dos;
            if(config.order_parameter) {
                try {
                    joint=wl::stitch_joint_dos(config.dos_grid(),result.fragments,
                        config.complete_range,result.converged,geometry.size(),
                        config.order_parameter->normalization);
                } catch(const wl::DisconnectedSupportError& error) {
                    postprocessing_status="disconnected_support";
                    postprocessing_failed=result.converged;
                    std::cerr<<"warning: "<<error.what()
                             <<"; joint DOS observables were not written\n";
                } catch(const std::exception& error) {
                    postprocessing_status="postprocessing_error";
                    postprocessing_failed=true;
                    std::cerr<<"warning: joint DOS postprocessing failed: "<<error.what()
                             <<"; raw diagnostics will still be written\n";
                }
            } else {
                try {
                    dos=wl::stitch_dos(config.grid,result.fragments,config.complete_range,
                                       geometry.size());
                } catch(const wl::InsufficientSupportError& error) {
                    postprocessing_status="insufficient_support";
                    postprocessing_failed=result.converged;
                    std::cerr<<"warning: "<<error.what()
                             <<"; global DOS observables were not written\n";
                } catch(const std::exception& error) {
                    postprocessing_status="postprocessing_error";
                    postprocessing_failed=true;
                    std::cerr<<"warning: DOS postprocessing failed: "<<error.what()
                             <<"; raw diagnostics will still be written\n";
                }
            }

            // Write the small diagnostic files before raw fragments and derived observables.
            // A later CSV or postprocessing failure can no longer leave a completed run silent.
            wl::write_metadata_json(config.output_prefix+"_metadata.json",config,*couplings,
                                    result.attempted,result.accepted,result.forced_accepted,
                                    result.exchange_attempted,result.exchange_accepted,
                                    result.converged,parallel.size(),wl::maximum_openmp_threads(),
                                    postprocessing_status);
            const bool no_global_dos=config.order_parameter?!joint:!dos;
            if(!result.converged||no_global_dos)
                wl::write_workers_stat_csv(config.output_prefix+"_workers_stat.csv",
                                           result.walker_statistics,geometry.size());

            if(config.order_parameter) {
                for(std::size_t i=0;i<result.fragments.size();++i)
                    wl::write_joint_fragment_csv(config.output_prefix+"_window_"+
                        std::to_string(i)+"_dos2d.csv",result.fragments[i],
                        config.order_parameter->normalization,i);
                if(joint) {
                    dos=wl::marginalize(*joint);
                    wl::write_joint_dos_csv(config.output_prefix+"_dos2d.csv",*joint);
                    wl::write_dos_csv(config.output_prefix+"_dos.csv",*dos);
                    wl::write_thermodynamics_csv(config.output_prefix+"_thermo.csv",
                                                 wl::thermodynamics(*dos,config.temperatures));
                    wl::write_order_parameter_thermodynamics_csv(config.output_prefix+"_q_thermo.csv",
                        wl::order_parameter_thermodynamics(*joint,config.temperatures,geometry.size()));
                    wl::write_order_parameter_distribution_csv(config.output_prefix+"_q_distribution.csv",
                        wl::order_parameter_distribution(*joint,config.temperatures));
                }
            } else {
                for(std::size_t i=0;i<result.fragments.size();++i)
                    wl::write_fragment_csv(config.output_prefix+"_window_"+std::to_string(i)+".csv",
                                           config.grid,result.fragments[i],i);
                if(dos) {
                    wl::write_dos_csv(config.output_prefix+"_dos.csv",*dos);
                    wl::write_thermodynamics_csv(config.output_prefix+"_thermo.csv",
                                                 wl::thermodynamics(*dos,config.temperatures));
                }
            }
            std::cout<<"attempted_flips="<<result.attempted
                     <<" aggregate_mcs="<<static_cast<double>(result.attempted)/static_cast<double>(geometry.size())
                     <<" accepted="<<result.accepted
                     <<" forced="<<result.forced_accepted
                     <<" exchanges="<<result.exchange_accepted<<'/'<<result.exchange_attempted
                     <<" converged="<<(result.converged?"yes":"no")
                     <<" elapsed_seconds="<<elapsed_seconds<<'\n';
            if(!result.converged)
                std::cerr<<"warning: max-mcs reached before final factor; walker statistics written to "
                         <<config.output_prefix<<"_workers_stat.csv\n";
        }
        if(postprocessing_failed) return 1;
        return result.converged||config.smoke_test?0:2;
    } catch(const std::exception& error) {
        std::cerr<<"wl_run: "<<error.what()<<'\n'; return 1;
    }
}
