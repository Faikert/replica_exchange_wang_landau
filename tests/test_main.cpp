#include "wl/parallel.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
void near(double a,double b,double tolerance,const char* message) {
    if(std::abs(a-b)>tolerance*std::max({1.0,std::abs(a),std::abs(b)}))
        throw std::runtime_error(message);
}

void test_pair_formula() {
    near(wl::dipolar_pair({1,0,0},{1,0,0},{1,0,0},1),-2,1e-14,"head-to-tail pair");
    near(wl::dipolar_pair({1,0,0},{0,0,1},{0,0,1},1),1,1e-14,"side-by-side pair");
    near(wl::dipolar_pair({2,0,0},{0,0,1},{0,0,1},1),0.125,1e-14,"inverse cube");
}

void test_minimum_image() {
    const wl::Box box{{10,8,6},true}; const auto d=wl::minimum_image({6,-5,3.1},box);
    near(d.x,-4,1e-14,"minimum image x"); near(d.y,3,1e-14,"minimum image y");
    near(d.z,-2.9,1e-14,"minimum image z");
    const wl::Box slab{{10,8,0},true}; const auto slab_d=wl::minimum_image({6,-5,31},slab);
    near(slab_d.x,-4,1e-14,"slab minimum image x");
    near(slab_d.y,3,1e-14,"slab minimum image y");
    near(slab_d.z,31,1e-14,"slab open z direction");
}

void test_csv_geometry_and_ini() {
    const auto directory=std::filesystem::temp_directory_path()/"wl_csv_ini_test";
    std::filesystem::create_directories(directory);
    const auto csv=directory/"system.csv"; const auto ini=directory/"run.ini";
    { std::ofstream out(csv); out<<"x,y,z,mx,my,mz\n0,0,0,0,0,1\n1,0,0,0,0,1\n"; }
    { std::ofstream out(ini); out<<"[geometry]\nfile=system.csv\nperiodic=false\n"
        <<"[energy]\nminimum=-4\nmaximum=4\nbin_width=0.25\n"
        <<"[parallel]\nexchange_interval_mcs=1.5\n"
        <<"[adaptive_windows]\nenabled=true\niterations=3\npilot_mcs=12\nsmoothing_width=0.75\n"
        <<"minimum_width=1.25\ndiffusivity_floor_fraction=0.08\ncurvature_weight=0.4\n"
        <<"round_trip_target=3\nmaximum_round_trip_penalty=2.5\nround_trip_margin_fraction=0.15\n"
        <<"[wl]\ncheck_interval_mcs=2.5\nforce_accept_after_mcs=3.5\ninverse_time=false\n"
        <<"[initialization]\nmax_attempts=123\ntarget_fraction=0.4\ntemperature_fraction=0.07\n"
        <<"stall_attempts_per_spin=17\ntemperature_multiplier=3\nmax_temperature_fraction=0.4\n"
        <<"[run]\nseed=17\nmax_mcs=4.5\ncheckpoint_interval_mcs=5.5\nprogress=2\n"
        <<"temperatures=0.5, 1.5\n[output]\nprefix=from_ini\n"; }
    const auto geometry=wl::Geometry::load_csv(csv.string(),{{2,2,2},false});
    require(geometry.size()==2,"CSV geometry size"); near(geometry.axes[1].z,1,1e-14,"CSV moment axis");
    std::vector<std::string> arguments{"test","--config",ini.string(),"--seed","99"};
    std::vector<char*> argv; for(auto& argument:arguments) argv.push_back(argument.data());
    auto config=wl::parse_arguments(static_cast<int>(argv.size()),argv.data());
    config.resolve_mcs(geometry.size());
    require(config.geometry_file==csv.lexically_normal().string(),"INI-relative CSV path");
    require(config.seed==99&&config.seed_explicit&&config.energy_grid_explicit,
            "CLI override and INI energy grid");
    std::vector<std::string> no_seed_arguments{"test"};
    std::vector<char*> no_seed_argv; for(auto& argument:no_seed_arguments) no_seed_argv.push_back(argument.data());
    require(!wl::parse_arguments(1,no_seed_argv.data()).seed_explicit,"omitted seed detection");
    require(config.exchange_interval_attempts==3&&config.wl.check_interval_attempts==5,
            "MCS exchange and check interval resolution");
    require(config.wl.force_accept_after_attempts==7&&config.max_attempts==9&&
            config.checkpoint_interval_attempts==11,"MCS run interval resolution");
    near(config.progress_interval_seconds,2.0,1e-14,"INI progress interval");
    require(!config.wl.inverse_time_enabled,"INI inverse-time switch");
    require(config.wl.initialization_max_attempts==123,"INI initialization attempts");
    near(config.wl.initialization_target_fraction,0.4,1e-14,"INI initialization target");
    near(config.wl.initialization_temperature_fraction,0.07,1e-14,
         "INI initialization temperature");
    require(config.wl.initialization_stall_attempts_per_spin==17,
            "INI initialization stall interval");
    near(config.wl.initialization_temperature_multiplier,3.0,1e-14,
         "INI initialization temperature multiplier");
    near(config.wl.initialization_max_temperature_fraction,0.4,1e-14,
         "INI initialization maximum temperature");
    require(config.adaptive_windows.enabled&&config.adaptive_windows.iterations==3,
            "INI adaptive-window switch and iterations");
    near(config.adaptive_windows.pilot_mcs,12.0,1e-14,"INI adaptive pilot length");
    near(config.adaptive_windows.smoothing_width,0.75,1e-14,"INI adaptive smoothing energy");
    near(config.adaptive_windows.minimum_width,1.25,1e-14,"INI adaptive minimum energy width");
    near(config.adaptive_windows.diffusivity_floor_fraction,0.08,1e-14,
         "INI adaptive diffusivity floor");
    near(config.adaptive_windows.curvature_weight,0.4,1e-14,"INI adaptive curvature weight");
    near(config.adaptive_windows.round_trip_target,3.0,1e-14,"INI adaptive round-trip target");
    near(config.adaptive_windows.maximum_round_trip_penalty,2.5,1e-14,
         "INI adaptive round-trip penalty");
    near(config.wl.round_trip_margin_fraction,0.15,1e-14,
         "INI adaptive round-trip margin");
    std::vector<std::string> progress_arguments{"test","--progress","3","--inverse-time","true"};
    std::vector<char*> progress_argv; for(auto& argument:progress_arguments) progress_argv.push_back(argument.data());
    const auto progress=wl::parse_arguments(static_cast<int>(progress_argv.size()),progress_argv.data());
    near(progress.progress_interval_seconds,3.0,1e-14,"CLI progress interval");
    require(progress.wl.inverse_time_enabled,"CLI inverse-time switch");
    std::vector<std::string> no_progress_arguments{"test","--progress","0"};
    std::vector<char*> no_progress_argv; for(auto& argument:no_progress_arguments) no_progress_argv.push_back(argument.data());
    require(wl::parse_arguments(static_cast<int>(no_progress_argv.size()),no_progress_argv.data()).progress_interval_seconds==0.0,
            "zero progress interval disables output");
    require(!config.uses_legacy_attempt_units(),"MCS options must not be marked legacy");
    std::vector<std::string> legacy_arguments{"test","--max-attempts","9"};
    std::vector<char*> legacy_argv; for(auto& argument:legacy_arguments) legacy_argv.push_back(argument.data());
    auto legacy=wl::parse_arguments(static_cast<int>(legacy_argv.size()),legacy_argv.data()); legacy.resolve_mcs(2);
    require(legacy.max_attempts==9&&legacy.max_mcs==4.5&&legacy.uses_legacy_attempt_units(),"legacy attempts compatibility");
    require(config.temperatures.size()==2&&config.output_prefix=="from_ini","INI lists and output");
    std::filesystem::remove_all(directory);
}

void test_backends_and_incremental_fields() {
    const auto geometry=wl::Geometry::simple_cubic(3,2,1,1.0,{0,0,1},false);
    wl::DenseCouplings dense(geometry,0.7); wl::CsrCouplings csr(geometry,0.7,100.0);
    for(std::size_t i=0;i<geometry.size();++i) for(std::size_t j=0;j<geometry.size();++j)
        near(dense.at(i,j),csr.at(i,j),1e-14,"dense/csr mismatch");
    std::vector<std::int8_t> spins{1,-1,1,1,-1,-1}; auto fields=wl::local_fields(dense,spins);
    const auto initial=wl::total_energy(dense,spins);
    for(std::size_t i=0;i<spins.size();++i) {
        const auto old=spins[i]; const auto delta=wl::flip_delta(i,spins,fields);
        dense.add_flip_delta(i,old,fields); spins[i]=-old;
        near(initial+delta,wl::total_energy(dense,spins),1e-12,"flip delta mismatch");
        const auto exact_fields=wl::local_fields(dense,spins);
        for(std::size_t j=0;j<spins.size();++j) near(fields[j],exact_fields[j],1e-12,"incremental field mismatch");
        dense.add_flip_delta(i,spins[i],fields); spins[i]=old;
    }
}

void test_energy_grid_and_windows() {
    const wl::EnergyGrid grid{-2,2,0.5}; require(grid.bins()==8,"grid bin count");
    require(grid.index(-2)==0&&grid.index(2)==7&&!grid.index(2.1),"grid boundary mapping");
    const auto windows=wl::partition_windows(100,4,0.75); require(windows.size()==4,"window count");
    require(windows.front().begin==0&&windows.back().end==100,"window coverage");
    for(std::size_t i=1;i<windows.size();++i) require(windows[i].begin<windows[i-1].end,"window overlap");
}

void test_target_metropolis_initialization() {
    const auto geometry=wl::Geometry::simple_cubic(2,1,1,1.0,{1,0,0},false);
    auto couplings=std::make_shared<wl::DenseCouplings>(geometry,1.0);
    const wl::EnergyGrid grid{-2.5,3.5,1.0};
    const wl::EnergyWindow window{0,grid.bins()};
    wl::WlParameters parameters{0.8,1,1e-8,10};
    parameters.initialization_max_attempts=100;

    const std::vector<std::int8_t> aligned(2,1);
    const auto edge_energy=wl::total_energy(*couplings,aligned);
    const auto target=0.5*(grid.minimum+grid.maximum);
    const auto target_half_width=0.5*parameters.initialization_target_fraction*
                                 (grid.maximum-grid.minimum);
    require(std::abs(edge_energy-target)>target_half_width,
            "test initial state must be outside the target band");

    wl::WangLandauWalker first(9,couplings,grid,window,parameters,123);
    wl::WangLandauWalker second(9,couplings,grid,window,parameters,123);
    require(std::abs(first.energy()-target)<=target_half_width,
            "target Metropolis must initialize inside the central band");
    require(first.snapshot().spins==second.snapshot().spins&&first.energy()==second.energy(),
            "target Metropolis initialization must be reproducible");
    const auto edge_bin=grid.index(edge_energy);
    require(edge_bin&&first.histogram()[*edge_bin]==0&&first.active_mask()[*edge_bin]==0,
            "initial edge state must not contaminate WL statistics");
    require(std::accumulate(first.histogram().begin(),first.histogram().end(),
                            std::uint64_t{0})==1,
            "only the selected initial bin is registered");
    require(first.attempted()==0&&first.accepted()==0,
            "initialization proposals are not production attempts");

    auto failed_parameters=parameters;
    failed_parameters.initialization_max_attempts=10;
    failed_parameters.initialization_stall_attempts_per_spin=1;
    failed_parameters.initialization_temperature_fraction=0.05;
    failed_parameters.initialization_temperature_multiplier=2.0;
    failed_parameters.initialization_max_temperature_fraction=0.5;
    const wl::EnergyGrid failed_grid{-2.0,2.0,0.1};
    std::string failure;
    try {
        wl::WangLandauWalker failed(10,couplings,failed_grid,{20,30},failed_parameters,123);
    } catch(const std::runtime_error& error) {
        failure=error.what();
    }
    require(failure.find("target band")!=std::string::npos&&
            failure.find("energy window bins [20, 30)")!=std::string::npos&&
            failure.find("energy bounds [0, 1)")!=std::string::npos&&
            failure.find("after 10 attempts")!=std::string::npos&&
            failure.find("reached energy range")!=std::string::npos&&
            failure.find("closest energy")!=std::string::npos&&
            failure.find("adaptive search used 3 temperature increases")!=std::string::npos&&
            failure.find("and 1 random restarts")!=std::string::npos,
            "failed adaptive initialization must report window bounds and search activity");
}

void test_walker_and_checkpoint() {
    auto couplings=std::make_shared<wl::DenseCouplings>(wl::Geometry::simple_cubic(2,2,1),1.0);
    wl::WlParameters p{0.1,1,0.1,10}; wl::EnergyGrid grid{-20,20,0.25};
    wl::WangLandauWalker walker(7,couplings,grid,{0,grid.bins()},p,42);
    const auto before=std::accumulate(walker.histogram().begin(),walker.histogram().end(),std::uint64_t{0});
    walker.run_attempts(100);
    const auto after=std::accumulate(walker.histogram().begin(),walker.histogram().end(),std::uint64_t{0});
    require(after-before==100,"histogram must update after every proposal");
    const auto path=(std::filesystem::temp_directory_path()/"wl_test_checkpoint.bin").string();
    wl::save_checkpoint(path,walker.snapshot());
    wl::WangLandauWalker restored(7,couplings,grid,{0,grid.bins()},p,42); restored.restore(wl::load_checkpoint(path));
    require(restored.attempted()==walker.attempted(),"checkpoint attempts"); near(restored.energy(),walker.energy(),1e-14,"checkpoint energy");
    require(restored.forced_accepted()==walker.forced_accepted(),"checkpoint forced acceptances");
    std::filesystem::remove(path);
}

void test_forced_acceptance() {
    const auto geometry=wl::Geometry::simple_cubic(2,1,1,1.0,{0,0,1},false);
    auto couplings=std::make_shared<wl::DenseCouplings>(geometry,1.0);
    wl::WlParameters parameters{0.8,1,1e-8,100,3};
    wl::EnergyGrid grid{-2,2,0.5};
    wl::WangLandauWalker walker(4,couplings,grid,{0,grid.bins()},parameters,123);
    auto snapshot=walker.snapshot();
    const auto old_bin=grid.index(snapshot.energy);
    const auto proposed=snapshot.energy+wl::flip_delta(0,snapshot.spins,snapshot.fields);
    const auto new_bin=grid.index(proposed);
    require(old_bin&&new_bin&&*old_bin!=*new_bin,"forced acceptance test bins");
    snapshot.log_g[*old_bin]=0.0;
    snapshot.log_g[*new_bin]=1000.0;
    walker.restore(snapshot);
    walker.run_attempts(3);
    require(walker.accepted()==0&&walker.forced_accepted()==0,
            "force threshold must allow N rejected attempts");
    require(walker.last_accepted_attempt()==0,
            "rejected attempts must not reset stuck age");
    require(walker.attempt_flip(),"first valid proposal after threshold must be forced");
    require(walker.accepted()==1&&walker.forced_accepted()==1,
            "forced acceptance counters");
    require(walker.last_accepted_attempt()==4,"forced acceptance resets stuck age");
}

void test_refinement_transition() {
    auto couplings=std::make_shared<wl::DenseCouplings>(wl::Geometry::simple_cubic(1,1,1),1.0);
    wl::WlParameters p{0.1,1,0.01,1}; wl::EnergyGrid grid{-1,1,0.5};
    wl::WangLandauWalker walker(1,couplings,grid,{0,grid.bins()},p,9);
    for(int iteration=0;iteration<10&&walker.stage()==wl::RefinementStage::wang_landau;++iteration) {
        walker.run_attempts(10); require(walker.ready_for_iteration(),"single-bin histogram coverage");
        walker.begin_next_iteration();
    }
    require(walker.stage()!=wl::RefinementStage::wang_landau,"WL must enter inverse-time refinement");

    auto traditional_parameters=p;
    traditional_parameters.inverse_time_enabled=false;
    wl::WangLandauWalker traditional(2,couplings,grid,{0,grid.bins()},traditional_parameters,9);
    for(int iteration=0;iteration<10&&traditional.stage()==wl::RefinementStage::wang_landau;++iteration) {
        traditional.run_attempts(10);
        require(traditional.ready_for_iteration(),"traditional WL single-bin flatness");
        traditional.begin_next_iteration();
        require(traditional.stage()!=wl::RefinementStage::inverse_time,"disabled inverse-time transition");
    }
    require(traditional.stage()==wl::RefinementStage::frozen,"traditional WL must stop at final factor");
}

void test_unlimited_max_attempts() {
    int argc=1;
    char program[]="wl_tests";
    char* arguments[]{program,nullptr};
    char** argv=arguments;
    wl::ParallelContext context(argc,argv);
    require(context.broadcast_seed(123456789ULL)==123456789ULL,
            "serial seed broadcast");
    auto couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    wl::RunConfig config;
    config.nx=1; config.ny=1; config.nz=1;
    config.grid={-1,1,0.5}; config.energy_grid_explicit=true;
    config.windows=1; config.walkers_per_rank=1;
    config.exchange_interval_attempts=1; config.max_attempts=0;
    config.wl={0.1,1,0.5,1};
    const auto result=wl::run_rewl(context,couplings,config);
    require(result.converged,"max_attempts=0 must run until convergence");
    require(result.attempted>0,"unlimited run must perform attempts");

    const auto nonzero_checkpoint=
        (std::filesystem::temp_directory_path()/"wl_global_nonzero_flatness.bin").string();
    wl::WangLandauWalker checkpoint_walker(
        0,couplings,config.grid,{0,config.grid.bins()},config.wl,config.seed);
    auto checkpoint_snapshot=checkpoint_walker.snapshot();
    const auto current_bin=*checkpoint_walker.energy_bin();
    const auto stale_bin=current_bin==0?1:0;
    checkpoint_snapshot.active[stale_bin]=1;
    checkpoint_snapshot.histogram[stale_bin]=0;
    wl::save_checkpoint(nonzero_checkpoint,checkpoint_snapshot);
    auto nonzero=config;
    nonzero.max_attempts=5;
    nonzero.checkpoint_path=nonzero_checkpoint;
    nonzero.checkpoint_interval_attempts=100;
    const auto nonzero_result=wl::run_rewl(context,couplings,nonzero);
    require(!nonzero_result.converged,"discovered zero H_total bin must block global flatness");
    require(nonzero_result.attempted==5,"discovered zero H_total attempt limit");
    std::filesystem::remove(nonzero_checkpoint);

    auto scaled=config;
    scaled.walkers_per_rank=2;
    scaled.max_attempts=1;
    scaled.wl={0.1,3,0.5,1};
    const auto scaled_result=wl::run_rewl(context,couplings,scaled);
    require(!scaled_result.converged,"global minimum visits must scale with walker count");
    require(scaled_result.attempted==2,"global minimum visits attempted count");
    require(result.walker_statistics.size()==1,"unlimited run walker statistics");

    auto blocked_couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(2,1,1,1.0,{0,0,1},false),1.0);
    auto blocked=config;
    blocked.nx=2;
    blocked.grid={0.5,1.5,0.25};
    blocked.max_attempts=5;
    blocked.wl={0.8,100,1e-8,1};
    blocked.wl.inverse_time_enabled=false;
    const auto blocked_result=wl::run_rewl(context,blocked_couplings,blocked);
    require(!blocked_result.converged,"blocked walker must stop unconverged");
    require(blocked_result.walker_statistics.size()==1,"blocked walker statistics");
    const auto& statistic=blocked_result.walker_statistics.front();
    require(statistic.attempted==5&&statistic.accepted==0,"blocked walker acceptance");
    require(statistic.attempts_since_last_accepted==5,"blocked walker last accepted flip age");
    near(statistic.factor,1.0,1e-14,"blocked walker factor");
    require(statistic.active_bins==1,"blocked walker active bins");
    require(statistic.minimum_histogram==6,"blocked walker min histogram");
    near(statistic.mean_histogram,6.0,1e-14,"blocked walker mean histogram");
    near(statistic.min_over_mean,1.0,1e-14,"blocked walker histogram ratio");
    near(statistic.energy,1.0,1e-14,"blocked walker final energy");

    const auto stat_path=std::filesystem::temp_directory_path()/"wl_workers_stat_test.csv";
    wl::write_workers_stat_csv(stat_path.string(),blocked_result.walker_statistics,2);
    std::ifstream stat_file(stat_path);
    std::string header;
    std::string row;
    std::getline(stat_file,header);
    std::getline(stat_file,row);
    require(header=="mpi_rank,window,walker_id,attempted_flips,attempted_mcs,accepted,forced_accepted,accepted_percent,attempts_since_last_accepted,mcs_since_last_accepted,energy,factor,active_bins,min_h,mean_h,min_over_mean,round_trips",
            "walker statistics CSV header");
    require(row=="0,0,0,5,2.5,0,0,0,5,2.5,1,1,1,6,6,1,0","walker statistics CSV row");
    stat_file.close();
    std::filesystem::remove(stat_path);

    const auto exact_geometry=wl::Geometry::simple_cubic(2,2,1,1.0,{0,0,1},false);
    auto exact_couplings=std::make_shared<wl::DenseCouplings>(exact_geometry,1.0);
    wl::RunConfig exact_config;
    exact_config.nx=2; exact_config.ny=2; exact_config.nz=1; exact_config.periodic=false;
    exact_config.grid={-20,20,0.25}; exact_config.energy_grid_explicit=true;
    exact_config.windows=1; exact_config.walkers_per_rank=4;
    exact_config.seed=1; exact_config.seed_explicit=true;
    exact_config.exchange_interval_attempts=1000; exact_config.max_attempts=0;
    exact_config.wl={0.8,100,0.001,1000,0,true};
    const auto classic=wl::run_rewl(context,exact_couplings,exact_config);
    require(classic.converged,"classic REWL exact regression must converge");
    const auto exact_fragment=wl::exact_enumeration(*exact_couplings,exact_config.grid);
    const auto& estimate=classic.fragments.front();
    double offset=0.0; std::size_t occupied=0;
    for(std::size_t i=0;i<exact_fragment.histogram.size();++i)
        if(exact_fragment.histogram[i]!=0) {
            require(estimate.valid[i]!=0,"classic REWL must retain every exact occupied bin");
            offset+=estimate.log_g[i]-exact_fragment.log_g[i]; ++occupied;
        }
    offset/=static_cast<double>(occupied);
    double squared_error=0.0;
    for(std::size_t i=0;i<exact_fragment.histogram.size();++i)
        if(exact_fragment.histogram[i]!=0) {
            const auto error=estimate.log_g[i]-exact_fragment.log_g[i]-offset;
            squared_error+=error*error;
        }
    require(std::sqrt(squared_error/static_cast<double>(occupied))<=0.05,
            "classic REWL exact-DOS RMSE");
    const auto repeated=wl::run_rewl(context,exact_couplings,exact_config);
    require(repeated.converged&&repeated.fragments.front().valid==estimate.valid,
            "classic REWL repeat convergence and mask");
    for(std::size_t i=0;i<estimate.log_g.size();++i) if(estimate.valid[i]!=0)
        require(repeated.fragments.front().log_g[i]==estimate.log_g[i],
                "classic REWL fixed-seed reproducibility");
}

void test_exact_enumeration_and_thermo() {
    const auto geometry=wl::Geometry::simple_cubic(2,2,1,1.0,{0,0,1},false);
    wl::DenseCouplings couplings(geometry,1.0); wl::EnergyGrid grid{-20,20,0.25};
    const auto exact=wl::exact_enumeration(couplings,grid);
    require(std::accumulate(exact.histogram.begin(),exact.histogram.end(),std::uint64_t{0})==16,"exact state count");
    auto dos=wl::stitch_dos(grid,std::span(&exact,1),true,geometry.size());
    const std::vector<double> temperatures{1.0,2.0}; const auto thermo=wl::thermodynamics(dos,temperatures);
    require(thermo.size()==2&&std::isfinite(thermo[0].heat_capacity),"finite thermodynamics");
    auto incomplete=exact;
    incomplete.valid[0]=0;
    incomplete.log_g[0]=std::numeric_limits<double>::quiet_NaN();
    bool rejected=false;
    try { (void)wl::stitch_dos(grid,std::span(&incomplete,1),true,geometry.size()); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"complete normalization rejects invalid bins");
    const auto relative=wl::stitch_dos(grid,std::span(&incomplete,1),false,geometry.size());
    require(relative.valid[0]==0&&std::isfinite(wl::thermodynamics(relative,temperatures)[0].heat_capacity),
            "relative thermodynamics ignores invalid bins");
}

void test_adaptive_energy_windows() {
    wl::EnergyGrid grid{-10.0,10.0,0.5};
    const auto initial=wl::partition_windows(grid.bins(),4,0.5);
    std::vector<wl::DosFragment> fragments;
    std::vector<wl::WindowSamplingStatistics> sampling;
    for(const auto window:initial) {
        wl::DosFragment fragment{window,std::vector<double>(grid.bins(),0.0),
            std::vector<std::uint64_t>(grid.bins(),0),std::vector<double>(grid.bins(),0.0),
            std::vector<std::uint8_t>(grid.bins(),0)};
        wl::WindowSamplingStatistics statistics{window,std::vector<double>(grid.bins(),0.0),
            std::vector<std::uint64_t>(grid.bins(),0),0,1};
        for(std::size_t i=window.begin;i<window.end;++i) {
            fragment.valid[i]=1;
            statistics.displacement_samples[i]=100;
            const auto diffusivity=grid.center(i)<-5.0?0.01:1.0;
            statistics.squared_energy_displacement[i]=100.0*diffusivity;
        }
        fragments.push_back(std::move(fragment));
        sampling.push_back(std::move(statistics));
    }
    wl::AdaptiveWindowParameters parameters;
    parameters.smoothing_width=0.5;
    parameters.minimum_width=1.0;
    parameters.curvature_weight=0.0;
    parameters.round_trip_target=0.0;
    const auto adapted=wl::adapt_energy_windows(grid,fragments,sampling,4,0.5,parameters);
    require(adapted.size()==4&&adapted.front().begin==0&&adapted.back().end==grid.bins(),
            "adaptive windows cover the energy range");
    for(std::size_t i=1;i<adapted.size();++i)
        require(adapted[i].begin<adapted[i-1].end&&adapted[i].begin>adapted[i-1].begin,
                "adaptive windows remain ordered and overlapping");
    const auto first_width=static_cast<double>(adapted.front().end-adapted.front().begin)*grid.width;
    const auto last_width=static_cast<double>(adapted.back().end-adapted.back().begin)*grid.width;
    require(first_width<last_width,"low diffusivity must produce a narrower energy window");

    wl::EnergyGrid fine_grid{-10.0,10.0,0.25};
    const auto fine_initial=wl::partition_windows(fine_grid.bins(),4,0.5);
    std::vector<wl::DosFragment> fine_fragments;
    std::vector<wl::WindowSamplingStatistics> fine_sampling;
    for(const auto window:fine_initial) {
        wl::DosFragment fragment{window,std::vector<double>(fine_grid.bins(),0.0),
            std::vector<std::uint64_t>(fine_grid.bins(),0),
            std::vector<double>(fine_grid.bins(),0.0),
            std::vector<std::uint8_t>(fine_grid.bins(),0)};
        wl::WindowSamplingStatistics statistics{window,
            std::vector<double>(fine_grid.bins(),0.0),
            std::vector<std::uint64_t>(fine_grid.bins(),0),0,1};
        for(std::size_t i=window.begin;i<window.end;++i) {
            fragment.valid[i]=1;
            statistics.displacement_samples[i]=100;
            statistics.squared_energy_displacement[i]=
                100.0*(fine_grid.center(i)<-5.0?0.01:1.0);
        }
        fine_fragments.push_back(std::move(fragment));
        fine_sampling.push_back(std::move(statistics));
    }
    const auto fine_adapted=wl::adapt_energy_windows(
        fine_grid,fine_fragments,fine_sampling,4,0.5,parameters);
    const auto coarse_first_upper=grid.minimum+static_cast<double>(adapted.front().end)*grid.width;
    const auto fine_first_upper=fine_grid.minimum+
        static_cast<double>(fine_adapted.front().end)*fine_grid.width;
    require(std::abs(coarse_first_upper-fine_first_upper)<=grid.width,
            "adaptive boundaries are stable under energy-grid refinement");

    auto couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(2,1,1,1.0,{0,0,1},false),1.0);
    wl::WlParameters walker_parameters{0.1,1,0.01,10};
    walker_parameters.collect_window_statistics=true;
    walker_parameters.round_trip_margin_fraction=0.1;
    wl::EnergyGrid walker_grid{-1.25,1.25,0.25};
    wl::WangLandauWalker walker(91,couplings,walker_grid,{0,walker_grid.bins()},
                                walker_parameters,123,{1,1});
    walker.run_attempts(5'000);
    const auto recorded=std::accumulate(walker.displacement_samples().begin(),
        walker.displacement_samples().end(),std::uint64_t{0});
    const auto motion=std::accumulate(walker.squared_energy_displacement().begin(),
        walker.squared_energy_displacement().end(),0.0);
    require(recorded==5'000&&motion>0.0,"energy diffusivity statistics are collected");
    require(walker.round_trips()>0,"complete low-high-low energy trips are counted");
    require(walker.representatives().size()==9,
            "adaptive pilot keeps a fixed physical-energy configuration bank");
    for(const auto& representative:walker.representatives())
        require(representative.spins.size()==2&&walker_grid.index(representative.energy).has_value(),
                "adaptive representative contains a valid reusable spin configuration");
    for(const auto& representative:walker.representatives())
        require(walker.representatives().front().energy<=representative.energy,
                "adaptive representative bank preserves its minimum energy");

    const std::vector<wl::EnergyWindow> seed_windows{{0,6},{4,10}};
    const std::vector<wl::EnergyRepresentative> seed_representatives{
        {-1.0,{-1,-1}},{-0.25,{-1,1}},{0.75,{1,-1}}};
    std::size_t missing=0,external_warm_starts=0;
    const auto starts=wl::select_adaptive_initial_configurations(
        walker_grid,seed_windows,seed_representatives,1,2,missing,external_warm_starts);
    require(starts.size()==4&&starts[0]==seed_representatives[0].spins&&
            starts[1]==seed_representatives[0].spins,
            "all walkers in the lower edge window start from the minimum-energy configuration");
    require(missing==0&&external_warm_starts==0,
            "in-window minimum seeds are counted as regular adaptive starts");
    const auto distributed_starts=wl::select_adaptive_initial_configurations(
        walker_grid,seed_windows,seed_representatives,4,2,missing,external_warm_starts);
    require(distributed_starts.size()==8&&
            std::all_of(distributed_starts.begin(),distributed_starts.begin()+4,
                [&](const auto& spins) { return spins==seed_representatives[0].spins; }),
            "every MPI owner of the lower edge window receives the minimum-energy seed");

    wl::WlParameters warm_parameters;
    warm_parameters.initialization_max_attempts=1;
    warm_parameters.initialization_target_fraction=0.1;
    wl::WangLandauWalker warm_start(92,couplings,walker_grid,{0,3},warm_parameters,321,{1,1});
    require(warm_start.energy_bin()&&warm_start.window().contains(*warm_start.energy_bin()),
            "external adaptive warm start only needs to enter its new energy window");
}

void test_histogram_statistics() {
    auto couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    wl::WlParameters parameters{0.8,1,1e-8,1};
    wl::EnergyGrid grid{-1,1,0.5};
    wl::WangLandauWalker walker(3,couplings,grid,{0,grid.bins()},parameters,11);
    auto snapshot=walker.snapshot();
    std::fill(snapshot.active.begin(),snapshot.active.end(),0);
    std::fill(snapshot.histogram.begin(),snapshot.histogram.end(),0);
    snapshot.active[0]=1; snapshot.histogram[0]=80;
    snapshot.active[1]=1; snapshot.histogram[1]=120;
    walker.restore(snapshot);
    auto statistics=walker.histogram_statistics();
    require(statistics.active_bins==2,"active histogram bins");
    require(statistics.covered_bins==2,"covered histogram bins");
    require(statistics.minimum==80,"minimum active histogram");
    near(statistics.mean,100.0,1e-14,"mean active histogram");
    near(statistics.min_over_mean,0.8,1e-14,"histogram min over mean");
    near(statistics.coverage,1.0,1e-14,"complete active-bin coverage");
    require(walker.flat(),"inactive zero bins must not affect flatness");
    require(walker.covered(),"inactive zero bins must not affect coverage");

    snapshot=walker.snapshot();
    snapshot.active[2]=1;
    snapshot.histogram[2]=0;
    walker.restore(snapshot);
    statistics=walker.histogram_statistics();
    require(statistics.active_bins==3&&statistics.minimum==0,"active zero histogram bin");
    require(statistics.covered_bins==2,"active zero bin is not covered");
    near(statistics.mean,200.0/3.0,1e-14,"mean with active zero bin");
    near(statistics.min_over_mean,0.0,1e-14,"zero histogram ratio");
    near(statistics.coverage,2.0/3.0,1e-14,"partial active-bin coverage");
    require(!walker.flat(),"active zero bin must block flatness");
    require(!walker.covered(),"active zero bin must block coverage");

    snapshot=walker.snapshot();
    snapshot.attempted=1000;
    snapshot.active[2]=0;
    walker.restore(snapshot);
    require(walker.ready_for_iteration(),"local histogram must control iteration readiness");
    walker.begin_next_iteration();
    statistics=walker.histogram_statistics();
    require(statistics.active_bins==2&&statistics.minimum==0,"local active mask is cumulative");
    near(walker.factor(),0.5,1e-14,"local flatness iteration factor");

    snapshot=walker.snapshot();
    snapshot.stage=wl::RefinementStage::wang_landau;
    snapshot.factor=1.0;
    snapshot.active[0]=snapshot.active[1]=1;
    snapshot.histogram[0]=1;
    snapshot.histogram[1]=1000;
    walker.restore(snapshot);
    require(!walker.flat()&&walker.covered()&&walker.ready_for_iteration(),
            "1/t initial stage uses complete coverage instead of histogram flatness");

    snapshot=walker.snapshot();
    snapshot.stage=wl::RefinementStage::inverse_time;
    snapshot.factor=0.002;
    walker.restore(snapshot);
    walker.attempt_flip();
    near(walker.factor(),2.0/1001.0,1e-14,"restored active-bin cache");
    walker.attempt_flip();
    near(walker.factor(),3.0/1002.0,1e-14,"new active bin increments cache");

    wl::WlParameters local_clock_parameters{0.99,100,1e-8,1};
    wl::WangLandauWalker early(31,couplings,grid,{0,grid.bins()},local_clock_parameters,19);
    wl::WangLandauWalker late(32,couplings,grid,{0,grid.bins()},local_clock_parameters,19);
    auto early_snapshot=early.snapshot();
    auto late_snapshot=late.snapshot();
    for(auto* state:{&early_snapshot,&late_snapshot}) {
        std::fill(state->active.begin(),state->active.end(),0);
        std::fill(state->histogram.begin(),state->histogram.end(),0);
        state->active[0]=state->active[1]=1;
        state->histogram[0]=1;
        state->histogram[1]=1000;
    }
    early_snapshot.attempted=100;
    early_snapshot.factor=0.02;
    late_snapshot.attempted=400;
    late_snapshot.factor=0.008;
    early.restore(early_snapshot);
    late.restore(late_snapshot);
    require(early.ready_for_iteration()&&late.ready_for_iteration(),
            "covered walkers are independently ready for 1/t");
    early.begin_next_iteration();
    late.begin_next_iteration();
    require(early.stage()==wl::RefinementStage::inverse_time&&
            late.stage()==wl::RefinementStage::inverse_time,
            "covered walkers enter inverse-time independently");
    near(early.factor(),2.0/100.0,1e-14,"early walker local 1/t clock");
    near(late.factor(),2.0/400.0,1e-14,"late walker local 1/t clock");
}

void test_classic_rewl_independence_and_summary() {
    auto couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    wl::WlParameters parameters{0.8,1,0.01,1,0,false};
    wl::EnergyGrid grid{-1,1,0.5};
    const wl::EnergyWindow window{0,grid.bins()};
    wl::WangLandauWalker first(10,couplings,grid,window,parameters,17);
    wl::WangLandauWalker second(11,couplings,grid,window,parameters,17);

    auto first_snapshot=first.snapshot();
    auto second_snapshot=second.snapshot();
    std::fill(first_snapshot.active.begin(),first_snapshot.active.end(),0);
    std::fill(second_snapshot.active.begin(),second_snapshot.active.end(),0);
    std::fill(first_snapshot.histogram.begin(),first_snapshot.histogram.end(),0);
    std::fill(second_snapshot.histogram.begin(),second_snapshot.histogram.end(),0);
    first_snapshot.active[0]=first_snapshot.active[1]=1;
    second_snapshot.active[0]=second_snapshot.active[1]=1;
    first_snapshot.histogram[0]=200;
    second_snapshot.histogram[1]=200;
    first.restore(first_snapshot); second.restore(second_snapshot);
    require(!first.flat()&&!second.flat(),"pooled flatness must not advance local walkers");

    first_snapshot=first.snapshot();
    first_snapshot.histogram[0]=100; first_snapshot.histogram[1]=100;
    first_snapshot.log_g[0]=2.0; first_snapshot.log_g[1]=5.0;
    first.restore(first_snapshot);
    const auto first_log_g=first.log_g();
    const auto second_before=second.snapshot();
    first.begin_next_iteration();
    require(first.log_g()==first_log_g,"local iteration must preserve its DOS");
    require(first.factor()==0.5,"local iteration factor reduction");
    require(std::all_of(first.histogram().begin(),first.histogram().end(),
                        [](const auto value){return value==0;}),"local histogram reset");
    const auto second_after=second.snapshot();
    require(second_after.log_g==second_before.log_g&&
            second_after.histogram==second_before.histogram&&
            second_after.active==second_before.active&&
            second_after.factor==second_before.factor&&second_after.stage==second_before.stage,
            "one walker iteration must not mutate another walker");

    first_snapshot=first.snapshot(); second_snapshot=second.snapshot();
    std::fill(first_snapshot.active.begin(),first_snapshot.active.end(),0);
    std::fill(second_snapshot.active.begin(),second_snapshot.active.end(),0);
    first_snapshot.active[0]=first_snapshot.active[1]=first_snapshot.active[2]=1;
    second_snapshot.active[1]=second_snapshot.active[2]=second_snapshot.active[3]=1;
    first_snapshot.log_g[1]=10.0; first_snapshot.log_g[2]=14.0;
    second_snapshot.log_g[1]=-3.0; second_snapshot.log_g[2]=3.0;
    first_snapshot.histogram[2]=7; second_snapshot.histogram[2]=9;
    first.restore(first_snapshot); second.restore(second_snapshot);
    const wl::WangLandauWalker* walkers[]{&first,&second};
    const auto fragment=wl::summarize_walkers(window,walkers,grid.bins());
    require(fragment.valid[0]==0&&fragment.valid[1]==1&&fragment.valid[2]==1&&
            fragment.valid[3]==0,"summary uses active-mask intersection");
    near(fragment.log_g[1],0.0,1e-14,"summary reference bin");
    near(fragment.log_g[2],5.0,1e-14,"summary aligned mean");
    near(fragment.standard_error[2],1.0,1e-14,"summary analytic SEM");
    require(fragment.histogram[2]==16,"summary histogram sum");
    require(std::isnan(fragment.log_g[0])&&std::isnan(fragment.standard_error[0]),
            "summary invalid bins are NaN");

    const wl::WangLandauWalker* single[]{&first};
    const auto single_fragment=wl::summarize_walkers(window,single,grid.bins());
    require(std::isnan(single_fragment.standard_error[1]),"single-walker SEM is undefined");

    first_snapshot=first.snapshot(); second_snapshot=second.snapshot();
    first_snapshot.stage=wl::RefinementStage::frozen; first_snapshot.factor=0.005;
    second_snapshot.stage=wl::RefinementStage::wang_landau; second_snapshot.factor=0.5;
    first.restore(first_snapshot); second.restore(second_snapshot);
    require(std::isfinite(first.exchange_log_probability(second)),
            "replica exchange supports different refinement stages");
    const auto first_estimator=first.snapshot(); const auto second_estimator=second.snapshot();
    first.swap_configuration(second);
    require(first.log_g()==first_estimator.log_g&&first.factor()==first_estimator.factor&&
            first.stage()==first_estimator.stage&&second.log_g()==second_estimator.log_g&&
            second.factor()==second_estimator.factor&&second.stage()==second_estimator.stage,
            "replica exchange must preserve both DOS estimators and stages");

    second_snapshot=second.snapshot();
    std::fill(second_snapshot.active.begin(),second_snapshot.active.end(),0);
    second_snapshot.active[3]=1; second.restore(second_snapshot);
    bool rejected=false;
    try { (void)wl::summarize_walkers(window,walkers,grid.bins()); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"summary must reject walkers without a common active bin");
}
}

int main() {
    const std::vector<std::pair<const char*,std::function<void()>>> tests{
      {"pair_formula",test_pair_formula},{"minimum_image",test_minimum_image},
      {"target_metropolis_initialization",test_target_metropolis_initialization},
      {"csv_geometry_ini",test_csv_geometry_and_ini},
      {"backends_incremental",test_backends_and_incremental_fields},
      {"energy_grid_windows",test_energy_grid_and_windows},{"walker_checkpoint",test_walker_and_checkpoint},
      {"forced_acceptance",test_forced_acceptance},
      {"refinement_transition",test_refinement_transition},
      {"histogram_statistics",test_histogram_statistics},
      {"classic_rewl_independence_summary",test_classic_rewl_independence_and_summary},
      {"unlimited_max_attempts",test_unlimited_max_attempts},
      {"exact_thermo",test_exact_enumeration_and_thermo},
      {"adaptive_energy_windows",test_adaptive_energy_windows}};
    int failed=0;
    for(const auto& [name,test]:tests) try { test(); std::cout<<"PASS "<<name<<'\n'; }
      catch(const std::exception& e) { ++failed; std::cerr<<"FAIL "<<name<<": "<<e.what()<<'\n'; }
    return failed?1:0;
}

