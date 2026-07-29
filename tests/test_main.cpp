#include "wl/parallel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <iterator>
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
        <<"nalivaiko_mod=true\nreturn_mode=true\n"
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
    require(config.wl.nalivaiko_mod,"INI Nalivaiko modification switch");
    require(config.wl.return_mode,"INI return-mode switch");
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
    require(!progress.wl.nalivaiko_mod,"Nalivaiko modification default");
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
    const auto invalid_ini=directory/"invalid.ini";
    { std::ofstream out(invalid_ini); out<<"[wl]\nnalivaiko_mod=maybe\n"; }
    std::vector<std::string> invalid_arguments{"test","--config",invalid_ini.string()};
    std::vector<char*> invalid_argv;
    for(auto& argument:invalid_arguments) invalid_argv.push_back(argument.data());
    bool invalid_rejected=false;
    try { (void)wl::parse_arguments(static_cast<int>(invalid_argv.size()),invalid_argv.data()); }
    catch(const std::runtime_error&) { invalid_rejected=true; }
    require(invalid_rejected,"invalid Nalivaiko boolean must be rejected");
    const auto metadata_path=directory/"metadata.json";
    const wl::DenseCouplings metadata_couplings(geometry,1.0);
    wl::write_metadata_json(metadata_path.string(),config,metadata_couplings,
                            0,0,0,0,0,false,1,1);
    std::ifstream metadata_input(metadata_path);
    const std::string metadata((std::istreambuf_iterator<char>(metadata_input)),
                               std::istreambuf_iterator<char>());
    metadata_input.close();
    require(metadata.find("\"nalivaiko_mod\": true")!=std::string::npos&&
            metadata.find("\"return_mode\": true")!=std::string::npos&&
            metadata.find("\"refinement_active_scope\": \"current_iteration\"")!=
                std::string::npos,
            "Nalivaiko mode must be recorded in metadata");
    std::filesystem::remove_all(directory);
}

void test_backends_and_incremental_fields() {
    const auto geometry=wl::Geometry::simple_cubic(3,2,1,1.0,{0,0,1},false);
    wl::DenseCouplings dense(geometry,0.7); wl::CsrCouplings csr(geometry,0.7,100.0);
    for(std::size_t i=0;i<geometry.size();++i) for(std::size_t j=0;j<geometry.size();++j)
        near(dense.at(i,j),csr.at(i,j),1e-14,"dense/csr mismatch");
    std::vector<std::int8_t> spins{1,-1,1,1,-1,-1}; auto fields=wl::local_fields(dense,spins);
    const auto initial=wl::total_energy(dense,spins);
    const auto csr_fields=wl::local_fields(csr,spins);
    require(fields==csr_fields,"dense/CSR backend-specific fields must be bitwise equal");
    require(wl::energy_from_fields(spins,fields)==initial,
            "energy_from_fields must reuse the exact computed field");
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

void test_sparse_dos_csv_output() {
    const auto directory=std::filesystem::temp_directory_path()/"wl_sparse_dos_csv_test";
    std::filesystem::create_directories(directory);
    const auto read_lines=[](const std::filesystem::path& path) {
        std::ifstream input(path);
        std::vector<std::string> lines;
        for(std::string line;std::getline(input,line);) lines.push_back(std::move(line));
        return lines;
    };
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    const wl::EnergyGrid energy_grid{-2.0,2.0,1.0};
    const std::vector<double> log_g{nan,1.0,nan,-std::numeric_limits<double>::infinity()};
    const std::vector<std::uint64_t> histogram{0,5,0,0};
    const std::vector<double> standard_error{nan,0.25,nan,0.0};
    const std::vector<std::uint8_t> valid{0,1,0,1};
    const wl::DensityOfStates dos{energy_grid,log_g,histogram,standard_error,valid,{},false};
    const auto dos_path=directory/"dos.csv";
    wl::write_dos_csv(dos_path.string(),dos);
    auto lines=read_lines(dos_path);
    require(lines.size()==3&&lines[1].starts_with("1,")&&lines[2].starts_with("3,"),
            "DOS CSV must omit invalid bins without renumbering valid bins");
    require(lines[2].find(",-inf,0,0,1")!=std::string::npos,
            "known structural-zero DOS bins must remain in output");

    const wl::DosFragment fragment{{0,energy_grid.bins()},log_g,histogram,standard_error,
                                   valid,wl::DosGrid{energy_grid,std::nullopt},{},{}};
    const auto fragment_path=directory/"window.csv";
    wl::write_fragment_csv(fragment_path.string(),energy_grid,fragment,7);
    lines=read_lines(fragment_path);
    require(lines.size()==3&&lines[1].starts_with("7,1,")&&lines[2].starts_with("7,3,"),
            "window DOS CSV must omit invalid bins");

    const auto order=wl::WeightedOrderParameter::create({1.0},1.0);
    const wl::DosGrid joint_grid{{-1.0,1.0,1.0},order.grid};
    const auto cells=joint_grid.cells();
    std::vector<double> joint_log_g(cells,nan);
    std::vector<std::uint64_t> joint_histogram(cells,0);
    std::vector<double> joint_error(cells,nan);
    std::vector<std::uint8_t> joint_valid(cells,0);
    std::vector<std::uint32_t> contributors(cells,0);
    std::vector<std::int32_t> components(cells,-1);
    const auto first=joint_grid.flatten(0,1);
    const auto second=joint_grid.flatten(1,2);
    for(const auto cell:{first,second}) {
        joint_log_g[cell]=2.0;
        joint_histogram[cell]=3;
        joint_error[cell]=0.5;
        joint_valid[cell]=1;
        contributors[cell]=1;
        components[cell]=0;
    }
    const wl::JointDensityOfStates joint{joint_grid,order.normalization,joint_log_g,
        joint_histogram,joint_error,joint_valid,contributors,components,false};
    const auto joint_path=directory/"dos2d.csv";
    wl::write_joint_dos_csv(joint_path.string(),joint);
    lines=read_lines(joint_path);
    require(lines.size()==3&&lines[1].starts_with("0,1,")&&lines[2].starts_with("1,2,"),
            "joint DOS CSV must omit invalid cells without renumbering coordinates");

    const wl::DosFragment joint_fragment{{0,joint_grid.energy_bins()},joint_log_g,
        joint_histogram,joint_error,joint_valid,joint_grid,contributors,components};
    const auto joint_fragment_path=directory/"window_dos2d.csv";
    wl::write_joint_fragment_csv(joint_fragment_path.string(),joint_fragment,
                                 order.normalization,4);
    lines=read_lines(joint_fragment_path);
    require(lines.size()==3&&lines[1].starts_with("4,0,1,")&&
            lines[2].starts_with("4,1,2,"),"joint window DOS CSV must omit invalid cells");
    std::filesystem::remove_all(directory);
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

void test_return_mode() {
    const auto geometry=wl::Geometry::simple_cubic(2,1,1,1.0,{1,0,0},false);
    auto couplings=std::make_shared<wl::DenseCouplings>(geometry,1.0);
    const wl::EnergyGrid grid{-2.5,3.5,1.0};
    wl::WlParameters parameters{0.1,1,0.1,1};
    parameters.inverse_time_enabled=false;
    parameters.return_mode=true;
    parameters.initialization_max_attempts=100;
    const std::vector<std::int8_t> reference{1,1};
    const std::vector<std::int8_t> excited{1,-1};

    wl::WangLandauWalker lowest(201,couplings,grid,{0,grid.bins()},parameters,17,
                                excited,reference);
    require(lowest.ready_for_iteration(),"return-mode test iteration must be ready");
    const auto attempted_before=lowest.attempted();
    lowest.begin_next_iteration();
    require(std::equal(lowest.spins().begin(),lowest.spins().end(),reference.begin()),
            "return mode must restore the reference configuration in the lowest window");
    near(lowest.energy(),wl::total_energy(*couplings,reference),1e-14,
         "return mode reference energy");
    require(lowest.attempted()==attempted_before,
            "return and initialization search must not increment production attempts");
    require(std::accumulate(lowest.histogram().begin(),lowest.histogram().end(),
                            std::uint64_t{0})==1,
            "the returned initial state must seed the new WL histogram once");
    near(lowest.factor(),0.5,1e-14,"return mode factor halving");

    const auto excited_bin=grid.index(wl::total_energy(*couplings,excited));
    require(excited_bin.has_value(),"excited state must lie on the test grid");
    const wl::EnergyWindow upper{*excited_bin,*excited_bin+1};
    wl::WangLandauWalker upper_walker(202,couplings,grid,upper,parameters,19,
                                      excited,reference);
    require(upper_walker.ready_for_iteration(),"upper return-mode iteration must be ready");
    upper_walker.begin_next_iteration();
    require(upper_walker.energy_bin()&&upper.contains(*upper_walker.energy_bin()),
            "return mode must search from the reference state back into a higher window");
    require(upper_walker.attempted()==0,
            "higher-window return search must not increment production attempts");
    const auto exact_fields=wl::local_fields(*couplings,upper_walker.spins());
    require(std::equal(upper_walker.fields().begin(),upper_walker.fields().end(),
                       exact_fields.begin()),"return-mode search must preserve local fields");
    near(upper_walker.energy(),wl::energy_from_fields(upper_walker.spins(),exact_fields),
         1e-14,"return-mode search must preserve energy");

    auto disabled=parameters;
    disabled.return_mode=false;
    wl::WangLandauWalker ordinary(203,couplings,grid,{0,grid.bins()},disabled,17,excited,
                                  reference);
    ordinary.begin_next_iteration();
    require(std::equal(ordinary.spins().begin(),ordinary.spins().end(),excited.begin()),
            "disabled return mode must preserve the current configuration");

    auto one_spin_couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    const auto order_value=wl::WeightedOrderParameter::create({2.0},1.0);
    auto order=std::make_shared<const wl::WeightedOrderParameter>(order_value);
    const wl::DosGrid joint_grid{{-1.0,1.0,0.5},order->grid};
    const std::vector<std::int8_t> q_initial{-1};
    const std::vector<std::int8_t> q_reference{1};
    wl::WangLandauWalker joint(204,one_spin_couplings,joint_grid,
        {0,joint_grid.energy_bins()},parameters,23,order,q_initial,q_reference);
    auto joint_before=joint.snapshot();
    joint_before.attempted=10;
    joint_before.last_new_cell_attempt=0;
    wl::WangLandauWalker joint_restored(204,one_spin_couplings,joint_grid,
        {0,joint_grid.energy_bins()},parameters,23,order,q_initial,q_reference);
    joint_restored.restore(joint_before);
    joint_restored.begin_next_iteration();
    near(joint_restored.order_parameter(),2.0,1e-14,
         "joint return mode must restore the cached order parameter");
    require(joint_restored.energy_bin().has_value()&&
            joint_grid.index(joint_restored.energy(),joint_restored.order_parameter()).has_value(),
            "joint return mode must refresh the cached DOS cell after checkpoint restore");
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

void test_joint_dos_and_order_parameter() {
    const auto directory=std::filesystem::temp_directory_path()/"wl_joint_dos_test";
    std::filesystem::create_directories(directory);
    const auto csv=directory/"weighted.csv";
    { std::ofstream out(csv); out<<"x,y,z,mx,my,mz,q_weight\n"
        <<"0,0,0,0,0,1,1\n1,0,0,0,0,1,-1\n"; }
    const auto geometry=wl::Geometry::load_csv(csv.string(),{{0,0,0},false});
    require(geometry.order_weights==std::vector<double>({1.0,-1.0}),"CSV q_weight values");
    std::vector<std::string> arguments{"test","--geometry",csv.string(),"--emin","-4",
        "--emax","4","--bin-width","0.5","--order-parameter","weighted_sum",
        "--q-bin-width","1","--support-stability-checks","3"};
    std::vector<char*> argv; for(auto& argument:arguments) argv.push_back(argument.data());
    auto config=wl::parse_arguments(static_cast<int>(argv.size()),argv.data());
    config.resolve_order_parameter(geometry); config.resolve_mcs(geometry.size()); config.validate();
    require(config.order_parameter&&config.wl.support_stability_checks==3,
            "CLI weighted order-parameter configuration");
    const auto order=wl::WeightedOrderParameter::create(geometry.order_weights,1.0);
    near(order.normalization,2.0,1e-14,"Q normalization");
    require(order.grid.bins()%2==1&&order.grid.index(-2.0)&&order.grid.index(2.0),
            "symmetric Q grid includes extrema");
    const std::vector<std::int8_t> spins{1,-1};
    near(order.evaluate(spins),2.0,1e-14,"weighted Q evaluation");
    near(order.flip_delta(0,spins[0]),-2.0,1e-14,"incremental Q delta");

    const wl::EnergyGrid energy_grid{-4,4,0.5};
    const wl::DosGrid grid{energy_grid,order.grid};
    require(grid.cells()==energy_grid.bins()*order.grid.bins(),"joint grid cell count");
    for(std::size_t e=0;e<grid.energy_bins();++e)
        for(std::size_t q=0;q<grid.q_bins();++q) {
            const auto cell=grid.flatten(e,q);
            require(grid.energy_bin(cell)==e&&grid.q_bin(cell)==q,"joint flatten round trip");
        }

    auto couplings=std::make_shared<wl::DenseCouplings>(geometry,1.0);
    const auto exact=wl::exact_enumeration(*couplings,grid,order);
    require(std::accumulate(exact.histogram.begin(),exact.histogram.end(),std::uint64_t{0})==4,
            "joint exact state count");
    for(std::size_t e=0;e<grid.energy_bins();++e)
        for(std::size_t q=0;q<grid.q_bins();++q)
            require(exact.histogram[grid.flatten(e,q)]==
                    exact.histogram[grid.flatten(e,grid.q_bins()-1-q)],
                    "global spin inversion gives g(E,Q)=g(E,-Q)");
    const auto joint=wl::stitch_joint_dos(grid,std::span(&exact,1),true,true,2,order.normalization);
    double state_sum=0.0;
    for(std::size_t i=0;i<joint.log_g.size();++i)
        if(joint.valid[i]&&std::isfinite(joint.log_g[i])) state_sum+=std::exp(joint.log_g[i]);
    near(state_sum,4.0,1e-13,"joint complete normalization");
    const auto marginal=wl::marginalize(joint);
    const auto direct=wl::exact_enumeration(*couplings,energy_grid);
    require(marginal.histogram==direct.histogram,"joint marginal histogram equals direct energy DOS");
    const std::array<double,2> temperatures{1.0,2.0};
    const auto q_thermo=wl::order_parameter_thermodynamics(joint,temperatures,2);
    require(q_thermo.size()==2&&std::isfinite(q_thermo[0].susceptibility)&&
            std::isfinite(q_thermo[0].binder_cumulant),"finite Q thermodynamics");
    const auto distribution=wl::order_parameter_distribution(joint,temperatures);
    for(std::size_t t=0;t<temperatures.size();++t) {
        double probability=0.0;
        for(std::size_t q=0;q<grid.q_bins();++q)
            probability+=distribution[t*grid.q_bins()+q].probability;
        near(probability,1.0,1e-13,"normalized Q distribution");
    }

    auto order_ptr=std::make_shared<wl::WeightedOrderParameter>(order);
    wl::WlParameters parameters; parameters.check_interval_attempts=1;
    parameters.support_stability_checks=1;
    wl::WangLandauWalker walker(31,couplings,grid,{0,grid.energy_bins()},parameters,77,
                                order_ptr,spins);
    auto rejected=walker.snapshot();
    std::fill(rejected.log_g.begin(),rejected.log_g.end(),1000.0);
    const auto rejected_cell=grid.index(rejected.energy,rejected.order_parameter);
    require(rejected_cell.has_value(),"current joint cell before rejection");
    rejected.log_g[*rejected_cell]=0.0;
    const auto visits_before=rejected.histogram[*rejected_cell];
    walker.restore(rejected);
    require(!walker.attempt_flip()&&walker.order_parameter()==rejected.order_parameter,
            "rejected joint flip preserves Q");
    require(walker.histogram()[*rejected_cell]==visits_before+1,
            "rejected joint flip updates current cell");
    for(int step=0;step<50;++step) {
        walker.attempt_flip();
        near(walker.order_parameter(),order.evaluate(walker.snapshot().spins),1e-13,
             "incremental walker Q");
    }
    const auto checkpoint=directory/"joint.chk";
    wl::save_checkpoint(checkpoint.string(),walker.snapshot());
    wl::WangLandauWalker restored(31,couplings,grid,{0,grid.energy_bins()},parameters,77,
                                  order_ptr,spins);
    restored.restore(wl::load_checkpoint(checkpoint.string()));
    near(restored.order_parameter(),walker.order_parameter(),1e-14,"joint checkpoint Q");
    require(restored.log_g()==walker.log_g()&&restored.histogram()==walker.histogram(),
            "joint checkpoint estimator");
    const auto walker_accept=walker.attempt_flip();
    const auto restored_accept=restored.attempt_flip();
    require(walker_accept==restored_accept&&walker.snapshot().spins==restored.snapshot().spins&&
            walker.log_g()==restored.log_g()&&walker.histogram()==restored.histogram(),
            "joint checkpoint deterministic continuation");

    auto single_couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    auto single_order=std::make_shared<wl::WeightedOrderParameter>(
        wl::WeightedOrderParameter::create(std::vector<double>{1.0},1.0));
    const wl::DosGrid single_grid{{-1,1,0.5},single_order->grid};
    wl::WlParameters stable; stable.inverse_time_enabled=false; stable.flatness=0.1;
    stable.minimum_visits=1; stable.check_interval_attempts=1; stable.support_stability_checks=1;
    wl::WangLandauWalker stability(32,single_couplings,single_grid,
        {0,single_grid.energy_bins()},stable,9,single_order,std::vector<std::int8_t>{1});
    stability.attempt_flip();
    require(stability.flat()&&!stability.ready_for_iteration(),
            "new joint cell resets support-stability clock");
    stability.attempt_flip();
    require(stability.ready_for_iteration(),"stable active joint support permits next iteration");
    std::filesystem::remove_all(directory);
}

void test_joint_relaxed_union_and_guards() {
    auto couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    const auto order_value=wl::WeightedOrderParameter::create({1.0},1.0);
    auto order=std::make_shared<const wl::WeightedOrderParameter>(order_value);
    const wl::DosGrid grid{{-1.0,1.0,1.0},order->grid};
    const wl::EnergyWindow window{0,grid.energy_bins()};
    wl::WlParameters parameters{0.8,1,1e-8,1};
    wl::WangLandauWalker first(101,couplings,grid,window,parameters,5,order);
    wl::WangLandauWalker second(102,couplings,grid,window,parameters,5,order);
    wl::WangLandauWalker third(103,couplings,grid,window,parameters,5,order);
    const std::array<std::size_t,4> cells{
        grid.flatten(0,0),grid.flatten(0,1),grid.flatten(1,1),grid.flatten(1,2)};
    const std::array<double,4> base{0.0,1.0,2.0,3.0};
    const auto prepare=[&](wl::WangLandauWalker& walker,double offset,
                           std::initializer_list<std::size_t> active_cells) {
        auto state=walker.snapshot();
        std::fill(state.active.begin(),state.active.end(),0);
        std::fill(state.histogram.begin(),state.histogram.end(),0);
        std::fill(state.log_g.begin(),state.log_g.end(),offset);
        for(std::size_t i=0;i<cells.size();++i) state.log_g[cells[i]]=base[i]+offset;
        for(const auto cell:active_cells) { state.active[cell]=1; state.histogram[cell]=10; }
        walker.restore(state);
    };
    prepare(first,10.0,{cells[0],cells[1]});
    prepare(second,-4.0,{cells[1],cells[2]});
    prepare(third,7.0,{cells[2],cells[3]});
    const std::array<const wl::WangLandauWalker*,3> walkers{&first,&second,&third};
    const auto fragment=wl::summarize_walkers(window,walkers,grid.energy_bins());
    require(fragment.valid[cells[0]]&&fragment.valid[cells[3]],
            "relaxed union retains cells visited by one walker");
    require(fragment.contributors[cells[0]]==1&&fragment.contributors[cells[1]]==2&&
            fragment.contributors[cells[2]]==2&&fragment.contributors[cells[3]]==1,
            "relaxed union contributor counts");
    require(fragment.support_component[cells[0]]==fragment.support_component[cells[3]],
            "chain overlap creates one support component");
    near(fragment.log_g[cells[1]]-fragment.log_g[cells[0]],1.0,1e-13,
         "weighted alignment preserves first DOS difference");
    near(fragment.log_g[cells[3]]-fragment.log_g[cells[2]],1.0,1e-13,
         "weighted alignment propagates through overlap chain");
    const auto stitched=wl::stitch_joint_dos(grid,std::span(&fragment,1),false,false,1,
                                              order->normalization);
    require(stitched.contributors[cells[1]]==2&&stitched.support_component[cells[3]]==0,
            "connected relaxed union stitches successfully");

    const wl::DosGrid window_grid{{-2.0,2.0,1.0},order->grid};
    const auto make_fragment=[&](wl::EnergyWindow energy_window) {
        const auto count=window_grid.cells();
        return wl::DosFragment{energy_window,
            std::vector<double>(count,std::numeric_limits<double>::quiet_NaN()),
            std::vector<std::uint64_t>(count,0),
            std::vector<double>(count,std::numeric_limits<double>::quiet_NaN()),
            std::vector<std::uint8_t>(count,0),window_grid,
            std::vector<std::uint32_t>(count,0),std::vector<std::int32_t>(count,-1)};
    };
    auto left_fragment=make_fragment({0,3});
    auto right_fragment=make_fragment({1,4});
    const auto left_only=window_grid.flatten(0,1);
    const auto shared=window_grid.flatten(1,1);
    const auto right_only=window_grid.flatten(3,1);
    const auto set_cell=[](wl::DosFragment& value,std::size_t cell,double log_g) {
        value.valid[cell]=1; value.log_g[cell]=log_g; value.histogram[cell]=10;
        value.standard_error[cell]=0.1; value.contributors[cell]=1;
        value.support_component[cell]=0;
    };
    set_cell(left_fragment,left_only,2.0); set_cell(left_fragment,shared,3.0);
    set_cell(right_fragment,shared,-4.0); set_cell(right_fragment,right_only,-2.0);
    const std::array<wl::DosFragment,2> window_fragments{left_fragment,right_fragment};
    const auto across_windows=wl::stitch_joint_dos(window_grid,window_fragments,false,false,1,
                                                    order->normalization);
    require(across_windows.valid[left_only]&&across_windows.valid[right_only],
            "support connected through an adjacent window is retained");
    near(across_windows.log_g[right_only]-across_windows.log_g[shared],2.0,1e-13,
         "inter-window weighted shift preserves the right DOS shape");

    wl::WangLandauWalker isolated(104,couplings,grid,window,parameters,5,order);
    prepare(isolated,2.0,{cells[3]});
    const std::array<const wl::WangLandauWalker*,2> disconnected_walkers{&first,&isolated};
    const auto disconnected=wl::summarize_walkers(window,disconnected_walkers,
                                                   grid.energy_bins());
    bool disconnected_rejected=false;
    try { (void)wl::stitch_joint_dos(grid,std::span(&disconnected,1),false,false,1,
                                     order->normalization); }
    catch(const wl::DisconnectedSupportError& error) {
        disconnected_rejected=error.components()==2;
    }
    require(disconnected_rejected,"disconnected support cannot receive an arbitrary global shift");

    const auto before=first.snapshot();
    auto corrupt=before; corrupt.fields[0]+=1.0;
    bool corrupt_rejected=false;
    try { first.restore(corrupt); } catch(const std::runtime_error&) { corrupt_rejected=true; }
    const auto after=first.snapshot();
    require(corrupt_rejected&&after.spins==before.spins&&after.fields==before.fields&&
            after.log_g==before.log_g&&after.histogram==before.histogram&&
            after.rng_state==before.rng_state,"failed restore is transactional");
    corrupt=before; corrupt.last_new_cell_attempt=corrupt.attempted+1;
    corrupt_rejected=false;
    try { first.restore(corrupt); } catch(const std::invalid_argument&) { corrupt_rejected=true; }
    require(corrupt_rejected,"checkpoint rejects a future last-new-cell clock");
    corrupt=before; corrupt.energy_grid.width=0.5;
    corrupt_rejected=false;
    try { first.restore(corrupt); } catch(const std::invalid_argument&) { corrupt_rejected=true; }
    require(corrupt_rejected,"checkpoint rejects a different energy layout");
    corrupt=before; corrupt.order_weights[0]=-corrupt.order_weights[0];
    corrupt_rejected=false;
    try { first.restore(corrupt); } catch(const std::invalid_argument&) { corrupt_rejected=true; }
    require(corrupt_rejected,"checkpoint rejects different order-parameter weights");
    corrupt=before; corrupt.format_version=4;
    corrupt_rejected=false;
    try { first.restore(corrupt); } catch(const std::invalid_argument&) { corrupt_rejected=true; }
    require(corrupt_rejected,"legacy joint checkpoint is rejected");

    const auto edge_order=wl::WeightedOrderParameter::create({1.5000000000005},1.0);
    require(edge_order.grid.index(edge_order.normalization)&&
            edge_order.grid.index(-edge_order.normalization),
            "Q grid includes extrema just above a half-integer ratio");

    wl::DosFragment structural{{0,grid.energy_bins()},
        std::vector<double>(grid.cells(),-std::numeric_limits<double>::infinity()),
        std::vector<std::uint64_t>(grid.cells(),0),std::vector<double>(grid.cells(),0.0),
        std::vector<std::uint8_t>(grid.cells(),1),grid,{}, {}};
    const auto structural_marginal=wl::marginalize_fragment(structural);
    require(std::all_of(structural_marginal.valid.begin(),structural_marginal.valid.end(),
                        [](auto value){return value!=0;})&&
            std::all_of(structural_marginal.log_g.begin(),structural_marginal.log_g.end(),
                        [](double value){return std::isinf(value)&&value<0.0;}),
            "known joint structural zeros remain known after marginalization");

    bool exact_rejected=false;
    try { (void)wl::exact_enumeration(*couplings,wl::EnergyGrid{1.0,2.0,0.5}); }
    catch(const std::runtime_error&) { exact_rejected=true; }
    require(exact_rejected,"exact enumeration rejects states outside the energy grid");

    wl::RunConfig invalid_temperature;
    invalid_temperature.temperatures={1.0,-1.0};
    bool temperature_rejected=false;
    try { invalid_temperature.validate(false); }
    catch(const std::invalid_argument&) { temperature_rejected=true; }
    require(temperature_rejected,"invalid temperatures are rejected before sampling");
}

void test_adaptive_energy_windows() {
    wl::EnergyGrid grid{-10.0,10.0,0.5};
    const auto initial=wl::partition_windows(grid.bins(),4,0.5);
    std::vector<wl::DosFragment> fragments;
    std::vector<wl::WindowSamplingStatistics> sampling;
    for(const auto window:initial) {
        wl::DosFragment fragment{window,std::vector<double>(grid.bins(),0.0),
            std::vector<std::uint64_t>(grid.bins(),0),std::vector<double>(grid.bins(),0.0),
            std::vector<std::uint8_t>(grid.bins(),0),wl::DosGrid{grid,std::nullopt},{},{}};
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
            std::vector<std::uint8_t>(fine_grid.bins(),0),
            wl::DosGrid{fine_grid,std::nullopt},{},{}};
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

void test_nalivaiko_refinement_mask() {
    auto couplings=std::make_shared<wl::DenseCouplings>(
        wl::Geometry::simple_cubic(1,1,1,1.0,{0,0,1},false),1.0);
    wl::EnergyGrid grid{-1,1,0.5};
    wl::WlParameters parameters{0.8,1,1e-8,1,0,false};
    parameters.nalivaiko_mod=true;
    wl::WangLandauWalker walker(40,couplings,grid,{0,grid.bins()},parameters,23);
    auto snapshot=walker.snapshot();
    std::fill(snapshot.active.begin(),snapshot.active.end(),0);
    std::fill(snapshot.histogram.begin(),snapshot.histogram.end(),0);
    snapshot.active[0]=snapshot.active[1]=1;
    snapshot.histogram[0]=snapshot.histogram[1]=10;
    walker.restore(snapshot);
    require(walker.ready_for_iteration(),"Nalivaiko stage must initially satisfy flatness");
    walker.begin_next_iteration();
    const auto reset_statistics=walker.histogram_statistics();
    require(reset_statistics.active_bins==0&&reset_statistics.covered_bins==0,
            "Nalivaiko refinement mask must reset with histogram");
    require(walker.active_mask()[0]!=0&&walker.active_mask()[1]!=0,
            "Nalivaiko reset must preserve cumulative active bins");
    const std::array<const wl::WangLandauWalker*,1> walkers{&walker};
    const auto fragment=wl::summarize_walkers({0,grid.bins()},walkers,grid.bins());
    require(fragment.valid[0]!=0&&fragment.valid[1]!=0,
            "Nalivaiko reset must preserve DOS validity");

    const auto checkpoint_path=
        (std::filesystem::temp_directory_path()/"wl_nalivaiko_checkpoint.bin").string();
    wl::save_checkpoint(checkpoint_path,walker.snapshot());
    wl::WangLandauWalker restored(40,couplings,grid,{0,grid.bins()},parameters,23);
    restored.restore(wl::load_checkpoint(checkpoint_path));
    std::filesystem::remove(checkpoint_path);
    require(restored.histogram_statistics().active_bins==0,
            "checkpoint restore must reconstruct an empty refinement mask");
    require(restored.active_mask()[0]!=0&&restored.active_mask()[1]!=0,
            "checkpoint restore must preserve cumulative active bins");
    restored.attempt_flip();
    const auto next_statistics=restored.histogram_statistics();
    require(next_statistics.active_bins==1&&next_statistics.covered_bins==1,
            "new stage visits must rebuild the Nalivaiko refinement mask");
    require(restored.flat()&&restored.covered(),
            "flatness and coverage must ignore cumulative bins from earlier stages");

    wl::WlParameters inverse_parameters{0.99,100,1e-8,1};
    inverse_parameters.nalivaiko_mod=true;
    wl::WangLandauWalker inverse(41,couplings,grid,{0,grid.bins()},inverse_parameters,29);
    auto inverse_snapshot=inverse.snapshot();
    std::fill(inverse_snapshot.active.begin(),inverse_snapshot.active.end(),0);
    std::fill(inverse_snapshot.histogram.begin(),inverse_snapshot.histogram.end(),0);
    inverse_snapshot.active[0]=inverse_snapshot.active[1]=1;
    inverse_snapshot.histogram[0]=inverse_snapshot.histogram[1]=1;
    inverse_snapshot.attempted=100;
    inverse_snapshot.factor=0.02;
    inverse.restore(inverse_snapshot);
    require(inverse.ready_for_iteration(),"Nalivaiko coverage must permit the 1/t transition");
    inverse.begin_next_iteration();
    require(inverse.stage()==wl::RefinementStage::inverse_time,
            "Nalivaiko walker must enter inverse-time refinement");
    near(inverse.factor(),2.0/100.0,1e-14,
         "Nalivaiko 1/t clock must use cumulative active bins");
    require(inverse.histogram_statistics().active_bins==0,
            "Nalivaiko refinement mask must also reset on the 1/t transition");
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

    const auto local_before=first.snapshot();
    const auto remote_configuration=second.snapshot();
    std::vector<std::int8_t> exchange_spins=remote_configuration.spins;
    std::vector<double> exchange_fields=remote_configuration.fields;
    first.swap_configuration_buffers(exchange_spins,exchange_fields,
                                     remote_configuration.energy,
                                     remote_configuration.order_parameter);
    require(std::equal(first.spins().begin(),first.spins().end(),
                       remote_configuration.spins.begin())&&
            std::equal(first.fields().begin(),first.fields().end(),
                       remote_configuration.fields.begin()),
            "buffer exchange installs the remote configuration");
    require(exchange_spins==local_before.spins&&exchange_fields==local_before.fields,
            "buffer exchange retains the old configuration for reuse");
    require(first.energy_bin()==grid.index(remote_configuration.energy),
            "buffer exchange refreshes the cached energy bin");

    second_snapshot=second.snapshot();
    std::fill(second_snapshot.active.begin(),second_snapshot.active.end(),0);
    second_snapshot.active[3]=1; second.restore(second_snapshot);
    bool rejected=false;
    try { (void)wl::summarize_walkers(window,walkers,grid.bins()); }
    catch(const std::runtime_error&) { rejected=true; }
    require(rejected,"summary must reject walkers without a common active bin");
    const auto diagnostic_fragment=
        wl::summarize_walkers(window,walkers,grid.bins(),true);
    require(std::none_of(diagnostic_fragment.valid.begin(),diagnostic_fragment.valid.end(),
                         [](auto value){return value!=0;}),
            "diagnostic summary preserves an empty intersection for output");
    bool insufficient=false;
    try {
        (void)wl::stitch_dos(grid,std::span(&diagnostic_fragment,1),false,1);
    } catch(const wl::InsufficientSupportError&) {
        insufficient=true;
    }
    require(insufficient,"empty diagnostic support has a typed postprocessing status");
}
}

int main() {
    const std::vector<std::pair<const char*,std::function<void()>>> tests{
      {"pair_formula",test_pair_formula},{"minimum_image",test_minimum_image},
      {"target_metropolis_initialization",test_target_metropolis_initialization},
      {"csv_geometry_ini",test_csv_geometry_and_ini},
      {"backends_incremental",test_backends_and_incremental_fields},
      {"energy_grid_windows",test_energy_grid_and_windows},
      {"sparse_dos_csv_output",test_sparse_dos_csv_output},
      {"walker_checkpoint",test_walker_and_checkpoint},
      {"forced_acceptance",test_forced_acceptance},
      {"refinement_transition",test_refinement_transition},
      {"return_mode",test_return_mode},
      {"histogram_statistics",test_histogram_statistics},
      {"nalivaiko_refinement_mask",test_nalivaiko_refinement_mask},
      {"classic_rewl_independence_summary",test_classic_rewl_independence_and_summary},
      {"unlimited_max_attempts",test_unlimited_max_attempts},
      {"exact_thermo",test_exact_enumeration_and_thermo},
      {"joint_dos_order_parameter",test_joint_dos_and_order_parameter},
      {"joint_relaxed_union_guards",test_joint_relaxed_union_and_guards},
      {"adaptive_energy_windows",test_adaptive_energy_windows}};
    int failed=0;
    for(const auto& [name,test]:tests) try { test(); std::cout<<"PASS "<<name<<'\n'; }
      catch(const std::exception& e) { ++failed; std::cerr<<"FAIL "<<name<<": "<<e.what()<<'\n'; }
    return failed?1:0;
}

