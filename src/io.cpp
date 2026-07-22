#include "wl/io.hpp"

#include <array>
#include <charconv>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <cctype>

namespace wl {
namespace {

template<class T> T number(std::string_view text, std::string_view option) {
    if constexpr (std::is_integral_v<T>) {
        T value{};
        const auto [end, error] = std::from_chars(text.data(), text.data()+text.size(), value);
        if (error != std::errc{} || end != text.data()+text.size())
            throw std::invalid_argument("Invalid value for " + std::string(option));
        return value;
    } else {
        std::string copy(text);
        std::size_t used = 0;
        const auto value = std::stod(copy, &used);
        if (used != copy.size()) throw std::invalid_argument("Invalid value for " + std::string(option));
        return static_cast<T>(value);
    }
}

bool boolean(std::string_view text, std::string_view option) {
    std::string normalized(text);
    std::transform(normalized.begin(),normalized.end(),normalized.begin(),[](unsigned char c){return static_cast<char>(std::tolower(c));});
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") return true;
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") return false;
    throw std::invalid_argument("Invalid boolean for " + std::string(option));
}

std::string trim(std::string value) {
    const auto first=value.find_first_not_of(" \t\r\n");
    if(first==std::string::npos) return {};
    const auto last=value.find_last_not_of(" \t\r\n");
    value=value.substr(first,last-first+1);
    if(value.size()>=2 && ((value.front()=='"'&&value.back()=='"')||(value.front()=='\''&&value.back()=='\'')))
        value=value.substr(1,value.size()-2);
    return value;
}

std::vector<double> number_list(std::string_view text,std::string_view key) {
    std::vector<double> result; std::stringstream stream{std::string(text)}; std::string item;
    while(std::getline(stream,item,',')) result.push_back(number<double>(trim(item),key));
    if(result.empty()) throw std::invalid_argument("Empty list for "+std::string(key));
    return result;
}

std::string json_escape(std::string_view value) {
    std::string result;
    for(const char ch:value) {
        switch(ch) {
            case '\\': result += "\\\\"; break;
            case '"': result += "\\\""; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default: result += ch; break;
        }
    }
    return result;
}

std::uint64_t attempts_from_mcs(double mcs, std::size_t spin_count,
                                bool allow_zero, std::string_view name) {
    if(spin_count==0) throw std::invalid_argument("Cannot resolve MCS for an empty system");
    if(!std::isfinite(mcs) || mcs<0.0 || (!allow_zero && mcs==0.0))
        throw std::invalid_argument("Invalid MCS value for "+std::string(name));
    if(mcs==0.0) return 0;
    const auto raw=static_cast<long double>(mcs)*static_cast<long double>(spin_count);
    const auto maximum=static_cast<long double>(std::numeric_limits<std::uint64_t>::max());
    if(raw>maximum-0.5L)
        throw std::overflow_error("MCS value is too large for "+std::string(name));
    const auto rounded=static_cast<std::uint64_t>(std::floor(raw+0.5L));
    return std::max<std::uint64_t>(1,rounded);
}

void apply_ini_setting(RunConfig& c,const std::string& section,const std::string& raw_key,
                       const std::string& raw_value,bool& have_min,bool& have_max,bool& have_width,
                       const std::filesystem::path& ini_directory) {
    auto key=raw_key;
    std::transform(key.begin(),key.end(),key.begin(),[](unsigned char ch){return static_cast<char>(std::tolower(ch));});
    const auto full=section.empty()?key:section+"."+key; const auto value=trim(raw_value);
    if(full=="geometry.file"||full=="geometry.csv") {
        auto path=std::filesystem::path(value); if(path.is_relative()) path=ini_directory/path;
        c.geometry_file=path.lexically_normal().string();
    } else if(full=="geometry.nx") c.nx=number<std::size_t>(value,full);
    else if(full=="geometry.ny") c.ny=number<std::size_t>(value,full);
    else if(full=="geometry.nz") c.nz=number<std::size_t>(value,full);
    else if(full=="geometry.spacing") c.spacing=number<double>(value,full);
    else if(full=="geometry.axis_x") c.axis.x=number<double>(value,full);
    else if(full=="geometry.axis_y") c.axis.y=number<double>(value,full);
    else if(full=="geometry.axis_z") c.axis.z=number<double>(value,full);
    else if(full=="geometry.periodic") c.periodic=boolean(value,full);
    else if(full=="geometry.box_x") c.box_lengths.x=number<double>(value,full);
    else if(full=="geometry.box_y") c.box_lengths.y=number<double>(value,full);
    else if(full=="geometry.box_z") c.box_lengths.z=number<double>(value,full);
    else if(full=="physics.coupling"||full=="physics.coupling_scale") c.coupling_scale=number<double>(value,full);
    else if(full=="physics.cutoff") c.cutoff=number<double>(value,full);
    else if(full=="energy.emin"||full=="energy.minimum") {c.grid.minimum=number<double>(value,full);have_min=true;}
    else if(full=="energy.emax"||full=="energy.maximum") {c.grid.maximum=number<double>(value,full);have_max=true;}
    else if(full=="energy.bin_width"||full=="energy.width") {c.grid.width=number<double>(value,full);have_width=true;}
    else if(full=="energy.complete_range") c.complete_range=boolean(value,full);
    else if(full=="parallel.windows") c.windows=number<std::size_t>(value,full);
    else if(full=="parallel.walkers"||full=="parallel.walkers_per_rank") c.walkers_per_rank=number<std::size_t>(value,full);
    else if(full=="parallel.overlap") c.overlap=number<double>(value,full);
    else if(full=="adaptive_windows.enabled") c.adaptive_windows.enabled=boolean(value,full);
    else if(full=="adaptive_windows.iterations") c.adaptive_windows.iterations=number<std::size_t>(value,full);
    else if(full=="adaptive_windows.pilot_mcs") c.adaptive_windows.pilot_mcs=number<double>(value,full);
    else if(full=="adaptive_windows.smoothing_width") c.adaptive_windows.smoothing_width=number<double>(value,full);
    else if(full=="adaptive_windows.minimum_width") c.adaptive_windows.minimum_width=number<double>(value,full);
    else if(full=="adaptive_windows.diffusivity_floor_fraction") c.adaptive_windows.diffusivity_floor_fraction=number<double>(value,full);
    else if(full=="adaptive_windows.curvature_weight") c.adaptive_windows.curvature_weight=number<double>(value,full);
    else if(full=="adaptive_windows.round_trip_target") c.adaptive_windows.round_trip_target=number<double>(value,full);
    else if(full=="adaptive_windows.maximum_round_trip_penalty") c.adaptive_windows.maximum_round_trip_penalty=number<double>(value,full);
    else if(full=="adaptive_windows.round_trip_margin_fraction") c.wl.round_trip_margin_fraction=number<double>(value,full);
    else if(full=="parallel.exchange_interval") { c.exchange_interval_attempts=number<std::uint64_t>(value,full); c.exchange_interval_uses_mcs=false; }
    else if(full=="parallel.exchange_interval_mcs") { c.exchange_interval_mcs=number<double>(value,full); c.exchange_interval_uses_mcs=true; }
    else if(full=="wl.flatness") c.wl.flatness=number<double>(value,full);
    else if(full=="wl.minimum_visits"||full=="wl.min_visits") c.wl.minimum_visits=number<std::uint64_t>(value,full);
    else if(full=="wl.final_factor") c.wl.final_factor=number<double>(value,full);
    else if(full=="wl.inverse_time") c.wl.inverse_time_enabled=boolean(value,full);
    else if(full=="wl.nalivaiko_mod") c.wl.nalivaiko_mod=boolean(value,full);
    else if(full=="wl.check_interval") { c.wl.check_interval_attempts=number<std::uint64_t>(value,full); c.check_interval_uses_mcs=false; }
    else if(full=="wl.check_interval_mcs") { c.check_interval_mcs=number<double>(value,full); c.check_interval_uses_mcs=true; }
    else if(full=="wl.force_accept_after") { c.wl.force_accept_after_attempts=number<std::uint64_t>(value,full); c.force_accept_after_uses_mcs=false; }
    else if(full=="wl.force_accept_after_mcs") { c.force_accept_after_mcs=number<double>(value,full); c.force_accept_after_uses_mcs=true; }
    else if(full=="initialization.max_attempts"||full=="wl.initialization_max_attempts")
        c.wl.initialization_max_attempts=number<std::uint64_t>(value,full);
    else if(full=="initialization.target_fraction"||full=="wl.initialization_target_fraction")
        c.wl.initialization_target_fraction=number<double>(value,full);
    else if(full=="initialization.temperature_fraction"||full=="wl.initialization_temperature_fraction")
        c.wl.initialization_temperature_fraction=number<double>(value,full);
    else if(full=="initialization.stall_attempts_per_spin"||
            full=="wl.initialization_stall_attempts_per_spin")
        c.wl.initialization_stall_attempts_per_spin=number<std::uint64_t>(value,full);
    else if(full=="initialization.temperature_multiplier"||
            full=="wl.initialization_temperature_multiplier")
        c.wl.initialization_temperature_multiplier=number<double>(value,full);
    else if(full=="initialization.max_temperature_fraction"||
            full=="wl.initialization_max_temperature_fraction")
        c.wl.initialization_max_temperature_fraction=number<double>(value,full);
    else if(full=="run.seed") { c.seed=number<std::uint64_t>(value,full); c.seed_explicit=true; }
    else if(full=="run.max_attempts") { c.max_attempts=number<std::uint64_t>(value,full); c.max_limit_uses_mcs=false; }
    else if(full=="run.max_mcs") { c.max_mcs=number<double>(value,full); c.max_limit_uses_mcs=true; }
    else if(full=="run.checkpoint_interval") { c.checkpoint_interval_attempts=number<std::uint64_t>(value,full); c.checkpoint_interval_uses_mcs=false; }
    else if(full=="run.checkpoint_interval_mcs") { c.checkpoint_interval_mcs=number<double>(value,full); c.checkpoint_interval_uses_mcs=true; }
    else if(full=="run.checkpoint") c.checkpoint_path=value;
    else if(full=="run.temperatures") c.temperatures=number_list(value,full);
    else if(full=="run.pilot") c.pilot=boolean(value,full);
    else if(full=="run.progress") c.progress_interval_seconds=number<double>(value,full);
    else if(full=="output.prefix") c.output_prefix=value;
    else throw std::invalid_argument("Unknown INI setting: "+full);
}

void load_ini(RunConfig& c,const std::string& path,bool& have_min,bool& have_max,bool& have_width) {
    std::ifstream input(path); if(!input) throw std::runtime_error("Cannot open INI file: "+path);
    c.config_file=path; const auto directory=std::filesystem::absolute(path).parent_path();
    std::string line,section; std::size_t line_number=0;
    while(std::getline(input,line)) {
        ++line_number; line=trim(line);
        if(line.empty()||line[0]=='#'||line[0]==';') continue;
        if(line.front()=='['&&line.back()==']') {
            section=trim(line.substr(1,line.size()-2));
            std::transform(section.begin(),section.end(),section.begin(),[](unsigned char ch){return static_cast<char>(std::tolower(ch));});
            continue;
        }
        const auto equals=line.find('=');
        if(equals==std::string::npos) throw std::runtime_error("Invalid INI line "+std::to_string(line_number));
        try { apply_ini_setting(c,section,trim(line.substr(0,equals)),line.substr(equals+1),
                                have_min,have_max,have_width,directory); }
        catch(const std::exception& error) {
            throw std::runtime_error("INI line "+std::to_string(line_number)+": "+error.what());
        }
    }
}

template<class T> void write_value(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}
template<class T> void read_value(std::istream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) throw std::runtime_error("Truncated checkpoint");
}
template<class T> void write_vector(std::ostream& out, const std::vector<T>& values) {
    const auto size = static_cast<std::uint64_t>(values.size()); write_value(out, size);
    out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(size*sizeof(T)));
}
template<class T> void read_vector(std::istream& in, std::vector<T>& values) {
    std::uint64_t size{}; read_value(in, size);
    if (size > 1'000'000'000ULL) throw std::runtime_error("Unreasonable checkpoint vector size");
    values.resize(static_cast<std::size_t>(size));
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(size*sizeof(T)));
    if (!in) throw std::runtime_error("Truncated checkpoint vector");
}

} // namespace

void RunConfig::resolve_mcs(std::size_t spin_count) {
    if(spin_count==0) throw std::invalid_argument("MCS requires at least one spin");
    if(resolved_spin_count!=0 && resolved_spin_count!=spin_count)
        throw std::logic_error("MCS intervals were resolved for a different spin count");

    if(exchange_interval_uses_mcs) {
        exchange_interval_attempts=attempts_from_mcs(exchange_interval_mcs,spin_count,false,"exchange_interval_mcs");
    } else exchange_interval_mcs=static_cast<double>(exchange_interval_attempts)/static_cast<double>(spin_count);

    if(check_interval_uses_mcs)
        wl.check_interval_attempts=attempts_from_mcs(check_interval_mcs,spin_count,false,"check_interval_mcs");
    else check_interval_mcs=static_cast<double>(wl.check_interval_attempts)/static_cast<double>(spin_count);

    if(force_accept_after_uses_mcs)
        wl.force_accept_after_attempts=attempts_from_mcs(force_accept_after_mcs,spin_count,true,"force_accept_after_mcs");
    else force_accept_after_mcs=static_cast<double>(wl.force_accept_after_attempts)/static_cast<double>(spin_count);

    if(max_limit_uses_mcs)
        max_attempts=attempts_from_mcs(max_mcs,spin_count,true,"max_mcs");
    else max_mcs=static_cast<double>(max_attempts)/static_cast<double>(spin_count);

    if(checkpoint_interval_uses_mcs)
        checkpoint_interval_attempts=attempts_from_mcs(checkpoint_interval_mcs,spin_count,false,"checkpoint_interval_mcs");
    else checkpoint_interval_mcs=static_cast<double>(checkpoint_interval_attempts)/static_cast<double>(spin_count);

    resolved_spin_count=spin_count;
}

bool RunConfig::uses_legacy_attempt_units() const noexcept {
    return !exchange_interval_uses_mcs || !check_interval_uses_mcs ||
           !force_accept_after_uses_mcs || !max_limit_uses_mcs || !checkpoint_interval_uses_mcs;
}

void RunConfig::validate(bool require_grid) const {
    if (geometry_file.empty() && (nx == 0 || ny == 0 || nz == 0))
        throw std::invalid_argument("Lattice dimensions must be positive");
    if (!(spacing > 0.0) || !std::isfinite(progress_interval_seconds) ||
        progress_interval_seconds < 0.0 || windows == 0 || walkers_per_rank == 0 ||
        overlap < 0.0 || overlap >= 1.0 || exchange_interval_attempts == 0 ||
        wl.check_interval_attempts == 0 || checkpoint_interval_attempts == 0 ||
        !std::isfinite(wl.round_trip_margin_fraction) ||
        wl.round_trip_margin_fraction<0.0 || wl.round_trip_margin_fraction>=0.5)
        throw std::invalid_argument("Invalid run configuration");
    if (require_grid && !energy_grid_explicit)
        throw std::invalid_argument("Production runs require --emin, --emax, and --bin-width");
    grid.validate();
    if (windows > grid.bins()) throw std::invalid_argument("More windows than energy bins");
    if(!explicit_windows.empty()) {
        if(explicit_windows.size()!=windows || explicit_windows.front().begin!=0 ||
           explicit_windows.back().end!=grid.bins())
            throw std::invalid_argument("Explicit energy windows must match the configured range");
        for(std::size_t i=0;i<explicit_windows.size();++i) {
            const auto& window=explicit_windows[i];
            if(window.begin>=window.end || window.end>grid.bins() ||
               (i!=0 && (window.begin<=explicit_windows[i-1].begin ||
                         window.begin>=explicit_windows[i-1].end)))
                throw std::invalid_argument("Explicit energy windows must be ordered and overlap");
        }
    }
    const auto& adaptive=adaptive_windows;
    if(adaptive.enabled && (adaptive.iterations==0 || !(adaptive.pilot_mcs>0.0) ||
       !std::isfinite(adaptive.pilot_mcs) || adaptive.smoothing_width<0.0 ||
       !std::isfinite(adaptive.smoothing_width) || adaptive.minimum_width<0.0 ||
       !std::isfinite(adaptive.minimum_width) ||
       !(adaptive.diffusivity_floor_fraction>0.0) ||
       !std::isfinite(adaptive.diffusivity_floor_fraction) ||
       adaptive.curvature_weight<0.0 || !std::isfinite(adaptive.curvature_weight) ||
       adaptive.round_trip_target<0.0 || !std::isfinite(adaptive.round_trip_target) ||
       adaptive.maximum_round_trip_penalty<1.0 ||
       !std::isfinite(adaptive.maximum_round_trip_penalty) ||
       (adaptive.minimum_width>0.0 && adaptive.minimum_width*static_cast<double>(windows)>
                                        grid.maximum-grid.minimum)))
        throw std::invalid_argument("Invalid adaptive-window configuration");
}

RunConfig parse_arguments(int argc, char** argv) {
    RunConfig c;
    bool have_min=false, have_max=false, have_width=false;
    for(int i=1;i<argc;++i) {
        if(std::string_view(argv[i])=="--config") {
            if(++i>=argc) throw std::invalid_argument("Missing value for --config");
            load_ini(c,argv[i],have_min,have_max,have_width);
        }
    }
    auto value = [&](int& i, std::string_view option) -> std::string_view {
        if (++i >= argc) throw std::invalid_argument("Missing value for " + std::string(option));
        return argv[i];
    };
    for (int i=1; i<argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--config") { ++i; continue; }
        else if (key == "--nx") c.nx=number<std::size_t>(value(i,key),key);
        else if (key == "--ny") c.ny=number<std::size_t>(value(i,key),key);
        else if (key == "--nz") c.nz=number<std::size_t>(value(i,key),key);
        else if (key == "--spacing") c.spacing=number<double>(value(i,key),key);
        else if (key == "--axis-x") c.axis.x=number<double>(value(i,key),key);
        else if (key == "--axis-y") c.axis.y=number<double>(value(i,key),key);
        else if (key == "--axis-z") c.axis.z=number<double>(value(i,key),key);
        else if (key == "--periodic") c.periodic=boolean(value(i,key),key);
        else if (key == "--geometry") c.geometry_file=value(i,key);
        else if (key == "--box-x") c.box_lengths.x=number<double>(value(i,key),key);
        else if (key == "--box-y") c.box_lengths.y=number<double>(value(i,key),key);
        else if (key == "--box-z") c.box_lengths.z=number<double>(value(i,key),key);
        else if (key == "--coupling") c.coupling_scale=number<double>(value(i,key),key);
        else if (key == "--cutoff") c.cutoff=number<double>(value(i,key),key);
        else if (key == "--emin") { c.grid.minimum=number<double>(value(i,key),key); have_min=true; }
        else if (key == "--emax") { c.grid.maximum=number<double>(value(i,key),key); have_max=true; }
        else if (key == "--bin-width") { c.grid.width=number<double>(value(i,key),key); have_width=true; }
        else if (key == "--windows") c.windows=number<std::size_t>(value(i,key),key);
        else if (key == "--walkers") c.walkers_per_rank=number<std::size_t>(value(i,key),key);
        else if (key == "--overlap") c.overlap=number<double>(value(i,key),key);
        else if (key == "--adaptive-windows") c.adaptive_windows.enabled=boolean(value(i,key),key);
        else if (key == "--adaptive-iterations") c.adaptive_windows.iterations=number<std::size_t>(value(i,key),key);
        else if (key == "--adaptive-pilot-mcs") c.adaptive_windows.pilot_mcs=number<double>(value(i,key),key);
        else if (key == "--adaptive-smoothing-width") c.adaptive_windows.smoothing_width=number<double>(value(i,key),key);
        else if (key == "--adaptive-minimum-width") c.adaptive_windows.minimum_width=number<double>(value(i,key),key);
        else if (key == "--adaptive-diffusivity-floor") c.adaptive_windows.diffusivity_floor_fraction=number<double>(value(i,key),key);
        else if (key == "--adaptive-curvature-weight") c.adaptive_windows.curvature_weight=number<double>(value(i,key),key);
        else if (key == "--adaptive-round-trip-target") c.adaptive_windows.round_trip_target=number<double>(value(i,key),key);
        else if (key == "--adaptive-round-trip-penalty") c.adaptive_windows.maximum_round_trip_penalty=number<double>(value(i,key),key);
        else if (key == "--adaptive-round-trip-margin") c.wl.round_trip_margin_fraction=number<double>(value(i,key),key);
        else if (key == "--seed") { c.seed=number<std::uint64_t>(value(i,key),key); c.seed_explicit=true; }
        else if (key == "--flatness") c.wl.flatness=number<double>(value(i,key),key);
        else if (key == "--min-visits") c.wl.minimum_visits=number<std::uint64_t>(value(i,key),key);
        else if (key == "--final-factor") c.wl.final_factor=number<double>(value(i,key),key);
        else if (key == "--check-interval") { c.wl.check_interval_attempts=number<std::uint64_t>(value(i,key),key); c.check_interval_uses_mcs=false; }
        else if (key == "--inverse-time") c.wl.inverse_time_enabled=boolean(value(i,key),key);
        else if (key == "--check-interval-mcs") { c.check_interval_mcs=number<double>(value(i,key),key); c.check_interval_uses_mcs=true; }
        else if (key == "--force-accept-after") { c.wl.force_accept_after_attempts=number<std::uint64_t>(value(i,key),key); c.force_accept_after_uses_mcs=false; }
        else if (key == "--force-accept-after-mcs") { c.force_accept_after_mcs=number<double>(value(i,key),key); c.force_accept_after_uses_mcs=true; }
        else if (key == "--initialization-max-attempts") c.wl.initialization_max_attempts=number<std::uint64_t>(value(i,key),key);
        else if (key == "--initialization-target-fraction") c.wl.initialization_target_fraction=number<double>(value(i,key),key);
        else if (key == "--initialization-temperature-fraction") c.wl.initialization_temperature_fraction=number<double>(value(i,key),key);
        else if (key == "--initialization-stall-attempts-per-spin") c.wl.initialization_stall_attempts_per_spin=number<std::uint64_t>(value(i,key),key);
        else if (key == "--initialization-temperature-multiplier") c.wl.initialization_temperature_multiplier=number<double>(value(i,key),key);
        else if (key == "--initialization-max-temperature-fraction") c.wl.initialization_max_temperature_fraction=number<double>(value(i,key),key);
        else if (key == "--exchange-interval") { c.exchange_interval_attempts=number<std::uint64_t>(value(i,key),key); c.exchange_interval_uses_mcs=false; }
        else if (key == "--exchange-interval-mcs") { c.exchange_interval_mcs=number<double>(value(i,key),key); c.exchange_interval_uses_mcs=true; }
        else if (key == "--max-attempts") { c.max_attempts=number<std::uint64_t>(value(i,key),key); c.max_limit_uses_mcs=false; }
        else if (key == "--max-mcs") { c.max_mcs=number<double>(value(i,key),key); c.max_limit_uses_mcs=true; }
        else if (key == "--checkpoint-interval") { c.checkpoint_interval_attempts=number<std::uint64_t>(value(i,key),key); c.checkpoint_interval_uses_mcs=false; }
        else if (key == "--checkpoint-interval-mcs") { c.checkpoint_interval_mcs=number<double>(value(i,key),key); c.checkpoint_interval_uses_mcs=true; }
        else if (key == "--checkpoint") c.checkpoint_path=value(i,key);
        else if (key == "--complete-range") c.complete_range=boolean(value(i,key),key);
        else if (key == "--output") c.output_prefix=value(i,key);
        else if (key == "--temperatures") {
            c.temperatures=number_list(value(i,key),key);
        }
        else if (key == "--smoke-test") c.smoke_test=true;
        else if (key == "--pilot") c.pilot=true;
        else if (key == "--progress") c.progress_interval_seconds=number<double>(value(i,key),key);
        else if (key == "--help" || key == "-h") {}
        else throw std::invalid_argument("Unknown option: " + std::string(key));
    }
    c.energy_grid_explicit = have_min && have_max && have_width;
    if (c.smoke_test) {
        c.nx=2; c.ny=2; c.nz=1; c.grid={-8.0,8.0,0.25}; c.energy_grid_explicit=true;
        c.max_mcs=2'500.0; c.max_limit_uses_mcs=true;
        c.wl.flatness=0.1; c.wl.minimum_visits=1; c.wl.final_factor=0.5;
        c.windows=1; c.walkers_per_rank=1; c.output_prefix="wl_smoke";
    }
    return c;
}

std::string usage(std::string_view program) {
    return "Usage: " + std::string(program) +
      " --emin E --emax E --bin-width dE [--nx N --ny N --nz N]\n"
      "  [--geometry file --box-x L --box-y L --box-z L] [--cutoff R]\n"
      "  [--config run.ini] (CLI options override INI values)\n"
      "  [--windows N --walkers N --overlap 0.75] [--seed N]\n"
      "  [--adaptive-windows true|false --adaptive-iterations N --adaptive-pilot-mcs MCS]\n"
      "  [--adaptive-smoothing-width dE --adaptive-minimum-width dE]\n"
      "  [--adaptive-diffusivity-floor F --adaptive-curvature-weight W]\n"
      "  [--adaptive-round-trip-target R --adaptive-round-trip-penalty P]\n"
      "  [--flatness 0.8 --min-visits 100 --final-factor 1e-8 --inverse-time true|false]\n"
      "  [--initialization-max-attempts N --initialization-target-fraction 0.5]\n"
      "  [--initialization-temperature-fraction 0.05]\n"
      "  [--initialization-stall-attempts-per-spin 1000]\n"
      "  [--initialization-temperature-multiplier 2 --initialization-max-temperature-fraction 0.5]\n"
      "  [--check-interval-mcs MCS --force-accept-after-mcs MCS]\n"
      "  [--exchange-interval-mcs MCS --max-mcs MCS --output prefix]\n"
      "  [--progress seconds] updates one console line; 0 disables progress.\n"
      "  MCS may be fractional; one MCS is N attempted single-spin flips per walker.\n"
      "  If --seed is omitted, a random seed is printed and saved in metadata.\n"
      "  Set --max-mcs 0 to run until convergence (not valid with --pilot).\n"
      "  Add --pilot to sample energies and print a non-rigorous grid suggestion.\n";
}

void write_dos_csv(const std::string& path, const DensityOfStates& dos) {
    if(dos.histogram.size()!=dos.log_g.size()||dos.standard_error.size()!=dos.log_g.size()||
       dos.valid.size()!=dos.log_g.size())
        throw std::invalid_argument("DOS output arrays have different sizes");
    std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path);
    out << "bin,energy,log_g,histogram,standard_error,valid\n" << std::setprecision(17);
    for (std::size_t i=0;i<dos.log_g.size();++i)
        out << i << ',' << dos.grid.center(i) << ',' << dos.log_g[i] << ','
            << dos.histogram[i] << ',' << dos.standard_error[i] << ','
            << static_cast<unsigned>(dos.valid[i]) << '\n';
}

void write_fragment_csv(const std::string& path, EnergyGrid grid, const DosFragment& f,
                        std::size_t window_id) {
    if(f.histogram.size()!=f.log_g.size()||f.standard_error.size()!=f.log_g.size()||
       f.valid.size()!=f.log_g.size()||f.window.end>f.log_g.size())
        throw std::invalid_argument("DOS fragment output arrays have different sizes");
    std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path);
    out << "window,bin,energy,log_g,histogram,standard_error,valid\n" << std::setprecision(17);
    for (auto i=f.window.begin;i<f.window.end;++i)
        out << window_id << ',' << i << ',' << grid.center(i) << ',' << f.log_g[i] << ','
            << f.histogram[i] << ',' << f.standard_error[i] << ','
            << static_cast<unsigned>(f.valid[i]) << '\n';
}

void write_thermodynamics_csv(const std::string& path, std::span<const ThermodynamicPoint> p) {
    std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path);
    out << "temperature,log_Z,U,C,F,S\n" << std::setprecision(17);
    for (const auto& x:p) out << x.temperature<<','<<x.log_partition<<','<<x.internal_energy<<','
                              <<x.heat_capacity<<','<<x.free_energy<<','<<x.entropy<<'\n';
}

void write_workers_stat_csv(const std::string& path,
                            std::span<const WalkerStatistics> statistics, std::size_t spin_count) {
    if(spin_count==0) throw std::invalid_argument("Cannot write MCS statistics for zero spins");
    std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path);
    out << "mpi_rank,window,walker_id,attempted_flips,attempted_mcs,accepted,forced_accepted,accepted_percent,"
           "attempts_since_last_accepted,mcs_since_last_accepted,energy,factor,active_bins,min_h,mean_h,min_over_mean,round_trips\n"
        << std::setprecision(17);
    for (const auto& walker : statistics) {
        const auto percent = walker.attempted == 0 ? 0.0 :
            100.0 * static_cast<double>(walker.accepted) / static_cast<double>(walker.attempted);
        out << walker.mpi_rank << ',' << walker.window_id << ',' << walker.walker_id << ','
            << walker.attempted << ','
            << static_cast<double>(walker.attempted)/static_cast<double>(spin_count) << ','
            << walker.accepted << ','
            << walker.forced_accepted << ',' << percent << ','
            << walker.attempts_since_last_accepted << ','
            << static_cast<double>(walker.attempts_since_last_accepted)/static_cast<double>(spin_count) << ','
            << walker.energy << ','
            << walker.factor << ',' << walker.active_bins << ','
            << walker.minimum_histogram << ',' << walker.mean_histogram << ','
            << walker.min_over_mean << ',' << walker.round_trips << '\n';
    }
}

void write_metadata_json(const std::string& path, const RunConfig& c, const Couplings& couplings,
                         std::uint64_t attempted, std::uint64_t accepted,
                         std::uint64_t forced_accepted,
                         std::uint64_t exchange_attempted, std::uint64_t exchange_accepted,
                         bool converged, int mpi_size, int omp_threads) {
    std::ofstream out(path); if (!out) throw std::runtime_error("Cannot write " + path);
    out << std::setprecision(17)
        << "{\n  \"format_version\": 4,\n  \"config_file\": \""<<json_escape(c.config_file)<<"\",\n"
        << "  \"geometry_file\": \""<<json_escape(c.geometry_file)<<"\",\n"
        << "  \"physics\": {\"model\": \"dipolar_ising\", "
        << "\"spins\": "<<couplings.size()<<", \"coupling_scale\": "<<c.coupling_scale<<", \"periodic\": "<<(c.periodic?"true":"false")
        << ", \"box\": ["<<c.box_lengths.x<<", "<<c.box_lengths.y<<", "<<c.box_lengths.z<<"]"
        << ", \"backend\": \""<<couplings.backend_name()<<"\", \"cutoff\": "<<c.cutoff<<"},\n"
        << "  \"energy_grid\": {\"minimum\": "<<c.grid.minimum<<", \"maximum\": "<<c.grid.maximum
        << ", \"bin_width\": "<<c.grid.width<<", \"complete_range\": "<<(c.complete_range?"true":"false")<<"},\n"
        << "  \"algorithm\": {\"variant\": \""<<(c.wl.inverse_time_enabled?"REWL-1/t":"REWL")
        << "\", \"windows\": "<<c.windows
        << ", \"walkers_per_rank\": "<<c.walkers_per_rank<<", \"overlap\": "<<c.overlap
        << ", \"flatness\": "<<c.wl.flatness<<", \"flatness_scope\": \"walker_local\""
        << ", \"initial_refinement_criterion\": \""
        <<(c.wl.inverse_time_enabled?
            (c.wl.nalivaiko_mod?"full_coverage_of_current_iteration_bins":
                                "full_coverage_of_discovered_bins"):
            (c.wl.nalivaiko_mod?"histogram_flatness_over_current_iteration_bins":
                                "histogram_flatness"))<<"\""
        << ", \"nalivaiko_mod\": "<<(c.wl.nalivaiko_mod?"true":"false")
        << ", \"refinement_active_scope\": \""
        <<(c.wl.nalivaiko_mod?"current_iteration":"cumulative_discovered_bins")<<"\""
        << ", \"inverse_time_clock\": \"walker_local_attempted_flips/active_bins\""
        << ", \"dos_synchronization\": \"none\""
        << ", \"refinement_schedule\": \"walker_independent\""
        << ", \"final_dos_combination\": \"aligned_log_mean\""
        << ", \"standard_error_kind\": \"within_run_walker_sem\""
        << ", \"minimum_visits_per_walker\": "<<c.wl.minimum_visits
        << ", \"check_interval_mcs\": "<<c.check_interval_mcs
        << ", \"check_interval_attempts\": "<<c.wl.check_interval_attempts
        << ", \"final_factor\": "<<c.wl.final_factor
        << ", \"force_accept_after_mcs\": "<<c.force_accept_after_mcs
        << ", \"inverse_time_enabled\": "<<(c.wl.inverse_time_enabled?"true":"false")
        << ", \"force_accept_after_attempts\": "<<c.wl.force_accept_after_attempts<<"},\n"
        << "  \"adaptive_windows\": {\"enabled\": "<<(c.adaptive_windows.enabled?"true":"false")
        << ", \"iterations\": "<<c.adaptive_windows.iterations
        << ", \"pilot_mcs\": "<<c.adaptive_windows.pilot_mcs
        << ", \"smoothing_width\": "<<c.adaptive_windows.smoothing_width
        << ", \"minimum_width\": "<<c.adaptive_windows.minimum_width
        << ", \"diffusivity_floor_fraction\": "<<c.adaptive_windows.diffusivity_floor_fraction
        << ", \"curvature_weight\": "<<c.adaptive_windows.curvature_weight
        << ", \"round_trip_target\": "<<c.adaptive_windows.round_trip_target
        << ", \"maximum_round_trip_penalty\": "<<c.adaptive_windows.maximum_round_trip_penalty
        << ", \"round_trip_margin_fraction\": "<<c.wl.round_trip_margin_fraction
        << ", \"external_warm_start_target_fraction\": 1"
        << ", \"energy_ranges\": [";
    const auto metadata_windows=c.explicit_windows.empty()?
        partition_windows(c.grid.bins(),c.windows,c.overlap):c.explicit_windows;
    for(std::size_t i=0;i<metadata_windows.size();++i) {
        if(i!=0) out<<", ";
        out<<'['<<c.grid.minimum+static_cast<double>(metadata_windows[i].begin)*c.grid.width
           <<", "<<c.grid.minimum+static_cast<double>(metadata_windows[i].end)*c.grid.width<<']';
    }
    out << "]},\n"
        << "  \"initialization\": {\"method\": \"adaptive_target_metropolis\""
        << ", \"max_attempts\": "<<c.wl.initialization_max_attempts
        << ", \"target_fraction\": "<<c.wl.initialization_target_fraction
        << ", \"temperature_fraction\": "<<c.wl.initialization_temperature_fraction
        << ", \"stall_attempts_per_spin\": "<<c.wl.initialization_stall_attempts_per_spin
        << ", \"temperature_multiplier\": "<<c.wl.initialization_temperature_multiplier
        << ", \"max_temperature_fraction\": "<<c.wl.initialization_max_temperature_fraction<<"},\n"
        << "  \"parallel\": {\"mpi_size\": "<<mpi_size<<", \"openmp_threads\": "<<omp_threads
        << ", \"exchange_interval_mcs\": "<<c.exchange_interval_mcs
        << ", \"exchange_interval_attempts\": "<<c.exchange_interval_attempts<<"},\n"
        << "  \"run_limits\": {\"max_mcs_per_walker\": "<<c.max_mcs<<", \"max_attempts_per_walker\": "<<c.max_attempts
        << ", \"checkpoint_interval_mcs\": "<<c.checkpoint_interval_mcs
        << ", \"checkpoint_interval_attempts\": "<<c.checkpoint_interval_attempts
        << ", \"progress_interval_seconds\": "<<c.progress_interval_seconds<<"},\n"
        << "  \"seed\": "<<c.seed<<",\n  \"seed_source\": \""<<(c.seed_explicit?"configured":"random")
        << "\",\n  \"attempted\": "<<attempted
        << ",\n  \"attempted_flips\": "<<attempted
        << ",\n  \"aggregate_mcs\": "<<static_cast<double>(attempted)/static_cast<double>(couplings.size())
        << ",\n  \"accepted\": "<<accepted
        << ",\n  \"forced_accepted\": "<<forced_accepted
        << ",\n  \"exchange_attempted\": "<<exchange_attempted
        << ",\n  \"exchange_accepted\": "<<exchange_accepted
        << ",\n  \"converged\": "<<(converged?"true":"false")<<"\n}\n";
}

void save_checkpoint(const std::string& path, const WalkerSnapshot& s) {
    const auto temporary = path + ".tmp";
    std::ofstream out(temporary, std::ios::binary|std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write checkpoint " + temporary);
    const std::array<char,8> magic{'W','L','C','H','K','P','3','\0'};
    out.write(magic.data(), magic.size()); write_value(out,s.walker_id);
    write_vector(out,s.spins); write_vector(out,s.fields); write_value(out,s.energy);
    write_vector(out,s.log_g); write_vector(out,s.histogram); write_vector(out,s.active);
    write_value(out,s.factor); write_value(out,s.attempted); write_value(out,s.accepted);
    write_value(out,s.forced_accepted);
    write_value(out,s.last_accepted_attempt);
    const auto stage=static_cast<std::uint8_t>(s.stage); write_value(out,stage);
    for(const auto word:s.rng_state) write_value(out,word);
    out.close(); if(!out) throw std::runtime_error("Failed writing checkpoint");
    std::error_code error; std::filesystem::remove(path,error); error.clear();
    std::filesystem::rename(temporary,path,error);
    if(error) throw std::runtime_error("Cannot atomically replace checkpoint: "+error.message());
}

WalkerSnapshot load_checkpoint(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("Cannot open checkpoint "+path);
    std::array<char,8> magic{}; in.read(magic.data(),magic.size());
    const auto version=std::string_view(magic.data(),7);
    if(version!="WLCHKP1" && version!="WLCHKP2" && version!="WLCHKP3")
        throw std::runtime_error("Invalid checkpoint format");
    WalkerSnapshot s; read_value(in,s.walker_id); read_vector(in,s.spins); read_vector(in,s.fields);
    read_value(in,s.energy); read_vector(in,s.log_g); read_vector(in,s.histogram); read_vector(in,s.active);
    read_value(in,s.factor); read_value(in,s.attempted); read_value(in,s.accepted);
    if(version=="WLCHKP3") read_value(in,s.forced_accepted);
    if(version=="WLCHKP2" || version=="WLCHKP3") read_value(in,s.last_accepted_attempt);
    else s.last_accepted_attempt=s.accepted==0?0:s.attempted;
    std::uint8_t stage{}; read_value(in,stage); s.stage=static_cast<RefinementStage>(stage);
    for(auto& word:s.rng_state) read_value(in,word); return s;
}

} // namespace wl
