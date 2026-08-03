#include "wl/io.hpp"

#include <fstream>
#include <iostream>
#include <cmath>
#include <limits>
#include <sstream>

namespace {
wl::DosFragment read_fragment(const std::string& path,const wl::EnergyGrid& grid) {
    std::ifstream in(path); if(!in) throw std::runtime_error("Cannot open "+path);
    const auto nan=std::numeric_limits<double>::quiet_NaN();
    wl::DosFragment f{{grid.bins(),0},std::vector<double>(grid.bins(),nan),
                      std::vector<std::uint64_t>(grid.bins(),0),
                      std::vector<double>(grid.bins(),nan),
                      std::vector<std::uint8_t>(grid.bins(),0),
                      wl::DosGrid{grid,std::nullopt},
                      std::vector<std::uint32_t>(grid.bins(),0),
                      std::vector<std::int32_t>(grid.bins(),-1)};
    std::string line; std::getline(in,line);
    while(std::getline(in,line)) {
        std::stringstream row(line); std::string value; std::vector<std::string> fields;
        while(std::getline(row,value,',')) fields.push_back(value);
        if(fields.size()<6) continue;
        const auto bin=static_cast<std::size_t>(std::stoull(fields[1]));
        f.window.begin=std::min(f.window.begin,bin); f.window.end=std::max(f.window.end,bin+1);
        f.log_g[bin]=std::stod(fields[3]); f.histogram[bin]=std::stoull(fields[4]);
        f.standard_error[bin]=std::stod(fields[5]);
        f.valid[bin]=fields.size()>=7?static_cast<std::uint8_t>(std::stoul(fields[6])!=0):
            static_cast<std::uint8_t>(std::isfinite(f.log_g[bin])&&f.histogram[bin]!=0);
        f.contributors[bin]=fields.size()>=8?static_cast<std::uint32_t>(std::stoul(fields[7])):1;
        f.support_component[bin]=fields.size()>=9?static_cast<std::int32_t>(
            std::stol(fields[8])):0;
        if(f.valid[bin]==0) {
            f.log_g[bin]=nan;
            f.standard_error[bin]=nan;
            f.contributors[bin]=0;
            f.support_component[bin]=-1;
        }
    }
    if(f.window.begin>=f.window.end) throw std::runtime_error("Empty fragment "+path);
    return f;
}
}

int main(int argc,char** argv) {
    try {
        if(argc<8) {
            std::cerr<<"Usage: wl_analyze E_min E_max bin_width spins complete(0|1) output fragment...\n";
            return 1;
        }
        wl::EnergyGrid grid{std::stod(argv[1]),std::stod(argv[2]),std::stod(argv[3])};
        const auto spins=static_cast<std::size_t>(std::stoull(argv[4])); const bool complete=std::stoi(argv[5])!=0;
        const std::string prefix=argv[6]; std::vector<wl::DosFragment> fragments;
        for(int i=7;i<argc;++i) fragments.push_back(read_fragment(argv[i],grid));
        const auto dos=wl::stitch_dos(grid,fragments,complete,spins);
        wl::write_dos_csv(prefix+"_dos.csv",dos);
        const std::vector<double> temperatures{0.25,0.5,1.0,2.0,5.0};
        wl::write_thermodynamics_csv(prefix+"_thermo.csv",wl::thermodynamics(dos,temperatures));
        return 0;
    } catch(const std::exception& error) { std::cerr<<"wl_analyze: "<<error.what()<<'\n'; return 1; }
}
