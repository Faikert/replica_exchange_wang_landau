#include "wl/io.hpp"
#include <exception>
#include <iostream>
#include <filesystem>

int main(int argc,char** argv) {
    try {
        for(int i=1;i<argc;++i) if(std::string_view(argv[i])=="--help"||std::string_view(argv[i])=="-h") {
            std::cout<<wl::usage(argv[0]); return 0;
        }
        auto c=wl::parse_arguments(argc,argv); c.validate(true);
        wl::Geometry geometry;
        if(c.geometry_file.empty()) geometry=wl::Geometry::simple_cubic(c.nx,c.ny,c.nz,c.spacing,c.axis,c.periodic);
        else if(std::filesystem::path(c.geometry_file).extension()==".csv")
            geometry=wl::Geometry::load_csv(c.geometry_file,{c.box_lengths,c.periodic});
        else geometry=wl::Geometry::load_xyz_axes(c.geometry_file,{c.box_lengths,c.periodic});
        std::unique_ptr<wl::Couplings> couplings;
        if(c.cutoff>0) couplings=std::make_unique<wl::CsrCouplings>(geometry,c.coupling_scale,c.cutoff);
        else couplings=std::make_unique<wl::DenseCouplings>(geometry,c.coupling_scale);
        const auto exact=wl::exact_enumeration(*couplings,c.grid);
        wl::write_fragment_csv(c.output_prefix+"_exact.csv",c.grid,exact,0);
        std::cout<<"enumerated "<<(std::uint64_t{1}<<geometry.size())<<" states\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<"wl_exact: "<<error.what()<<'\n'; return 1; }
}
