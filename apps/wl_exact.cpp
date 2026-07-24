#include "wl/io.hpp"
#include <exception>
#include <iostream>
#include <filesystem>

int main(int argc,char** argv) {
    try {
        for(int i=1;i<argc;++i) if(std::string_view(argv[i])=="--help"||std::string_view(argv[i])=="-h") {
            std::cout<<wl::usage(argv[0]); return 0;
        }
        auto c=wl::parse_arguments(argc,argv);
        wl::Geometry geometry;
        if(c.geometry_file.empty()) geometry=wl::Geometry::simple_cubic(c.nx,c.ny,c.nz,c.spacing,c.axis,c.periodic);
        else if(std::filesystem::path(c.geometry_file).extension()==".csv")
            geometry=wl::Geometry::load_csv(c.geometry_file,{c.box_lengths,c.periodic});
        else geometry=wl::Geometry::load_xyz_axes(c.geometry_file,{c.box_lengths,c.periodic});
        c.resolve_order_parameter(geometry); c.resolve_mcs(geometry.size()); c.validate(true);
        std::unique_ptr<wl::Couplings> couplings;
        if(c.cutoff>0) couplings=std::make_unique<wl::CsrCouplings>(geometry,c.coupling_scale,c.cutoff);
        else couplings=std::make_unique<wl::DenseCouplings>(geometry,c.coupling_scale);
        if(c.order_parameter) {
            const auto exact=wl::exact_enumeration(*couplings,c.dos_grid(),*c.order_parameter);
            const auto joint=wl::stitch_joint_dos(c.dos_grid(),std::span(&exact,1),true,true,
                                                  geometry.size(),c.order_parameter->normalization);
            const auto marginal=wl::marginalize(joint);
            wl::write_joint_dos_csv(c.output_prefix+"_dos2d.csv",joint);
            wl::write_dos_csv(c.output_prefix+"_dos.csv",marginal);
            wl::write_thermodynamics_csv(c.output_prefix+"_thermo.csv",
                                         wl::thermodynamics(marginal,c.temperatures));
            wl::write_order_parameter_thermodynamics_csv(c.output_prefix+"_q_thermo.csv",
                wl::order_parameter_thermodynamics(joint,c.temperatures,geometry.size()));
            wl::write_order_parameter_distribution_csv(c.output_prefix+"_q_distribution.csv",
                wl::order_parameter_distribution(joint,c.temperatures));
        } else {
            const auto exact=wl::exact_enumeration(*couplings,c.grid);
            wl::write_fragment_csv(c.output_prefix+"_exact.csv",c.grid,exact,0);
        }
        std::cout<<"enumerated "<<(std::uint64_t{1}<<geometry.size())<<" states\n";
        return 0;
    } catch(const std::exception& error) { std::cerr<<"wl_exact: "<<error.what()<<'\n'; return 1; }
}
