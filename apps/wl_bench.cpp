#include "wl/wang_landau.hpp"
#include "wl/rng.hpp"

#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

int main(int argc,char** argv) {
    try {
        const auto n=argc>1?static_cast<std::size_t>(std::stoull(argv[1])):16;
        const auto attempts=argc>2?std::stoull(argv[2]):1'000'000ULL;
        const std::string backend=argc>3?argv[3]:"dense";
        const auto cutoff=argc>4?std::stod(argv[4]):3.0;
        const bool joint=argc>5&&std::string(argv[5])=="joint";
        const auto geometry=wl::Geometry::simple_cubic(n,n,1);
        std::unique_ptr<wl::Couplings> couplings;
        if(backend=="dense") couplings=std::make_unique<wl::DenseCouplings>(geometry,1.0);
        else if(backend=="csr") couplings=std::make_unique<wl::CsrCouplings>(geometry,1.0,cutoff);
        else throw std::invalid_argument("backend must be dense or csr");
        std::vector<std::int8_t> spins(geometry.size(),1);
        auto fields=wl::local_fields(*couplings,spins); wl::Xoshiro256StarStar rng(1); std::uint64_t accepted=0;
        auto energy=wl::energy_from_fields(spins,fields);
        std::vector<double> q_weights(spins.size(),1.0);
        for(std::size_t i=1;i<q_weights.size();i+=2) q_weights[i]=-1.0;
        const auto order=wl::WeightedOrderParameter::create(q_weights,1.0);
        const wl::DosGrid dos_grid{{-1.0e9,1.0e9,1.0e6},joint?
            std::optional<wl::OrderParameterGrid>(order.grid):std::nullopt};
        auto q_value=joint?order.evaluate(spins):0.0;
        std::size_t cell_checksum=0;
        const auto start=std::chrono::steady_clock::now();
        for(std::uint64_t step=0;step<attempts;++step) {
            const auto i=static_cast<std::size_t>(rng.bounded(spins.size())); const auto old=spins[i];
            const auto delta=wl::flip_delta(i,spins,fields);
            const auto q_delta=joint?order.flip_delta(i,old):0.0;
            if(joint) if(const auto cell=dos_grid.index(energy+delta,q_value+q_delta))
                cell_checksum^=*cell;
            if(delta<=0.0||rng.uniform()<std::exp(-delta)) {
                couplings->add_flip_delta(i,old,fields); spins[i]=-old; energy+=delta;
                q_value+=q_delta; ++accepted;
            }
        }
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"backend="<<couplings->backend_name()<<" spins="<<spins.size()<<" attempted="<<attempts<<" accepted="<<accepted
                 <<" joint="<<(joint?"yes":"no")<<" dos_cells="<<dos_grid.cells()
                 <<" cell_checksum="<<cell_checksum
                 <<" attempted_mcs="<<static_cast<double>(attempts)/static_cast<double>(spins.size())
                 <<" seconds="<<seconds<<" attempted_flips_per_second="
                 <<static_cast<double>(attempts)/seconds
                 <<" mcs_per_second="<<static_cast<double>(attempts)/
                    (seconds*static_cast<double>(spins.size()))<<'\n';
    } catch(const std::exception& error) { std::cerr<<"wl_bench: "<<error.what()<<'\n'; return 1; }
}
