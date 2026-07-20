#include "wl/physics.hpp"
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
        const auto geometry=wl::Geometry::simple_cubic(n,n,1);
        std::unique_ptr<wl::Couplings> couplings;
        if(backend=="dense") couplings=std::make_unique<wl::DenseCouplings>(geometry,1.0);
        else if(backend=="csr") couplings=std::make_unique<wl::CsrCouplings>(geometry,1.0,cutoff);
        else throw std::invalid_argument("backend must be dense or csr");
        std::vector<std::int8_t> spins(geometry.size(),1);
        auto fields=wl::local_fields(*couplings,spins); wl::Xoshiro256StarStar rng(1); std::uint64_t accepted=0;
        const auto start=std::chrono::steady_clock::now();
        for(std::uint64_t step=0;step<attempts;++step) {
            const auto i=static_cast<std::size_t>(rng.bounded(spins.size())); const auto old=spins[i];
            const auto delta=wl::flip_delta(i,spins,fields);
            if(delta<=0.0||rng.uniform()<std::exp(-delta)) {
                couplings->add_flip_delta(i,old,fields); spins[i]=-old; ++accepted;
            }
        }
        const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout<<"backend="<<couplings->backend_name()<<" spins="<<spins.size()<<" attempted="<<attempts<<" accepted="<<accepted
                 <<" attempted_mcs="<<static_cast<double>(attempts)/static_cast<double>(spins.size())
                 <<" seconds="<<seconds<<" attempted_flips_per_second="<<attempts/seconds
                 <<" mcs_per_second="<<attempts/(seconds*static_cast<double>(spins.size()))<<'\n';
    } catch(const std::exception& error) { std::cerr<<"wl_bench: "<<error.what()<<'\n'; return 1; }
}
