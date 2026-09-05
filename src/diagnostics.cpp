#include "wl/io.hpp"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace wl {
const char* stage_name(RefinementStage stage) noexcept {
    switch(stage) {
    case RefinementStage::wang_landau: return "WL";
    case RefinementStage::inverse_time: return "1t";
    case RefinementStage::frozen: return "frozen";
    }
    return "unknown";
}

WalkerStatistics collect_walker_statistics(const WangLandauWalker& walker,
                                            std::size_t window, int mpi_rank) {
    const auto h=walker.histogram_statistics();
    WalkerStatistics s{walker.id(),static_cast<std::uint64_t>(window),mpi_rank,
        walker.attempted(),walker.accepted(),walker.forced_accepted(),
        walker.attempted()-walker.last_accepted_attempt(),walker.energy(),walker.factor(),
        h.active_bins,h.minimum,h.mean,h.min_over_mean,walker.round_trips()};
    s.stage=walker.stage(); s.covered_bins=h.covered_bins; s.coverage=h.coverage;
    for(std::size_t cell=0;cell<walker.active_mask().size();++cell) if(walker.active_mask()[cell]) {
        ++s.cumulative_active_bins;
        if(walker.histogram()[cell]) ++s.cumulative_covered_bins;
    }
    s.attempts_since_last_iteration=walker.attempts_since_last_iteration();
    s.seconds_since_last_iteration=walker.seconds_since_last_iteration();
    s.initialization_attempts=walker.initialization_attempts();
    s.initialization_restarts=walker.initialization_restarts();
    s.initialization_seconds=walker.initialization_seconds();
    s.returns_enabled=walker.returns_enabled();
    if(s.returns_enabled) s.return_reference_energy=walker.return_reference_energy();
    s.return_count=walker.return_count();
    return s;
}

std::vector<std::size_t> missing_histogram_cells(const WangLandauWalker& walker) {
    std::vector<std::size_t> missing;
    for(std::size_t cell=0;cell<walker.active_mask().size();++cell)
        if(walker.active_mask()[cell] && !walker.histogram()[cell]) missing.push_back(cell);
    return missing;
}

void write_missing_bins_csv(const std::string& path, std::span<const MissingBin> bins, DosGrid grid) {
    grid.validate();
    std::ofstream out(path); if(!out) throw std::runtime_error("Cannot write "+path);
    out << "mpi_rank,window,walker_id,cell,energy_bin,energy,q_bin,q\n" << std::setprecision(17);
    for(const auto& b:bins) {
        if(b.cell>=grid.cells()) throw std::invalid_argument("Diagnostic cell outside DOS grid");
        const auto e=grid.energy_bin(b.cell);
        out << b.mpi_rank << ',' << b.window_id << ',' << b.walker_id << ',' << b.cell
            << ',' << e << ',' << grid.energy.center(e) << ',';
        if(grid.joint()) {
            const auto q=grid.q_bin(b.cell);
            out << q << ',' << grid.order_parameter->center(q);
        } else out << ',';
        out << '\n';
    }
}

void write_exchange_stat_csv(const std::string& path, std::span<const ExchangeStatistics> statistics) {
    std::ofstream out(path); if(!out) throw std::runtime_error("Cannot write "+path);
    out << "left_window,right_window,attempted,accepted,accepted_percent\n" << std::setprecision(17);
    for(std::size_t b=0;b<statistics.size();++b) {
        const auto& s=statistics[b];
        out << b << ',' << b+1 << ',' << s.attempted << ',' << s.accepted << ','
            << (s.attempted?100.0*static_cast<double>(s.accepted)/static_cast<double>(s.attempted):0.0) << '\n';
    }
}
} // namespace wl
