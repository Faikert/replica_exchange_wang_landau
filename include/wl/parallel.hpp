#pragma once
#include "wl/io.hpp"
#include <memory>
#include <span>
#include <vector>
namespace wl {
class ParallelContext {
public:
    ParallelContext(int& argc, char**& argv);
    ~ParallelContext();
    ParallelContext(const ParallelContext&) = delete;
    ParallelContext& operator=(const ParallelContext&) = delete;
    [[nodiscard]] int rank() const noexcept { return rank_; }
    [[nodiscard]] int size() const noexcept { return size_; }
    [[nodiscard]] std::uint64_t broadcast_seed(std::uint64_t seed) const;
    void broadcast_windows(std::vector<EnergyWindow>& windows) const;
    void broadcast_spin_configurations(
        std::vector<std::vector<std::int8_t>>& configurations) const;
    [[nodiscard]] bool mpi_enabled() const noexcept { return mpi_enabled_; }
private:
    int rank_{0}; int size_{1}; bool mpi_enabled_{false}; bool owns_mpi_{false};
};
struct RewlResult {
    std::vector<DosFragment> fragments;
    std::vector<WalkerStatistics> walker_statistics;
    std::vector<MissingBin> missing_bins;
    std::vector<ExchangeStatistics> exchange_statistics;
    std::vector<WindowSamplingStatistics> sampling_statistics;
    std::vector<EnergyRepresentative> representatives;
    std::uint64_t attempted{}, accepted{}, forced_accepted{};
    std::uint64_t exchange_attempted{}, exchange_accepted{};
    bool converged{false};
};
[[nodiscard]] DosFragment summarize_walkers(
    EnergyWindow window,std::span<const WangLandauWalker* const> walkers,std::size_t bins,
    bool allow_empty_intersection=false);
[[nodiscard]] RewlResult run_rewl(const ParallelContext&, std::shared_ptr<const Couplings>,
                                  const RunConfig&);
[[nodiscard]] int maximum_openmp_threads() noexcept;
} // namespace wl
