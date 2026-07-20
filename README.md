# Dipolar Wang–Landau

C++20 implementation of single-walker WL and replica-exchange Wang–Landau (REWL) for Ising dipoles. The implementation uses independent density-of-states estimators, overlapping energy windows, replica exchange, and the Belardinelli–Pereyra `1/t` refinement stage.

## Physics

Each site has a position `r_i`, a unit easy axis `e_i`, and `sigma_i = +/-1`. The implemented pair coupling is

```
J_ij = D [e_i.e_j - 3(e_i.rhat_ij)(e_j.rhat_ij)] / r_ij^3
E    = sum_{i<j} J_ij sigma_i sigma_j.
```

For periodic orthorhombic cells, displacement vectors use the minimum-image convention. A positive box length enables periodicity along that axis; a zero length leaves that axis open. For example, `periodic=true`, `box_x>0`, `box_y>0`, `box_z=0` defines slab geometry periodic only in x and y. This is a truncated periodic model, not an Ewald sum. Set `--cutoff R` for a sparse CSR neighbor backend; without it the dense minimum-image matrix contains every pair. Accepted flips update local fields incrementally in `O(N)` (dense) or `O(z)` (CSR).

Dipolar energies are generally not commensurate. `--emin`, `--emax`, and `--bin-width` therefore define the coarse-grained DOS and are mandatory. A range must only be marked complete with `--complete-range true` when it is known to include every configuration; otherwise output is a relative DOS with `max(log_g)=0`.

`wl_run --pilot --nx ... --max-mcs ...` performs an infinite-temperature random walk and prints sampled bounds plus 5% padding. These are only starting suggestions and are explicitly not proof of the full range.

## Build

```powershell
cmake -S . -B build -DWL_ENABLE_OPENMP=ON -DWL_ENABLE_MPI=ON
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

MPI and OpenMP are discovered independently. If MPI is unavailable, serial and OpenMP programs still build. No third-party C++ library is required.

## Run

The recommended input uses a CSV system file with the exact header `x,y,z,mx,my,mz`. The `m` columns are unit local Ising axes; non-unit vectors are rejected rather than silently rescaled. Empty lines and lines beginning with `#` are allowed.

All run parameters can be placed in an INI file. Paths in `geometry.file` are resolved relative to the INI file, and command-line options override INI values:

```powershell
build/Release/wl_run.exe --config examples/run.ini
build/Release/wl_run.exe --config examples/run.ini --seed 42 --output overridden
```

If `seed` is omitted from both the INI file and the command line, MPI rank 0 generates a random seed, broadcasts it to every rank, prints it to the console, and saves it in the metadata. Set `seed` explicitly when an exactly reproducible run is required.

See `examples/system.csv` and `examples/run.ini` for the supported sections and keys.

User-facing work intervals are expressed in Monte Carlo sweeps (MCS): one MCS is `N` attempted single-spin flips per walker, with spins selected randomly with replacement. Fractional MCS values are allowed and resolve to the nearest positive integer number of flip attempts. The preferred keys are `exchange_interval_mcs`, `check_interval_mcs`, `force_accept_after_mcs`, `max_mcs`, and `checkpoint_interval_mcs`. Legacy attempt-based keys remain accepted with a warning and retain their old meaning.

Exact attempted-flip counts are still stored because histogram updates, checkpoint restart, and the internal `1/t` refinement use one update per proposal. Metadata and runtime output contain both MCS and resolved flip-attempt counts.

Use `--progress 2` or `[run] progress = 2` to refresh a single console line approximately every two seconds; `progress = 0` disables it. The line contains MCS and exact attempted flips per walker, the minimum last-checked flatness among walkers still in the WL stage, the largest current modification factor, and `WL/1t/frozen` walker counts. The timer is checked between exchange batches, so a long `exchange_interval_mcs` can delay an update. MPI mode aggregates the values and only rank 0 writes the line.

```powershell
build/Release/wl_run.exe --nx 2 --ny 2 --nz 2 `
  --emin -20 --emax 20 --bin-width 0.25 `
  --windows 1 --walkers 2 --seed 1234 --output run
```

For MPI, the number of ranks must be a multiple of `--windows`. Each rank owns `--walkers` OpenMP walkers. Ranks belonging to one window use an MPI subcommunicator for DOS averaging; corresponding ranks in adjacent windows exchange replicas using an even/odd schedule. MPI calls are made by the master thread under `MPI_THREAD_FUNNELED`.

The program writes per-window CSV files, a stitched DOS with a within-run standard error across walkers, thermodynamic observables, and JSON metadata including exchange and forced acceptance. Each DOS CSV has a `valid` column; only bins visited by every walker in the corresponding window contribute to the combined DOS. A one-walker run reports `standard_error=nan`. This uncertainty describes dispersion inside one REWL run and does not replace an ensemble of independent master seeds. If a run stops with `converged=no`, it also writes `${output_prefix}_workers_stat.csv` with each walker's attempted flips and MCS, flip acceptance, forced-acceptance count, age of the last accepted flip in both units, final energy, modification factor, active-bin count, and `min(H)/mean(H)` diagnostics. A checkpoint is supported for a single-window/single-walker run using `--checkpoint path`; its RNG state and forced-acceptance count are included.

Use exact enumeration for small validation systems:

```powershell
build/Release/wl_exact.exe --nx 2 --ny 2 --nz 2 `
  --emin -20 --emax 20 --bin-width 0.25 --output exact
```

Stitch previously generated fragment CSV files with `wl_analyze`. Measure incremental-update throughput with `wl_bench [linear_size] [attempts] [dense|csr] [cutoff]`; it reports both attempted flips/s and MCS/s. Repeat under the intended build type and affinity settings for publishable numbers.

## Algorithm notes

- Every walker owns `log_g`, `H`, its cumulative active mask, modification factor, and refinement stage. Walkers never copy or average DOS during sampling.
- Unless an explicit in-window spin configuration is supplied through the C++ API, initialization uses an adaptive target Metropolis chain with `pi(E) proportional to exp(-abs(E-E_target)/T_search)`. `E_target` is the window center, the default accepted target band is the central 50% of the window, and the initial `T_search` is 5% of the window width (never below one energy-bin width). After `stall_attempts_per_spin*N` proposals without a closer energy, `T_search` is multiplied by `temperature_multiplier` up to `max_temperature_fraction` of the window width. Another stall at the maximum temperature reproducibly randomizes the spins and restarts at the initial temperature. The defaults are `1000`, `2`, and `0.5`, respectively. Configure these values in `[initialization]` or with the corresponding `--initialization-*` CLI options. Initialization proposals and restarts do not update WL counters; only the selected starting bin receives the initial `H` and `log_g` update.
- Initial iterations use `ln(f)=1` and `ln(f) <- ln(f)/2`. Each walker checks flatness independently every `check_interval_mcs` over its own cumulative active mask and begins its next iteration without waiting for the other walkers in the window.
- By default, each walker changes independently to `1/t` when `ln(f) <= 1/t`, with `t=attempted_flips/active_bins`; this internal counter is deliberately not converted to MCS. Set `[wl] inverse_time = false` or pass `--inverse-time false` to keep the traditional Wang–Landau schedule `ln(f) <- ln(f)/2` until `final_factor`. The active-bin count is cached and updated only on first visits and checkpoint restore, so a `1/t` flip remains `O(1)` apart from the coupling update.
- After sampling, window DOS estimates are aligned at the first bin in the intersection of all walker active masks and averaged in log space. Bins outside that intersection are written as `valid=0` with `log_g=nan`; adjacent windows must share at least one valid overlap bin.
- `force_accept_after_mcs=MCS` (or `--force-accept-after-mcs MCS`) forces the next otherwise rejected proposal that remains inside the walker's energy window after that many MCS without an accepted flip. The default `0` disables it. This is a non-standard escape heuristic that can bias the DOS; its use and actual forced-acceptance count are saved in metadata.
- Set `max_mcs=0` (or `--max-mcs 0`) to disable the work limit and run until every walker reaches `final_factor`. Pilot mode still requires a finite positive limit.
- Adjacent windows overlap by 75% by default. DOS fragments are joined where their local linear estimates of `d log(g)/dE` agree best.
- A run that reaches `--max-mcs` before `--final-factor` is marked `converged=false`; it is never silently presented as converged.

CUDA/HIP, Ewald summation, continuous kernel DOS, and joint `g(E,M)` are intentionally outside this first CPU implementation.
