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

### Joint density of states g(E,Q)

Set `[order_parameter] mode = weighted_sum` to sample the signed joint DOS for
`Q = sum_i q_weight_i*sigma_i`. The geometry CSV must then contain the seventh column
`q_weight`. Its `bin_width` defines the coarse-grained Q resolution; the symmetric range is
constructed automatically from `W = sum_i |q_weight_i|`, and output also contains `q=Q/W`.
Energy windows still partition only E.

```powershell
build/Release/wl_run.exe --config examples/run_2d.ini
build/Release/wl_exact.exe --config examples/run_2d.ini --output exact_2d
```

The 2D run writes `_dos2d.csv`, the marginal `_dos.csv`, `_q_thermo.csv`,
`_q_distribution.csv`, and per-window `_window_N_dos2d.csv` files. The Q thermodynamics
include signed and absolute moments, `chi_q=N(<q^2>-<q>^2)/(k_B T)`, and the Binder
cumulant `U4=1-<q^4>/(3<q^2>^2)`. Moments use Q-bin centers. Joint CSV files also report
the number of contributing walkers and the connected support component. Walkers are aligned
by weighted least squares over their shared cells; cells visited by at least one aligned walker
form the relaxed-union estimate. If the support graph is disconnected, window fragments,
metadata, and walker diagnostics are still written, but a global DOS and thermodynamics are not.
All DOS CSV files are sparse: bins with `valid=0` are omitted, while the original global bin
indices are retained. Exact-enumeration structural zeros remain present because they are known
parts of the support (`valid=1`, `log_g=-inf`), rather than unvisited sampling bins.

Both ordinary and joint DOS use the same relaxed-union policy. Walkers are aligned by global
histogram-weighted least squares over shared finite bins, and a bin remains valid when at least
one aligned walker visited it. The 1D DOS and window CSV files therefore also contain
`contributors` and `support_component`. A newly discovered DOS bin delays the next factor
reduction for `[wl] support_stability_checks` histogram-check intervals (default 10). The old
`[order_parameter]` location of this key remains accepted for compatibility.

With `nalivaiko_mod=false`, inverse-time coverage is evaluated over all cumulatively
discovered bins: after each histogram reset every known bin must be visited again. With
`nalivaiko_mod=true`, both flatness and coverage use only bins visited in the current
iteration. Coverage is therefore normally one, while the cumulative active mask still preserves
rare bins for relaxed-union stitching, DOS validity, output, and the `1/t` active-cell count.

User-facing work intervals are expressed in Monte Carlo sweeps (MCS): one MCS is `N` attempted single-spin flips per walker, with spins selected randomly with replacement. Fractional MCS values are allowed and resolve to the nearest positive integer number of flip attempts. The preferred keys are `exchange_interval_mcs`, `check_interval_mcs`, `force_accept_after_mcs`, `max_mcs`, and `checkpoint_interval_mcs`. Legacy attempt-based keys remain accepted with a warning and retain their old meaning.

Exact attempted-flip counts are still stored because histogram updates, checkpoint restart, and the internal `1/t` refinement use one update per proposal. Metadata and runtime output contain both MCS and resolved flip-attempt counts.

Use `--progress 2` or `[run] progress = 2` to refresh a single console line approximately every two seconds; `progress = 0` disables it. With `inverse_time=true`, the line reports the minimum current-stage coverage of each walker's cumulative discovered energy support; traditional WL runs report histogram flatness instead. It also contains MCS, exact attempted flips per walker, the largest current modification factor, and `WL/1t/frozen` walker counts. The timer is checked between exchange batches, so a long `exchange_interval_mcs` can delay an update. MPI mode aggregates the values and only rank 0 writes the line.

```powershell
build/Release/wl_run.exe --nx 2 --ny 2 --nz 2 `
  --emin -20 --emax 20 --bin-width 0.25 `
  --windows 1 --walkers 2 --seed 1234 --output run
```

For MPI, the number of ranks must be a multiple of `--windows`. Each rank owns `--walkers` OpenMP walkers. Ranks belonging to one window use an MPI subcommunicator for DOS averaging; corresponding ranks in adjacent windows exchange replicas using an even/odd schedule. MPI calls are made by the master thread under `MPI_THREAD_FUNNELED`.

### Adaptive energy windows

Set `[adaptive_windows] enabled = true` or pass `--adaptive-windows true` to optimize window boundaries before production. Each pilot records the accepted squared energy displacement at the proposal energy, complete low-high-low trips through the interior of each window, and the local curvature of `log_g(E)`. The optimizer constructs the continuous difficulty density

```
w(E) = round_trip_penalty(E) * [1 + curvature_weight*C(E)] / sqrt(max(D(E), D_floor))
```

and places equal difficulty mass in every core energy interval. Low-diffusivity or high-curvature regions therefore receive narrower windows. Overlaps are added afterward using the configured fractional `parallel.overlap`. `smoothing_width` and `minimum_width` are physical energy differences and are independent of the DOS grid resolution; zero selects automatic values from the full energy span. Each pilot also retains a small bank of actually visited spin configurations distributed across every old energy window; the first bank entry is reserved for the minimum energy visited in that window. New windows and the final production run are seeded from the closest bank configurations, except that every walker in the lower edge window is seeded from the global minimum-energy bank configuration. A configuration already inside the new range is used directly. If the bank has no configuration inside, the closest external configuration is used as a warm start and target Metropolis only has to enter the new window, rather than reach its central target fraction. Runtime output reports both `seeded_walkers` and `external_warm_starts`. After every pilot, all pilot `log_g` and histogram data are discarded. Production starts with fixed adapted boundaries and zeroed DOS estimators, so neither the pilot DOS nor its histogram enters the production estimate.

```ini
[adaptive_windows]
enabled = true
iterations = 2
pilot_mcs = 10000
smoothing_width = 5.0
minimum_width = 40.0
diffusivity_floor_fraction = 0.05
curvature_weight = 0.25
round_trip_target = 2
maximum_round_trip_penalty = 3
round_trip_margin_fraction = 0.1
```

If `smoothing_width` or `minimum_width` is zero, the automatic choice is recorded as zero in the configuration metadata while the final continuous energy ranges are recorded explicitly. A pilot should be long enough for ordinary windows to complete several trips. Windows with no completed trip receive the maximum configured difficulty penalty. Adaptation is deterministic for a fixed master seed.

The program writes per-window CSV files, a stitched DOS with a within-run standard error across walkers, thermodynamic observables, and JSON metadata including exchange and forced acceptance. The absolute output prefix is printed before sampling. The original 1D validity rule remains unchanged: only bins visited by every walker in the corresponding window contribute. A one-walker run reports `standard_error=nan`. This uncertainty describes dispersion inside one REWL run and does not replace an ensemble of independent master seeds. If a run stops with `converged=no`, it also writes `${output_prefix}_workers_stat.csv` with each walker's attempted flips and MCS, flip acceptance, forced-acceptance count, age of the last accepted flip in both units, final energy, modification factor, active-bin count, `min(H)/mean(H)`, and completed energy round trips. Metadata, worker statistics, and raw window fragments are written even when the walkers or neighboring windows have insufficient common support for a global DOS; metadata then records `postprocessing_status=insufficient_support` or `disconnected_support`. WLCHKP5 checkpoint/restart is supported only for a single MPI process, single window, and single walker; it validates the complete E/Q layout and order-parameter weights before a transactional restore. Legacy checkpoints are accepted only in 1D with reduced layout verification.

Use exact enumeration for small validation systems:

```powershell
build/Release/wl_exact.exe --nx 2 --ny 2 --nz 2 `
  --emin -20 --emax 20 --bin-width 0.25 --output exact
```

Stitch previously generated fragment CSV files with `wl_analyze`. Measure incremental-update throughput with `wl_bench [linear_size] [attempts] [dense|csr] [cutoff]`; it reports both attempted flips/s and MCS/s. Repeat under the intended build type and affinity settings for publishable numbers.

## Algorithm notes

- Every walker owns `log_g`, `H`, its cumulative active mask, modification factor, and refinement stage. Walkers never copy or average DOS during sampling.
- Unless an explicit in-window spin configuration is supplied through the C++ API, initialization uses an adaptive target Metropolis chain with `pi(E) proportional to exp(-abs(E-E_target)/T_search)`. `E_target` is the window center, the default accepted target band is the central 50% of the window, and the initial `T_search` is 5% of the window width (never below one energy-bin width). After `stall_attempts_per_spin*N` proposals without a closer energy, `T_search` is multiplied by `temperature_multiplier` up to `max_temperature_fraction` of the window width. Another stall at the maximum temperature reproducibly randomizes the spins and restarts at the initial temperature. The defaults are `1000`, `2`, and `0.5`, respectively. Configure these values in `[initialization]` or with the corresponding `--initialization-*` CLI options. Initialization proposals and restarts do not update WL counters; only the selected starting bin receives the initial `H` and `log_g` update.
- Initial iterations use `ln(f)=1` and `ln(f) <- ln(f)/2`. With `inverse_time=true`, a walker advances after every bin in its cumulative discovered energy support has been visited at least once during the current stage; the configured flatness threshold is not used. A traditional `inverse_time=false` run retains the histogram-flatness and minimum-visit criteria. Walkers advance independently without waiting for the other walkers in the window.
- Set `[wl] return_mode = true` or pass `--return-mode true` to restore every walker to a common reference configuration after each `ln(f) <- ln(f)/2` step. Without adaptive warm starts the reference is the all-`+1` state. When a warm-start bank is available, the lowest-energy configuration in that bank is used for every window, so the lower edge window returns directly to the best available ground-state candidate. Higher windows run the same adaptive target-Metropolis initialization from that reference until they re-enter their own energy range. This search does not increment production attempted/accepted counters or visit intermediate DOS cells; its final state seeds the new histogram once. The continuous `1/t` stage does not perform further returns because it no longer halves `ln(f)`.
- Set `[wl] nalivaiko_mod = true` to reset a separate flatness/coverage active-bin mask whenever `ln(f)` is halved. The cumulative active mask used for final DOS validity and the `1/t` active-bin count is preserved. The default is `false`, which retains cumulative discovered-bin checks.
- By default, each walker changes independently to `1/t` when `ln(f) <= 1/t`, with `t=attempted_flips/active_bins`; this internal counter is deliberately not converted to MCS. Set `[wl] inverse_time = false` or pass `--inverse-time false` to keep the traditional Wang–Landau schedule `ln(f) <- ln(f)/2` until `final_factor`. The active-bin count is cached and updated only on first visits and checkpoint restore, so a `1/t` flip remains `O(1)` apart from the coupling update.
- After sampling, window DOS estimates are aligned at the first bin in the intersection of all walker active masks and averaged in log space. Bins outside that intersection are written as `valid=0` with `log_g=nan`; adjacent windows must share at least one valid overlap bin.
- `force_accept_after_mcs=MCS` (or `--force-accept-after-mcs MCS`) forces the next otherwise rejected proposal that remains inside the walker's energy window after that many MCS without an accepted flip. The default `0` disables it. This is a non-standard escape heuristic that can bias the DOS; its use and actual forced-acceptance count are saved in metadata.
- Set `max_mcs=0` (or `--max-mcs 0`) to disable the work limit and run until every walker reaches `final_factor`. Pilot mode still requires a finite positive limit.
- Adjacent windows overlap by 75% by default. DOS fragments are joined where their local linear estimates of `d log(g)/dE` agree best.
- A run that reaches `--max-mcs` before `--final-factor` is marked `converged=false`; it is never silently presented as converged.

CUDA/HIP, Ewald summation, and continuous-kernel DOS are not implemented.
