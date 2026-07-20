# Project instructions

We are building a high-performance C++20 implementation of the Wang–Landau algorithm for an Ising-like dipolar spin system.

Primary goals:
    
* Correct physical implementation of dipole–dipole Ising interactions.
* High-performance Monte Carlo updates.
* Parallel Wang–Landau sampling.
* Reproducible scientific results.
* Clear tests, benchmarks, and documentation.

Language and tooling:

* Use modern C++20.
* Use CMake.
* Use OpenMP for shared-memory parallelism.
* Use MPI for multi-process/multi-walker Wang–Landau runs.
* Keep CUDA/HIP as an optional later backend, not required for the first working version.
* Avoid unnecessary dependencies.
* Prefer standard library, CMake, OpenMP, MPI, and small header-only dependencies only if clearly justified.

Code style:

* Use clear, modular C++.
* Separate physics, algorithm, I/O, tests, and CLI.
* Avoid global mutable state.
* Make random number generation reproducible.
* Use RAII, const-correctness, and explicit ownership.
* Prefer structure-of-arrays or cache-friendly layouts for performance-critical data.
* Document all nontrivial physical formulas.

Performance requirements:

* Do not recompute full O(N²) energy after every spin flip.
* Precompute pair couplings `J_ij` into a neighbor-list or dense/cutoff backend.
* Maintain local fields `h_i = sum_j J_ij sigma_j`.
* Compute a spin-flip energy difference as `dE = -2 * sigma_i * h_i`.
* After an accepted flip, update affected local fields incrementally.
* Provide benchmarks in attempted flips per second.

Testing requirements:

* Unit-test pair interaction formulas.
* Unit-test total energy against direct summation.
* Unit-test `dE` against full recomputation after random flips.
* Unit-test periodic boundary minimum-image convention.
* Unit-test Wang–Landau histogram/bin updates.
* For very small systems, compare against exact enumeration of the density of states.

Scientific output:

* Save `log_g(E)`, histogram `H(E)`, run metadata, random seed, parameters, and thermodynamic observables.
* Provide examples in `examples/`.
* Provide a `README.md` with build and run instructions.

Important:

* Do not silently change the physics model.
* If something is ambiguous, implement a reasonable default and document it.
* Prefer a correct CPU implementation first, then optimize.
