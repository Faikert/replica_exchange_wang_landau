if(NOT DEFINED MPIEXEC OR NOT DEFINED NUMPROC_FLAG OR NOT DEFINED WL_RUN OR
   NOT DEFINED GEOMETRY OR NOT DEFINED PREFIX OR NOT DEFINED RANKS OR
   NOT DEFINED WINDOWS)
  message(FATAL_ERROR "MPI output test is missing a required argument")
endif()

set(expected_files
  "${PREFIX}_metadata.json"
  "${PREFIX}_workers_stat.csv")
math(EXPR last_window "${WINDOWS}-1")
foreach(window RANGE 0 ${last_window})
  list(APPEND expected_files "${PREFIX}_window_${window}_dos2d.csv")
endforeach()
file(REMOVE ${expected_files})

execute_process(
  COMMAND "${MPIEXEC}" "${NUMPROC_FLAG}" "${RANKS}" "${WL_RUN}"
          --geometry "${GEOMETRY}"
          --periodic false
          --emin -8 --emax 8 --bin-width 0.25
          --order-parameter weighted_sum
          --q-bin-width 1
          --support-stability-checks 1
          --windows "${WINDOWS}"
          --walkers 1
          --initialization-target-fraction 1
          --max-mcs 1
          --check-interval-mcs 1
          --final-factor 1e-30
          --seed 7
          --progress 0
          --output "${PREFIX}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
  TIMEOUT 30)

if(NOT run_result EQUAL 2)
  message(FATAL_ERROR
    "MPI joint nonconverged run returned ${run_result}, expected 2\nstdout:\n${run_stdout}\nstderr:\n${run_stderr}")
endif()

foreach(path IN LISTS expected_files)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR
      "MPI run returned successfully but did not create ${path}\nstdout:\n${run_stdout}\nstderr:\n${run_stderr}")
  endif()
  file(SIZE "${path}" output_size)
  if(output_size EQUAL 0)
    message(FATAL_ERROR "MPI run created an empty output file: ${path}")
  endif()
endforeach()
