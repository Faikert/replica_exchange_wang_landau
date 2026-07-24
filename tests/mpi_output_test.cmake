if(NOT DEFINED MPIEXEC OR NOT DEFINED NUMPROC_FLAG OR NOT DEFINED WL_RUN OR
   NOT DEFINED GEOMETRY OR NOT DEFINED PREFIX)
  message(FATAL_ERROR "MPI output test is missing a required argument")
endif()

set(expected_files
  "${PREFIX}_metadata.json"
  "${PREFIX}_workers_stat.csv"
  "${PREFIX}_window_0_dos2d.csv")
file(REMOVE ${expected_files})

execute_process(
  COMMAND "${MPIEXEC}" "${NUMPROC_FLAG}" 2 "${WL_RUN}"
          --smoke-test
          --geometry "${GEOMETRY}"
          --periodic false
          --order-parameter weighted_sum
          --q-bin-width 1
          --support-stability-checks 1
          --output "${PREFIX}"
  RESULT_VARIABLE run_result
  OUTPUT_VARIABLE run_stdout
  ERROR_VARIABLE run_stderr
  TIMEOUT 30)

if(NOT run_result EQUAL 0)
  message(FATAL_ERROR
    "MPI joint smoke failed with ${run_result}\nstdout:\n${run_stdout}\nstderr:\n${run_stderr}")
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
