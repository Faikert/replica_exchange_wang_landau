cmake_minimum_required(VERSION 3.20)
project(dipolar_wang_landau VERSION 0.1.0 LANGUAGES CXX)

option(WL_ENABLE_OPENMP "Enable OpenMP shared-memory parallelism" ON)
option(WL_ENABLE_MPI "Enable MPI replica-exchange support" ON)
option(WL_BUILD_TESTS "Build unit and integration tests" ON)
option(WL_BUILD_BENCHMARKS "Build the flip-rate benchmark" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(wl_physics src/physics.cpp)
target_include_directories(wl_physics PUBLIC include)

add_library(wl_algorithm src/wang_landau.cpp src/analysis.cpp)
target_include_directories(wl_algorithm PUBLIC include)
target_link_libraries(wl_algorithm PUBLIC wl_physics)

add_library(wl_io src/io.cpp)
target_include_directories(wl_io PUBLIC include)
target_link_libraries(wl_io PUBLIC wl_algorithm)

add_library(wl_parallel src/parallel.cpp)
target_include_directories(wl_parallel PUBLIC include)
target_link_libraries(wl_parallel PUBLIC wl_io)

if(WL_ENABLE_OPENMP)
  find_package(OpenMP QUIET)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(wl_physics PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(wl_parallel PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(wl_parallel PUBLIC WL_HAS_OPENMP=1)
  else()
    message(WARNING "OpenMP requested but not found; building without it")
  endif()
endif()

if(WL_ENABLE_MPI)
  find_package(MPI QUIET COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    target_link_libraries(wl_parallel PUBLIC MPI::MPI_CXX)
    target_compile_definitions(wl_parallel PUBLIC WL_HAS_MPI=1)
  else()
    message(WARNING "MPI requested but not found; building serial/OpenMP executables")
  endif()
endif()

foreach(app wl_run wl_exact wl_analyze)
  add_executable(${app} apps/${app}.cpp)
  target_link_libraries(${app} PRIVATE wl_parallel)
endforeach()

if(WL_BUILD_BENCHMARKS)
  add_executable(wl_bench apps/wl_bench.cpp)
  target_link_libraries(wl_bench PRIVATE wl_parallel)
endif()

if(WL_BUILD_TESTS)
  enable_testing()
  add_executable(wl_tests tests/test_main.cpp)
  target_link_libraries(wl_tests PRIVATE wl_parallel)
  add_test(NAME wl_tests COMMAND wl_tests)
  if(MPI_CXX_FOUND)
    add_test(NAME wl_mpi_smoke
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:wl_run> --smoke-test)
    add_test(NAME wl_mpi_initialization_failure
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
              $<TARGET_FILE:wl_run>
              --nx 1 --ny 1 --nz 1 --periodic false
              --emin -1 --emax 3 --bin-width 0.5
              --windows 2 --overlap 0.5 --walkers 1
              --initialization-max-attempts 1
              --initialization-target-fraction 0.5
              --max-mcs 1 --seed 7 --output wl_mpi_initialization_failure)
    set_tests_properties(wl_mpi_initialization_failure PROPERTIES
      WILL_FAIL TRUE TIMEOUT 15)
  endif()
endif()

install(TARGETS wl_run wl_exact wl_analyze
        RUNTIME DESTINATION bin)
