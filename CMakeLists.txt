cmake_minimum_required(VERSION 3.25)

set(CMAKE_BUILD_TYPE Debug)

# Enforce C++ compiler for CUDA
# Check if /usr/bin/g++-12 exists
# execute_process(
#     COMMAND /usr/bin/g++-12 --version
#     RESULT_VARIABLE GPP_VERSION_CHECK_RESULT
#     OUTPUT_VARIABLE GPP_VERSION_OUTPUT
#     ERROR_QUIET
# )

# # If g++-12 exists, set CMAKE_CUDA_HOST_COMPILER
# if(GPP_VERSION_CHECK_RESULT EQUAL 0)
#     set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/g++-12")
# else()
#     message(STATUS "g++-12 not found, not setting CMAKE_CUDA_HOST_COMPILER")
# endif()

# if(GPP_VERSION_CHECK_RESULT EQUAL 0)
#     set(CMAKE_CXX_COMPILER "/usr/bin/g++-12")
# else()
#     message(STATUS "g++-12 not found, not setting CMAKE_CXX_COMPILER")
# endif()

# Specify the maximum number of concurrent processes for building
set(CMAKE_BUILD_PARALLEL_LEVEL 8)

# specify architecture for rtx 4060
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "86;89")
endif()

# get current folder name
string(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})

project("coherent_dedispersion")
enable_language(CUDA)

set(MY_INCLUDE_DIRS
  ${PROJECT_SOURCE_DIR}/utils/include
  ${PROJECT_SOURCE_DIR}/psrdada/include
  ${PROJECT_SOURCE_DIR}/folding/include
  ${PROJECT_SOURCE_DIR}/gpu/include
  ${PROJECT_SOURCE_DIR}/dedispersion/include
  ${PROJECT_SOURCE_DIR}/simulated_pulsar/include
  ${PROJECT_SOURCE_DIR}/data_save/include
)

set(MY_SOURCE_CPP_FILES
  ${PROJECT_SOURCE_DIR}/utils/*.cpp
  ${PROJECT_SOURCE_DIR}/psrdada/*.cpp
  ${PROJECT_SOURCE_DIR}/folding/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/msosm/*.cpp
  ${PROJECT_SOURCE_DIR}/dedispersion/overlap_save/*.cpp
  ${PROJECT_SOURCE_DIR}/simulated_pulsar/*.cpp
  ${PROJECT_SOURCE_DIR}/data_save/*.cpp
)

set(MY_SOURCE_CU_FILES
  ${PROJECT_SOURCE_DIR}/gpu/*.cu
)

include_directories(${MY_INCLUDE_DIRS})

# file
file(GLOB CPP_FILES ${MY_SOURCE_CPP_FILES})
file(GLOB CU_FILES ${MY_SOURCE_CU_FILES})

# List of executables
set(EXECTABLES
  msosm_psrdada
  gpu_info
  msosm_save_simulated
  # Add more executables here
)

# List of benchmark executables
set(BENCHMARK_EXECUTABLES
  test_msosm_speed
  check_msosm
  # Add more benchmark executables here
)

# cuFFT
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Development NumPy)

find_package(Python COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# liburing
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBURING REQUIRED liburing)

foreach(EXE ${EXECTABLES})

  # target
  add_executable(${EXE} ${EXE}.cpp ${CPP_FILES} ${CU_FILES})
  target_include_directories(${EXE} PRIVATE ${MY_INCLUDE_DIRS})

  # cuFFT
  target_link_libraries(${EXE} CUDA::cufft)

  target_link_libraries(${EXE} Python::Python Python::NumPy)

  # FFTW
  target_link_libraries(${EXE} fftw3f)

  # NVTX
  target_link_libraries(${EXE} CUDA::nvtx3)

  # liburing
  target_link_libraries(${EXE} ${LIBURING_LIBRARIES})
  target_include_directories(${EXE} PRIVATE ${LIBURING_INCLUDE_DIRS})

endforeach()

foreach(BENCH_EXE ${BENCHMARK_EXECUTABLES})

  # target
  add_executable(${BENCH_EXE} ${PROJECT_SOURCE_DIR}/benchmark/${BENCH_EXE}.cpp ${CPP_FILES} ${CU_FILES})
  target_include_directories(${BENCH_EXE} PRIVATE ${MY_INCLUDE_DIRS})

  # cuFFT
  target_link_libraries(${BENCH_EXE} CUDA::cufft)

  target_link_libraries(${BENCH_EXE} Python::Python Python::NumPy)

  # FFTW
  target_link_libraries(${BENCH_EXE} fftw3f)

  # NVTX
  target_link_libraries(${BENCH_EXE} CUDA::nvtx3)

  # liburing
  target_link_libraries(${BENCH_EXE} ${LIBURING_LIBRARIES})
  target_include_directories(${BENCH_EXE} PRIVATE ${LIBURING_INCLUDE_DIRS})

endforeach()