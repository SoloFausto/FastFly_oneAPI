cmake_minimum_required(VERSION 3.20)
project(FastFly LANGUAGES CXX)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsycl")
check_cxx_source_compiles("#include <sycl/sycl.hpp>\nint main() { sycl::queue q; q.wait_and_throw(); }" FASTFLY_HAS_SYCL)
unset(CMAKE_REQUIRED_FLAGS)
if(NOT FASTFLY_HAS_SYCL)
    message(FATAL_ERROR "A SYCL compiler is required. Initialize Intel oneAPI, then configure with -DCMAKE_CXX_COMPILER=icpx (Linux) or icx (Windows).")
endif()

add_library(fastfly_sycl SHARED fastfly_sycl.cpp)
target_compile_features(fastfly_sycl PUBLIC cxx_std_17)
target_compile_options(fastfly_sycl PRIVATE -fsycl)
target_link_options(fastfly_sycl PRIVATE -fsycl)
target_include_directories(fastfly_sycl PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(flywire_sim flywire_sim.cpp)
target_compile_features(flywire_sim PRIVATE cxx_std_17)
target_link_libraries(flywire_sim PRIVATE fastfly_sycl)
# Keep the executable and Python-loadable library together for every generator.
set_target_properties(fastfly_sycl flywire_sim PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
