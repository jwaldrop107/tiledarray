cmake_minimum_required(VERSION 3.12)

project(python-tiledarray)

if (TARGET tiledarray)
  get_target_property(tiledarray_target_type tiledarray TYPE)
  if (NOT ${tiledarray_target_type} STREQUAL SHARED_LIBRARY)
    message(FATAL_ERROR "tiledarray target must be shared library, found '${tiledarray_target_type}' instead")
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall)

find_package(Eigen3 3.3 REQUIRED)

# find_package(MPI REQUIRED)
# include_directories(${MPI_INCLUDE_PATH})

find_package(Boost REQUIRED) # spirit absent from cmake
# include_directories(${BOOST_INCLUDE_DIRS})

#find_package (Python COMPONENTS Interpreter Development)
#include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
#python_add_library(python-tiledarray MODULE src/tiledarray.cc WITH_SOABI)

add_subdirectory(pybind11)
pybind11_add_module(python-tiledarray MODULE src/tiledarray.cc)

# mpi c libs need come from tiledarray
target_link_libraries(python-tiledarray PRIVATE tiledarray)

set_target_properties(
  python-tiledarray
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME tiledarray
  )

if (Eigen3::Eigen)
  target_link_libraries(python-tiledarray INTERFACE Eigen3::Eigen)
else()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(python-tiledarray INTERFACE Boost::boost)

enable_testing()

# add the executable
add_test(
  NAME test_tiledarray
  COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/test_tiledarray.py
  )
