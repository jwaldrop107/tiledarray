#
#  This file is a part of TiledArray.
#  Copyright (C) 2013  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  Justus Calvin
#  Department of Chemistry, Virginia Tech
#
#  CMakeLists.txt
#  Jul 19, 2013
#

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_config_h.in
  ${PROJECT_BINARY_DIR}/tests/unit_test_config.h
)

# Create the ta_test executable
set(executable ta_test)
# N.B.: The order of files here represents the order in which the tests are run.
# N.B. 2: if you want to trim this down you may need to resolve linker errors due to missing fixture deps manually
set(ta_test_src_files  ta_test.cpp
    range.cpp
    btas.cpp
    meta.cpp
    type_traits.cpp
    utility.cpp
    permutation.cpp
    symm_permutation_group.cpp
    symm_irrep.cpp
    symm_representation.cpp
    block_range.cpp
    perm_index.cpp
    transform_iterator.cpp
    bitset.cpp
    math_outer.cpp
    math_partial_reduce.cpp
    math_transpose.cpp
    math_blas.cpp
    tensor.cpp
    tensor_of_tensor.cpp
    tensor_tensor_view.cpp
    tensor_shift_wrapper.cpp
    tiled_range1.cpp
    tiled_range.cpp
    blocked_pmap.cpp
    hash_pmap.cpp
    cyclic_pmap.cpp
    replicated_pmap.cpp
    dense_shape.cpp
    sparse_shape.cpp
    distributed_storage.cpp
    tensor_impl.cpp
    array_impl.cpp
    variable_list.cpp
    dist_array.cpp
    conversions.cpp
    eigen.cpp
    dist_op_dist_cache.cpp
    dist_op_group.cpp
    dist_op_communicator.cpp
    tile_op_noop.cpp
    tile_op_scal.cpp
    dist_eval_array_eval.cpp
    dist_eval_unary_eval.cpp
    tile_op_add.cpp
    tile_op_scal_add.cpp
    tile_op_subt.cpp
    tile_op_scal_subt.cpp
    dist_eval_binary_eval.cpp
    tile_op_mult.cpp
    tile_op_scal_mult.cpp
    tile_op_contract_reduce.cpp
    reduce_task.cpp
    proc_grid.cpp
    dist_eval_contraction_eval.cpp
    expressions.cpp
    expressions_sparse.cpp
    expressions_complex.cpp
    expressions_btas.cpp
    expressions_mixed.cpp
    foreach.cpp
    solvers.cpp
    initializer_list.cpp
    scalapack.cpp
    diagonal_array.cpp
)

if(CUDA_FOUND)
    list(APPEND ta_test_src_files cutt.cpp expressions_cuda_um.cpp tensor_um.cpp)
endif()

# if tiledarray library was compiled without exceptions, use TA header-only (see below)
if (NOT TA_DEFAULT_ERROR EQUAL 1 AND NOT CUDA_FOUND)
  add_ta_executable(${executable} "${ta_test_src_files}" "MADworld;${TILEDARRAY_PRIVATE_LINK_LIBRARIES}")
  target_compile_definitions(${executable}-lib PRIVATE TILEDARRAY_HEADER_ONLY=1)
  if (LAPACK_INCLUDE_DIRS)
    target_include_directories(${executable}-lib PRIVATE ${LAPACK_INCLUDE_DIRS})
  endif(LAPACK_INCLUDE_DIRS)
  if (LAPACK_COMPILE_OPTIONS)
    target_compile_options(${executable}-lib PRIVATE ${LAPACK_COMPILE_OPTIONS})
  endif(LAPACK_COMPILE_OPTIONS)
  if (LAPACK_COMPILE_DEFINITIONS)
    target_compile_definitions(${executable}-lib PRIVATE ${LAPACK_COMPILE_DEFINITIONS})
  endif(LAPACK_COMPILE_DEFINITIONS)
else()
  add_ta_executable(${executable} "${ta_test_src_files}" "tiledarray")
endif()
# Add include directories and compiler flags for ta_test
target_include_directories(${executable}-lib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/tests
    ${Boost_INCLUDE_DIRS}
    )
# unit tests
# 1. should disable error messages
# 2. must emit exceptions in asserts since some tests check for exceptions
target_compile_definitions(${executable}-lib PRIVATE TILEDARRAY_NO_USER_ERROR_MESSAGES=1
    TA_EXCEPTION_ERROR=1 MADNESS_DISPLAY_EXCEPTION_BREAK_MESSAGE=0)

# Add targets
add_test(build_${executable} "${CMAKE_COMMAND}" --build ${PROJECT_BINARY_DIR} --target ${executable})

# Add a test(s)
if(ENABLE_MPI)
  foreach(p RANGE 1 2)
    add_test(NAME ${executable}-np-${p}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${p} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:${executable}> --log_level=test_suite --show-progress ${MPIEXEC_POSTFLAGS})
    set_tests_properties(${executable}-np-${p} PROPERTIES DEPENDS build_${executable})
  endforeach(p)
else()
  add_test(NAME ${executable}
      COMMAND ${executable})
  set_tests_properties(${executable} PROPERTIES DEPENDS build_${executable})
endif()
