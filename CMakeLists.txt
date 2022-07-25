#
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#  - Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  - Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  - Neither the name(s) of the copyright holder(s) nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

cmake_minimum_required(VERSION 3.17)

option(USE_GPU "Use GPU" OFF)

if (USE_GPU)
    project(torque LANGUAGES CXX CUDA)
else ()
    project(torque LANGUAGES CXX)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

IF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    FIND_PACKAGE(Git)
    IF (GIT_FOUND)
        EXECUTE_PROCESS(
                COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE "TORQUE_BUILD_VERSION"
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        MESSAGE(STATUS "Git version: ${TORQUE_BUILD_VERSION}")
    ELSE (GIT_FOUND)
        SET(TORQUE_BUILD_VERSION 0)
    ENDIF (GIT_FOUND)
ENDIF (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/src/version.h)

include_directories(${CMAKE_CURRENT_BINARY_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(torque SHARED

        src/tensor/dense.cpp
        src/tensor/block_sparse.cpp
        src/tensor/sparse.cpp

        src/util/space.cpp

        )


enable_testing()

################################
#  Compile external libraries  #
################################

#####Armadillo######

find_package(
        Armadillo
        8.500.0
        REQUIRED
)

if (${ARMADILLO_FOUND})
    message("Found Armadillo Library,")
    message("     ARMADILLO_VERSION: " ${ARMADILLO_VERSION_STRING})
    message("     ARMADILLO_LIBRARIES   : " ${ARMADILLO_LIBRARIES})
    message("     ARMADILLO_INCLUDE_DIRS: " ${ARMADILLO_INCLUDE_DIRS})

    include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})
    target_link_libraries(torque ${ARMADILLO_LIBRARIES})
endif ()

#######Catch2#######

add_library(catch2 SHARED external/catch2/catch2_defined.cpp)
include_directories("external/catch2")
target_include_directories(catch2 PUBLIC external/catch2/)

########fmt#########
add_subdirectory(external/fmt)
include_directories("external/fmt/include")
target_link_libraries(torque fmt::fmt-header-only)
add_definitions(-DHAVE_FMT)

set_target_properties(torque
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


#########openacc#########
######OpenMP######

FIND_PACKAGE(OpenMP)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

# ##########################################
# cuTENSOR dependencies
# ##########################################

option(WITH_CUTENSOR "Use cuTensor" OFF)

if (USE_GPU)
    find_package(CUDA)

    include_directories("${CUDA_INCLUDE_DIRS}")

    find_package(CUDAToolkit REQUIRED)

    if (WITH_CUTENSOR)

        if (NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
            message("cuTensor not found. A homebrew implementation will be used.")
            set(WITH_CUTENSOR OFF)
        else ()
            if (DEFINED ENV{CUTENSOR_ROOT})
                set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
            endif ()
            message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
            if (NOT EXISTS ${CUTENSOR_ROOT})
                message("Cannot find CUTENSOR_ROOT")
            else ()
                add_definitions(-DUSE_CUTENSOR)

                if(LOW_MEMORY)
                    add_definitions(-DLOW_MEMORY)
                endif()

                if (WITH_CUTENSOR)
                    if (NOT TARGET cutensor)
                        add_library(cutensor SHARED IMPORTED)
                        if (WIN32)
                            set(CUTENSOR_LIBRARY_NAME "cutensor.dll")
                            set(CUTENSOR_LIBRARY_DEF "cutensor.lib")
                        else ()
                            set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
                        endif ()
                        set(LIB_DIR "/lib/11")

                        set_target_properties(cutensor PROPERTIES
                                IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
                                IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
                                INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
                    endif ()

                    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS} ${CUTENSOR_ROOT}/include)

                    message("-- cuTENSOR linked.")
                endif ()
            endif ()
        endif ()

    endif ()

    add_library(gpu_torque SHARED

            src/gpu/sparse.cu

            )


    target_link_libraries(gpu_torque ${ARMADILLO_LIBRARIES})

    if (WITH_CUTENSOR)
        target_link_libraries(gpu_torque cutensor CUDA::cublas CUDA::cudart)
    else ()
        target_link_libraries(gpu_torque CUDA::cublas CUDA::cudart)
    endif ()

    if (CUTT_ROOT)

        ## cutt library ##
        set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

        find_package(
                CUTT
                REQUIRED
        )

        if (${CUTT_FOUND})
            message("Found CUTT Library,")
            message("     CUTT_LIBRARIES   : " ${CUTT_LIBRARIES})
            message("     CUTT_INCLUDE_DIRS: " ${CUTT_INCLUDE_DIRS})

            target_sources(gpu_torque PUBLIC
                    src/gpu/block_sparse.cu
                    src/gpu/dense.cu)

            include_directories(SYSTEM ${CUTT_INCLUDE_DIRS})
            target_link_libraries(gpu_torque ${CUTT_LIBRARIES})
        endif ()

    endif ()


endif ()


####################z
#       Test       #
####################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

add_executable(test_tensor
        test/dense_tensor_test.cpp
        test/sparse_tensor_test.cpp
        test/block_sparse_tensor_test.cpp
        )

target_link_libraries(test_tensor catch2 torque)


if (USE_GPU)
    add_executable(test_gpu_tensor
            test/gpu/sparse_test.cu
            )

    if (CUTT_ROOT)
        target_sources(test_gpu_tensor PUBLIC
                test/gpu/dense_test.cu
                test/gpu/block_sparse_test.cu
                )
    endif ()
    target_link_libraries(test_gpu_tensor catch2 torque gpu_torque)

    add_executable(cpu_gpu_benchmark
            test/gpu/cpu_gpu_comparison.cu
            )

    target_link_libraries(cpu_gpu_benchmark catch2 torque gpu_torque)

    add_executable(peps_benchmark
            test/gpu/peps_test.cu
            )

    target_link_libraries(peps_benchmark catch2 torque gpu_torque)

    add_executable(dense_peps_benchmark
            test/gpu/dense_peps_test.cu
            )

    target_link_libraries(dense_peps_benchmark catch2 torque gpu_torque)

    add_executable(large_tensor_benchmark
            test/gpu/large_tensor_benchmark.cu
            )

    target_link_libraries(large_tensor_benchmark catch2 torque gpu_torque)

    add_executable(block_sparse_benchmark
            test/gpu/block_sparse_benchmark.cu
            )

    target_link_libraries(block_sparse_benchmark catch2 torque gpu_torque)

    add_executable(random_generated_block_sparse_benchmark
            test/gpu/random_generated_block_sparse_benchmark.cu
            )

    target_link_libraries(random_generated_block_sparse_benchmark catch2 torque gpu_torque)
endif ()



