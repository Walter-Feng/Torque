cmake_minimum_required(VERSION 3.17)
project(torque LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

################################
#  Compile external libraries  #
################################

#####Armadillo######

find_package(
        Armadillo
        8.500.0
        REQUIRED
)

add_library(torque STATIC
        src/tensor/tensor.cpp
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
include_directories("external/Catch2")
target_include_directories(catch2 PUBLIC external/Catch2/)

########fmt#########
add_subdirectory(external/fmt)
include_directories("external/fmt/include")
target_link_libraries(torque fmt::fmt-header-only)
add_definitions(-DHAVE_FMT)

FIND_PACKAGE(OpenMP)

if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif ()

set_target_properties(torque
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

