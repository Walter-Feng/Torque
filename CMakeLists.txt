cmake_minimum_required(VERSION 3.17)
project(torque LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

IF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)
    FIND_PACKAGE(Git)
    IF(GIT_FOUND)
        EXECUTE_PROCESS(
                COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                OUTPUT_VARIABLE "TORQUE_BUILD_VERSION"
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        MESSAGE( STATUS "Git version: ${TORQUE_BUILD_VERSION}" )
    ELSE(GIT_FOUND)
        SET(TORQUE_BUILD_VERSION 0)
    ENDIF(GIT_FOUND)
ENDIF(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git)

CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/src/version.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(torque STATIC
        src/tensor/tensor.cpp
        src/tensor/dense.cpp
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
include_directories("external/Catch2")
target_include_directories(catch2 PUBLIC external/Catch2/)

########fmt#########
add_subdirectory(external/fmt)
include_directories("external/fmt/include")
target_link_libraries(torque fmt::fmt-header-only)
add_definitions(-DHAVE_FMT)

set_target_properties(torque
        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

####################z
#       Test       #
####################

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test)

add_executable(test_tensor_dense
            test/dense_tensor_test.cpp
        )

target_link_libraries(test_tensor_dense catch2 torque)

add_test(NAME UNITTEST_core
        COMMAND bin/test_tensor_dense)