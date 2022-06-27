include(FindPackageHandleStandardArgs)

if(EXISTS "$ENV{CUTT_ROOT_DIR}")
  file( TO_CMAKE_PATH "$ENV{CUTT_ROOT_DIR}" CUTT_ROOT_DIR)
  set(CUTT_ROOT_DIR "${CUTT_ROOT_DIR}" CACHE PATH "Prefix for CUTT installation.")
endif()

find_path(CUTT_INCLUDE_DIR 
    NAMES cutt.h
    HINTS ${CUTT_ROOT_DIR}/include)
find_library(CUTT_LIBRARY 
    NAMES cutt
    HINTS ${CUTT_ROOT_DIR}/lib)


find_package_handle_standard_args(CUTT
    FOUND_VAR
        CUTT_FOUND
    REQUIRED_VARS
        CUTT_INCLUDE_DIR
        CUTT_LIBRARY
)

mark_as_advanced(CUTT_LIBRARY CUTT_INCLUDE_DIR)

if(CUTT_FOUND)
    set(CUTT_LIBRARIES    ${CUTT_LIBRARY})
    set(CUTT_INCLUDE_DIRS ${CUTT_INCLUDE_DIR})
endif()
