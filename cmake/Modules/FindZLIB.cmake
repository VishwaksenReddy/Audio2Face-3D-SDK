# FindZLIB.cmake
# Find the ZLIB compression library
#
# This module creates the following imported target:
#  ZLIB::ZLIB - The ZLIB library target
#
# Usage:
#  find_package(ZLIB REQUIRED)
#  target_link_libraries(your_target PRIVATE ZLIB::ZLIB)
#
# You can specify the ZLIB installation directory by setting:
#  ZLIB_ROOT         - Environment variable or CMake variable
#
# If not specified, the module will search in standard locations and
# the project's typical dependency location.

# Allow overriding the search path via environment variable or CMake variable
set(_zlib_search_paths)

# Priority 1: CMake variable ZLIB_ROOT
if(ZLIB_ROOT)
    list(APPEND _zlib_search_paths ${ZLIB_ROOT})
endif()

# Priority 2: Environment variable ZLIB_ROOT
if(DEFINED ENV{ZLIB_ROOT})
    list(APPEND _zlib_search_paths $ENV{ZLIB_ROOT})
endif()

# Priority 3: Standard system locations
list(APPEND _zlib_search_paths
    /usr/local
    /usr
    /opt/local
    /opt
)

# For multi-config generators, try to find both debug and release versions
set(_found_debug FALSE)
set(_found_release FALSE)

# Find debug version
find_path(ZLIB_INCLUDE_DIR_DEBUG
    NAMES zlib.h
    PATHS ${_zlib_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/zlib/debug
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    DOC "ZLIB debug include directory"
)

find_library(ZLIB_LIBRARY_DEBUG
    NAMES z zlib zlibd
    PATHS ${_zlib_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/zlib/debug
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    DOC "ZLIB debug library"
)

if(ZLIB_INCLUDE_DIR_DEBUG AND ZLIB_LIBRARY_DEBUG)
    set(_found_debug TRUE)
endif()

# Find release version
find_path(ZLIB_INCLUDE_DIR_RELEASE
    NAMES zlib.h
    PATHS ${_zlib_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/zlib/release
    PATH_SUFFIXES include
    NO_DEFAULT_PATH
    DOC "ZLIB release include directory"
)

find_library(ZLIB_LIBRARY_RELEASE
    NAMES z zlib
    PATHS ${_zlib_search_paths} ${PROJECT_SOURCE_DIR}/_deps/target-deps/zlib/release
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
    DOC "ZLIB release library"
)

if(ZLIB_INCLUDE_DIR_RELEASE AND ZLIB_LIBRARY_RELEASE)
    set(_found_release TRUE)
endif()

# Fallback to old behavior if multi-config versions not found
if(NOT _found_debug AND NOT _found_release)
    find_path(ZLIB_INCLUDE_DIR
        NAMES zlib.h
        PATHS ${_zlib_search_paths}
        PATH_SUFFIXES include
        DOC "ZLIB include directory"
    )

    find_library(ZLIB_LIBRARY
        NAMES z zlib
        PATHS ${_zlib_search_paths}
        PATH_SUFFIXES lib lib64
        DOC "ZLIB library"
    )
endif()

# Determine what we found for the standard args check
if(_found_debug OR _found_release)
    set(ZLIB_FOUND TRUE)
    if(_found_debug)
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIR_DEBUG})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARY_DEBUG})
    else()
        set(ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIR_RELEASE})
        set(ZLIB_LIBRARY ${ZLIB_LIBRARY_RELEASE})
    endif()
endif()

# Determine the version from zlib.h if we found it
if(EXISTS "${ZLIB_INCLUDE_DIR}/zlib.h")
    file(STRINGS "${ZLIB_INCLUDE_DIR}/zlib.h" ZLIB_H REGEX "^#define ZLIB_VERSION \"[^\"]*\"$")
    string(REGEX REPLACE "^.*ZLIB_VERSION \"([0-9]+).*$" "\\1" ZLIB_VERSION_MAJOR "${ZLIB_H}")
    string(REGEX REPLACE "^.*ZLIB_VERSION \"[0-9]+\\.([0-9]+).*$" "\\1" ZLIB_VERSION_MINOR  "${ZLIB_H}")
    string(REGEX REPLACE "^.*ZLIB_VERSION \"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ZLIB_VERSION_PATCH "${ZLIB_H}")
    set(ZLIB_VERSION_STRING "${ZLIB_VERSION_MAJOR}.${ZLIB_VERSION_MINOR}.${ZLIB_VERSION_PATCH}")
        
    # Compatibility
    set(ZLIB_MAJOR_VERSION "${ZLIB_VERSION_MAJOR}")
    set(ZLIB_MINOR_VERSION "${ZLIB_VERSION_MINOR}")
    set(ZLIB_PATCH_VERSION "${ZLIB_VERSION_PATCH}")
endif()

# Handle standard arguments (with version check)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZLIB
    FOUND_VAR ZLIB_FOUND
    REQUIRED_VARS ZLIB_INCLUDE_DIR ZLIB_LIBRARY
    VERSION_VAR ZLIB_VERSION_STRING
)

# Create imported target for modern CMake
if(ZLIB_FOUND)
    if(NOT TARGET ZLIB::ZLIB)
        add_library(ZLIB::ZLIB UNKNOWN IMPORTED)
        
        # Use multi-config approach if we found both debug and release
        if(_found_debug AND _found_release)
            # Use generator expressions to select the right version based on config
            set_target_properties(ZLIB::ZLIB PROPERTIES
                IMPORTED_LOCATION_DEBUG "${ZLIB_LIBRARY_DEBUG}"
                IMPORTED_LOCATION_RELEASE "${ZLIB_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_RELWITHDEBINFO "${ZLIB_LIBRARY_RELEASE}"
                IMPORTED_LOCATION_MINSIZEREL "${ZLIB_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "$<IF:$<CONFIG:Debug>,${ZLIB_INCLUDE_DIR_DEBUG},${ZLIB_INCLUDE_DIR_RELEASE}>"
            )
        elseif(_found_debug)
            # Only debug version found
            set_target_properties(ZLIB::ZLIB PROPERTIES
                IMPORTED_LOCATION "${ZLIB_LIBRARY_DEBUG}"
                INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR_DEBUG}"
            )
        elseif(_found_release)
            # Only release version found
            set_target_properties(ZLIB::ZLIB PROPERTIES
                IMPORTED_LOCATION "${ZLIB_LIBRARY_RELEASE}"
                INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR_RELEASE}"
            )
        else()
            # Fallback to old behavior
            set_target_properties(ZLIB::ZLIB PROPERTIES
                IMPORTED_LOCATION "${ZLIB_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIR}"
            )
        endif()
    endif()
endif()

# Clean up temporary variables
unset(_zlib_search_paths)
unset(_found_debug)
unset(_found_release)
unset(ZLIB_INCLUDE_DIR)
unset(ZLIB_LIBRARY)
unset(ZLIB_INCLUDE_DIR_DEBUG)
unset(ZLIB_LIBRARY_DEBUG)
unset(ZLIB_INCLUDE_DIR_RELEASE)
unset(ZLIB_LIBRARY_RELEASE)
