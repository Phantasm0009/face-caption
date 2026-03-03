# SIMDe (SIMD Everywhere). Required by libobs when using OBS build.
# Search: OBS deps, then vcpkg/CMAKE_PREFIX_PATH, then system.
set(_simde_search_paths)
if(OBS_DEPS_DIR AND EXISTS "${OBS_DEPS_DIR}/include")
    list(APPEND _simde_search_paths "${OBS_DEPS_DIR}/include")
endif()
if(OBS_BUILD_DIR)
    if(EXISTS "${OBS_BUILD_DIR}/deps/include")
        list(APPEND _simde_search_paths "${OBS_BUILD_DIR}/deps/include")
    endif()
    get_filename_component(_obs_src "${OBS_BUILD_DIR}" DIRECTORY)
    if(EXISTS "${_obs_src}/.deps")
        file(GLOB _obs_deps "${_obs_src}/.deps/obs-deps-*-x64")
        foreach(_d ${_obs_deps})
            if(EXISTS "${_d}/include")
                list(APPEND _simde_search_paths "${_d}/include")
            endif()
        endforeach()
    endif()
endif()
# vcpkg: derive installed dir from toolchain path (e.g. C:/vcpkg/scripts/buildsystems/vcpkg.cmake -> C:/vcpkg/installed/x64-windows)
if(CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "[Vv]cpkg")
    get_filename_component(_vcpkg_scripts "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(_vcpkg_root "${_vcpkg_scripts}" DIRECTORY)
    get_filename_component(_vcpkg_root "${_vcpkg_root}" DIRECTORY)
    if(EXISTS "${_vcpkg_root}/installed/x64-windows/include")
        list(APPEND _simde_search_paths "${_vcpkg_root}/installed/x64-windows/include")
    endif()
endif()
foreach(_p ${CMAKE_PREFIX_PATH})
    if(EXISTS "${_p}/include")
        list(APPEND _simde_search_paths "${_p}/include")
    endif()
    if(EXISTS "${_p}")
        list(APPEND _simde_search_paths "${_p}")
    endif()
endforeach()
if(_simde_search_paths)
    find_path(SIMDe_INCLUDE_DIR simde/simde.h
        PATHS ${_simde_search_paths}
        NO_DEFAULT_PATH
    )
    # vcpkg's simde port copies the simde/ folder; try alternate header if main one not found
    if(NOT SIMDe_INCLUDE_DIR)
        find_path(SIMDe_INCLUDE_DIR simde/arm/neon.h
            PATHS ${_simde_search_paths}
            NO_DEFAULT_PATH
        )
    endif()
endif()
if(NOT SIMDe_INCLUDE_DIR)
    find_path(SIMDe_INCLUDE_DIR simde/simde.h
        PATHS /usr/include /usr/local/include
        PATH_SUFFIXES simde
    )
endif()
if(SIMDe_INCLUDE_DIR)
    set(SIMDe_FOUND TRUE)
    if(NOT TARGET SIMDe::SIMDe)
        add_library(SIMDe::SIMDe INTERFACE IMPORTED)
        set_target_properties(SIMDe::SIMDe PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SIMDe_INCLUDE_DIR}"
        )
    endif()
else()
    set(SIMDe_FOUND FALSE)
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SIMDe DEFAULT_MSG SIMDe_INCLUDE_DIR)
