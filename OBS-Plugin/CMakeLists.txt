cmake_minimum_required(VERSION 3.16)
project(obs-face-captions VERSION 1.0.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OBS_BUILD_DIR "" CACHE PATH "Path to obs-studio build directory")
set(OBS_DEPS_DIR "" CACHE PATH "Path to OBS deps (optional, for Windows)")

if(OBS_BUILD_DIR)
    set(_obs_prefix "${OBS_BUILD_DIR}")
    if(OBS_DEPS_DIR)
        list(APPEND _obs_prefix "${OBS_DEPS_DIR}")
    endif()
    if(EXISTS "${OBS_BUILD_DIR}/deps")
        list(APPEND _obs_prefix "${OBS_BUILD_DIR}/deps")
    endif()
    set(CMAKE_PREFIX_PATH "${_obs_prefix};${CMAKE_PREFIX_PATH}")
endif()
# Ensure vcpkg installed dir is on CMAKE_PREFIX_PATH so FindSIMDe (used by libobs) can find SIMDe
if(CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "[Vv]cpkg")
    get_filename_component(_vcpkg_scripts "${CMAKE_TOOLCHAIN_FILE}" DIRECTORY)
    get_filename_component(_vcpkg_root "${_vcpkg_scripts}" DIRECTORY)
    get_filename_component(_vcpkg_root "${_vcpkg_root}" DIRECTORY)
    if(EXISTS "${_vcpkg_root}/installed/x64-windows")
        list(PREPEND CMAKE_PREFIX_PATH "${_vcpkg_root}/installed/x64-windows")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(SIMDe QUIET)
find_package(libobs REQUIRED)
find_package(OpenCV REQUIRED)

option(WITH_VOSK "Build with Vosk speech-to-text support" ON)
if(WITH_VOSK)
    find_path(VOSK_INCLUDE_DIR vosk_api.h PATHS /usr/include /usr/local/include)
    find_library(VOSK_LIBRARY vosk PATHS /usr/lib /usr/local/lib)
    if(VOSK_INCLUDE_DIR AND VOSK_LIBRARY)
        set(HAVE_VOSK 1)
    elseif(WIN32)
        set(VOSK_ROOT "" CACHE PATH "Vosk C API root (include + lib)")
        if(VOSK_ROOT)
            if(EXISTS "${VOSK_ROOT}/include/vosk_api.h")
                set(VOSK_INCLUDE_DIR "${VOSK_ROOT}/include")
                set(VOSK_LIBRARY "${VOSK_ROOT}/lib/vosk.lib")
                set(VOSK_DLL_NAME "vosk.dll")
                set(HAVE_VOSK 1)
            elseif(EXISTS "${VOSK_ROOT}/vosk_api.h")
                set(VOSK_INCLUDE_DIR "${VOSK_ROOT}")
                if(EXISTS "${VOSK_ROOT}/libvosk.lib")
                    set(VOSK_LIBRARY "${VOSK_ROOT}/libvosk.lib")
                    set(VOSK_DLL_NAME "libvosk.dll")
                else()
                    set(VOSK_LIBRARY "${VOSK_ROOT}/lib/vosk.lib")
                    set(VOSK_DLL_NAME "vosk.dll")
                endif()
                set(HAVE_VOSK 1)
            endif()
        endif()
    endif()
endif()

set(PLUGIN_SOURCES
    src/plugin-main.cpp
    src/face-captions-filter.cpp
    src/face-detector.cpp
    src/caption-renderer.cpp
    src/kalman-tracker.cpp
    src/stt-vosk.cpp
)

if(HAVE_VOSK)
    add_compile_definitions(HAVE_VOSK=1)
endif()

add_library(obs-face-captions MODULE ${PLUGIN_SOURCES})

target_include_directories(obs-face-captions PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenCV_INCLUDE_DIRS}
)
if(OBS_BUILD_DIR)
    get_filename_component(OBS_SOURCE_DIR "${OBS_BUILD_DIR}" DIRECTORY)
    target_include_directories(obs-face-captions PRIVATE ${OBS_SOURCE_DIR}/libobs)
endif()
if(HAVE_VOSK)
    target_include_directories(obs-face-captions PRIVATE ${VOSK_INCLUDE_DIR})
endif()

target_link_libraries(obs-face-captions PRIVATE
    OBS::libobs
    ${OpenCV_LIBS}
)
if(HAVE_VOSK)
    target_link_libraries(obs-face-captions PRIVATE ${VOSK_LIBRARY})
endif()

if(WIN32)
    target_compile_definitions(obs-face-captions PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
endif()

if(WIN32)
    set(OBS_PLUGINS_DEST "bin/64bit/obs-plugins")
else()
    set(OBS_PLUGINS_DEST "lib/obs-plugins")
endif()

install(TARGETS obs-face-captions
    LIBRARY DESTINATION lib/obs-plugins
    RUNTIME DESTINATION ${OBS_PLUGINS_DEST}
    BUNDLE DESTINATION lib/obs-plugins
)

if(HAVE_VOSK AND WIN32 AND VOSK_ROOT)
    if(NOT DEFINED VOSK_DLL_NAME)
        set(VOSK_DLL_NAME "vosk.dll")
    endif()
    set(VOSK_DLL "${VOSK_ROOT}/lib/${VOSK_DLL_NAME}")
    if(NOT EXISTS "${VOSK_DLL}")
        set(VOSK_DLL "${VOSK_ROOT}/${VOSK_DLL_NAME}")
    endif()
    if(EXISTS "${VOSK_DLL}")
        add_custom_command(TARGET obs-face-captions POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${VOSK_DLL}" "$<TARGET_FILE_DIR:obs-face-captions>"
            COMMENT "Copying Vosk DLL to plugin output"
        )
    endif()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/data/haarcascade_frontalface_default.xml" AND OBS_BUILD_DIR)
    set(OBS_PLUGIN_DATA "${OBS_BUILD_DIR}/rundir/RelWithDebInfo/data/obs-plugins/obs-face-captions")
    add_custom_command(TARGET obs-face-captions POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${OBS_PLUGIN_DATA}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/data/haarcascade_frontalface_default.xml"
            "${OBS_PLUGIN_DATA}/"
        COMMENT "Copying Haar cascade to OBS plugin data directory"
    )
endif()

set(VOSK_MODELS_FOUND FALSE)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/vosk-model-small-en-us-0.15/conf/model.conf")
    set(VOSK_MODELS_FOUND TRUE)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/vosk-model-en-us-0.22/conf/model.conf")
    set(VOSK_MODELS_FOUND TRUE)
endif()
if(VOSK_MODELS_FOUND)
    add_custom_command(TARGET obs-face-captions POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models"
            "$<TARGET_FILE_DIR:obs-face-captions>/models"
        COMMENT "Copying Vosk model(s) to build output"
    )
    if(OBS_BUILD_DIR)
        set(OBS_PLUGIN_DATA "${OBS_BUILD_DIR}/rundir/RelWithDebInfo/data/obs-plugins/obs-face-captions")
        add_custom_command(TARGET obs-face-captions POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OBS_PLUGIN_DATA}/models"
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/models"
                "${OBS_PLUGIN_DATA}/models"
            COMMENT "Copying Vosk model(s) to OBS plugin data directory"
        )
    endif()
    message(STATUS "Bundled Vosk model(s) will be installed with the plugin")
else()
    message(STATUS "Run scripts/download_vosk_model.ps1 or download_vosk_model_large.ps1 to bundle a Vosk model")
endif()
