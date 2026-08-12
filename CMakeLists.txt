cmake_minimum_required(VERSION 3.25)

project(animate-any-mesh.cpp VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
if(IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
    get_filename_component(AAM_INSTALL_LIBDIR_NAME "${CMAKE_INSTALL_LIBDIR}" NAME)
else()
    set(AAM_INSTALL_LIBDIR_NAME "${CMAKE_INSTALL_LIBDIR}")
endif()
set(CMAKE_INSTALL_RPATH "$ORIGIN/../${AAM_INSTALL_LIBDIR_NAME};$ORIGIN")

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AAM_BUILD_SHARED "Build libanimate_any_mesh as a shared library" ON)
option(AAM_BUILD_CLI "Build the aam-cli executable" ON)
option(AAM_BUILD_TESTS "Build the test suite" ON)
option(AAM_VULKAN "Build the ggml Vulkan backend" ON)
option(AAM_DYNAMIC_BACKENDS "Build ggml backends as dynamically loaded libraries" ON)
option(AAM_CPU_ALL_VARIANTS "Build all supported ggml CPU variants" ON)
option(AAM_WARNINGS_AS_ERRORS "Treat project warnings as errors" OFF)
option(AAM_SANITIZE "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(AAM_FUZZ "Build libFuzzer harnesses (Clang only)" OFF)
option(AAM_USE_EXTERNAL_GGML "Use an installed ggml CMake package" OFF)
set(AAM_GGML_SOURCE_DIR "" CACHE PATH "Path to a pinned ggml source tree")

if(NOT AAM_GGML_SOURCE_DIR AND DEFINED ENV{AAM_GGML_SOURCE_DIR})
    set(AAM_GGML_SOURCE_DIR "$ENV{AAM_GGML_SOURCE_DIR}")
endif()

if(AAM_SANITIZE OR AAM_FUZZ)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "AAM_SANITIZE/AAM_FUZZ require Clang")
    endif()
    # Instrument this project and all direct consumers, but not pinned ggml:
    # ggml deliberately performs no-allocation tensor pointer arithmetic and
    # uses ISA-dispatch function casts that trip Clang's generic UBSan checks.
    add_library(aam_sanitizers INTERFACE)
    target_compile_options(aam_sanitizers INTERFACE
        -fno-omit-frame-pointer -fsanitize=address,undefined)
    target_link_options(aam_sanitizers INTERFACE
        -fno-omit-frame-pointer -fsanitize=address,undefined)
    if(AAM_FUZZ)
        # Instrument library code for coverage while leaving the fuzzer's main
        # function to each harness executable.
        target_compile_options(aam_sanitizers INTERFACE -fsanitize=fuzzer-no-link)
    endif()
endif()

if(AAM_USE_EXTERNAL_GGML)
    find_package(ggml CONFIG REQUIRED)
    set(AAM_GGML_TARGET ggml::ggml)
else()
    if(NOT AAM_GGML_SOURCE_DIR)
        if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/ggml/CMakeLists.txt")
            set(AAM_GGML_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
        else()
            message(FATAL_ERROR
                "ggml was not found. Use the Nix shell, initialize the ggml submodule, "
                "or set -DAAM_GGML_SOURCE_DIR=/path/to/ggml")
        endif()
    endif()

    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_VULKAN ${AAM_VULKAN} CACHE BOOL "" FORCE)
    set(GGML_BACKEND_DL ${AAM_DYNAMIC_BACKENDS} CACHE BOOL "" FORCE)
    set(GGML_CPU_ALL_VARIANTS ${AAM_CPU_ALL_VARIANTS} CACHE BOOL "" FORCE)
    if(AAM_DYNAMIC_BACKENDS)
        set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    endif()
    add_subdirectory("${AAM_GGML_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/ggml" EXCLUDE_FROM_ALL)
    set(AAM_GGML_TARGET ggml)
endif()

find_package(Threads REQUIRED)
find_package(meshoptimizer CONFIG REQUIRED)
find_package(nlohmann_json 3.11 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(UTF8PROC REQUIRED IMPORTED_TARGET libutf8proc)

set(AAM_SOURCES
    src/backend.cpp
    src/clip.cpp
    src/export.cpp
    src/graph.cpp
    src/mesh.cpp
    src/model.cpp
    src/rf.cpp
    src/tokenizer.cpp
    src/vae.cpp
)

if(AAM_BUILD_SHARED)
    add_library(animate_any_mesh SHARED ${AAM_SOURCES})
    target_compile_definitions(animate_any_mesh PRIVATE AAM_BUILD AAM_SHARED)
    target_compile_definitions(animate_any_mesh PUBLIC AAM_SHARED)
else()
    add_library(animate_any_mesh STATIC ${AAM_SOURCES})
endif()

add_library(aam::aam ALIAS animate_any_mesh)
target_include_directories(animate_any_mesh
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE src
)
target_link_libraries(animate_any_mesh
    PRIVATE $<BUILD_INTERFACE:${AAM_GGML_TARGET}>
            Threads::Threads meshoptimizer::meshoptimizer nlohmann_json::nlohmann_json
            PkgConfig::UTF8PROC
)
if(TARGET aam_sanitizers)
    target_link_libraries(animate_any_mesh PUBLIC aam_sanitizers)
endif()
target_compile_features(animate_any_mesh PUBLIC cxx_std_23)
set_target_properties(animate_any_mesh PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    POSITION_INDEPENDENT_CODE ON
)

if(MSVC)
    target_compile_options(animate_any_mesh PRIVATE /W4 /permissive- /sdl)
else()
    target_compile_options(animate_any_mesh PRIVATE
        -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wshadow
        -Wformat=2 -Wundef -Wcast-align -Wnull-dereference
    )
    if(AAM_WARNINGS_AS_ERRORS)
        target_compile_options(animate_any_mesh PRIVATE -Werror)
    endif()
    if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
        target_compile_options(animate_any_mesh PRIVATE -fstack-protector-strong)
        target_compile_definitions(animate_any_mesh PRIVATE _FORTIFY_SOURCE=3)
    endif()
endif()

if(AAM_BUILD_CLI)
    add_executable(aam-cli src/cli.cpp)
    target_link_libraries(aam-cli PRIVATE aam::aam)
    # ggml's dynamic backend loader searches beside the executable. Keep the
    # CLI next to the generated CPU/Vulkan modules in build and install trees.
    set_target_properties(aam-cli PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

include(CTest)
if(AAM_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(AAM_FUZZ)
    add_subdirectory(fuzz)
endif()

install(TARGETS animate_any_mesh
    EXPORT animate-any-mesh-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
if(AAM_BUILD_CLI)
    install(TARGETS aam-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(AAM_DYNAMIC_BACKENDS AND NOT AAM_USE_EXTERNAL_GGML)
    # Install module targets through CMake so their build RPATH is rewritten.
    # A raw directory copy would leak the build directory into Nix packages.
    function(aam_collect_ggml_modules directory output)
        get_property(local_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
        get_property(subdirectories DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
        set(modules)
        foreach(candidate IN LISTS local_targets)
            get_target_property(candidate_type ${candidate} TYPE)
            if(candidate_type STREQUAL "MODULE_LIBRARY" AND
               candidate MATCHES "^ggml-(cpu|vulkan)")
                list(APPEND modules ${candidate})
            endif()
        endforeach()
        foreach(subdirectory IN LISTS subdirectories)
            aam_collect_ggml_modules("${subdirectory}" nested_modules)
            list(APPEND modules ${nested_modules})
        endforeach()
        set(${output} "${modules}" PARENT_SCOPE)
    endfunction()
    aam_collect_ggml_modules("${AAM_GGML_SOURCE_DIR}" AAM_GGML_MODULE_TARGETS)
    if(NOT AAM_GGML_MODULE_TARGETS)
        message(FATAL_ERROR "dynamic GGML build produced no loadable backend modules")
    endif()
    get_target_property(AAM_GGML_PUBLIC_HEADERS ggml PUBLIC_HEADER)
    set_target_properties(ggml PROPERTIES PUBLIC_HEADER "")
    install(TARGETS ggml ggml-base LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
    set_target_properties(ggml PROPERTIES PUBLIC_HEADER "${AAM_GGML_PUBLIC_HEADERS}")
    install(TARGETS ${AAM_GGML_MODULE_TARGETS}
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
