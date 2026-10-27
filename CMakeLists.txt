# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.28)
project(Mouse VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)
include(cmake/CompilerOptions.cmake)
include(cmake/OnnxRuntime.cmake)

option(MOUSE_USE_PEXT "Use BMI2 PEXT sliding attacks (CPU support is checked explicitly)" OFF)
option(MOUSE_ENABLE_GPU_TESTS "Register local GPU provider tests (requires explicit SDK directories)" OFF)
option(MOUSE_CLANG_TIDY "Run clang-tidy while compiling project sources" OFF)
set(MOUSE_SANITIZER "" CACHE STRING "Instrumentation: address or thread")
set_property(CACHE MOUSE_SANITIZER PROPERTY STRINGS "" address thread)
# Empty cache values select checkout metadata, or explicit unknown for an archive.
get_property(old_version_help CACHE MOUSE_VERSION PROPERTY HELPSTRING)
if(old_version_help STREQUAL "Development version when HEAD has no release tag"
   AND MOUSE_VERSION STREQUAL "${PROJECT_VERSION}-dev")
  unset(MOUSE_VERSION CACHE)
endif()
set(MOUSE_VERSION "" CACHE STRING "Optional SemVer build identity override; archives default to unknown")
set(MOUSE_GIT_HASH "" CACHE STRING "Optional hexadecimal commit identity override; archives default to unknown")
if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
  find_package(Git REQUIRED)
  set(MOUSE_SOURCE_IS_ARCHIVE FALSE)
else()
  set(MOUSE_SOURCE_IS_ARCHIVE TRUE)
endif()
if(NOT CMAKE_BUILD_TYPE)
  message(FATAL_ERROR "Set CMAKE_BUILD_TYPE or use a documented configure preset")
endif()
set(MOUSE_COMPILER "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
include(cmake/ValidateBuildIdentity.cmake)
mouse_validate_version("${MOUSE_VERSION}")
mouse_validate_hash("${MOUSE_GIT_HASH}")
configure_file(cmake/build_identity.cmake.in generated/build_identity.cmake @ONLY)
set(mouse_identity_command "${CMAKE_COMMAND}"
  "-DMOUSE_IDENTITY_CONFIG=${PROJECT_BINARY_DIR}/generated/build_identity.cmake"
  -P "${PROJECT_SOURCE_DIR}/cmake/GenerateBuildInfo.cmake")
# Generate once for compiler dependency scanning, then refresh before every build.
execute_process(COMMAND ${mouse_identity_command} COMMAND_ERROR_IS_FATAL ANY)
add_custom_target(mouse_build_identity ALL
  COMMAND ${mouse_identity_command}
  BYPRODUCTS "${PROJECT_BINARY_DIR}/generated/engine/util/build_info.h"
  COMMENT "Refreshing source version and dirty identity"
  VERBATIM)
find_package(tl-expected CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)
find_package(Threads REQUIRED)
add_library(mouse_util STATIC engine/util/runtime_library.cpp engine/util/sha256.cpp engine/util/hash128.cpp)
target_include_directories(mouse_util PUBLIC "${PROJECT_SOURCE_DIR}" "${PROJECT_BINARY_DIR}/generated")
target_include_directories(mouse_util SYSTEM PUBLIC "${MOUSE_ORT_INCLUDE_DIR}")
target_link_libraries(mouse_util PUBLIC tl::expected PRIVATE "${CMAKE_DL_LIBS}")
mouse_target_options(mouse_util)

add_library(mouse_chess STATIC
  engine/chess/attacks.cpp
  engine/chess/types.cpp
  engine/chess/position.cpp
  engine/chess/fen.cpp
  engine/chess/movegen.cpp
  engine/chess/perft.cpp
  engine/chess/zobrist.cpp)
target_include_directories(mouse_chess PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(mouse_chess PUBLIC tl::expected)
target_compile_definitions(mouse_chess PUBLIC MOUSE_USE_PEXT=$<BOOL:${MOUSE_USE_PEXT}>)
mouse_target_options(mouse_chess)
add_library(mouse_nn STATIC engine/nn/encoder.cpp engine/nn/policy_map.cpp
  engine/nn/history.cpp engine/nn/model_metadata.cpp engine/nn/model_artifact.cpp engine/nn/commands.cpp engine/nn/output.cpp
  engine/nn/backend.cpp engine/nn/backend_gpu.cpp engine/nn/cache.cpp
  engine/nn/cached_backend.cpp engine/nn/legal_output.cpp engine/nn/legal_cache.cpp engine/nn/legal_evaluator.cpp engine/nn/command_runner.cpp)
target_include_directories(mouse_nn PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(mouse_nn PUBLIC mouse_chess mouse_util PRIVATE Threads::Threads)
# Header-only vendor API declarations give dynamic GPU calls their exact ABI.
# No CUDA language, NVCC, host compiler downgrade or CUDA import link is used.
find_path(MOUSE_CUDA_INCLUDE_DIR NAMES cuda.h HINTS "$ENV{CUDA_PATH}/include"
  DOC "CUDA 12.8 SDK include directory for the optional explicit GPU backends")
if(MOUSE_CUDA_INCLUDE_DIR)
  foreach(header IN ITEMS cuda_runtime_api.h nvml.h)
    if(NOT EXISTS "${MOUSE_CUDA_INCLUDE_DIR}/${header}")
      message(FATAL_ERROR "CUDA SDK include directory lacks ${header}")
    endif()
  endforeach()
  target_include_directories(mouse_nn SYSTEM PRIVATE "${MOUSE_CUDA_INCLUDE_DIR}")
  target_compile_definitions(mouse_nn PRIVATE MOUSE_HAS_CUDA_HEADERS=1)
else()
  target_compile_definitions(mouse_nn PRIVATE MOUSE_HAS_CUDA_HEADERS=0)
endif()
mouse_target_options(mouse_nn)
add_library(mouse_search STATIC
  engine/search/tree.cpp engine/search/proof.cpp engine/search/moves_left.cpp engine/search/prior.cpp engine/search/selection.cpp engine/search/pruning.cpp engine/search/collisions.cpp
  engine/search/rules.cpp engine/search/commands.cpp engine/search/search.cpp engine/search/command_runner.cpp)
target_include_directories(mouse_search PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(mouse_search PUBLIC mouse_nn PRIVATE Threads::Threads)
mouse_target_options(mouse_search)
add_library(mouse_uci STATIC engine/uci/commands.cpp engine/uci/emergency.cpp engine/uci/score.cpp engine/uci/options.cpp engine/uci/time.cpp engine/uci/session.cpp
    engine/uci/report.cpp engine/uci/input.cpp engine/uci/command_runner.cpp engine/uci/bench.cpp)
target_include_directories(mouse_uci PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(mouse_uci PUBLIC mouse_search)
if(WIN32)
  target_link_libraries(mouse_uci PRIVATE winmm)
endif()
mouse_target_options(mouse_uci)
add_executable(mouse engine/main.cpp)
target_link_libraries(mouse PRIVATE mouse_util mouse_chess mouse_nn mouse_search mouse_uci)
add_dependencies(mouse mouse_build_identity)
mouse_target_options(mouse)
mouse_copy_ort_runtime(mouse)

if(BUILD_TESTING)
  if(MOUSE_SANITIZER STREQUAL "address")
    add_executable(mouse_ubsan_probe tests/toolchain/ubsan_probe.cpp)
    mouse_target_options(mouse_ubsan_probe)
    add_executable(mouse_lsan_probe tests/toolchain/lsan_probe.cpp)
    target_link_libraries(mouse_lsan_probe PRIVATE mouse_util)
    target_compile_definitions(mouse_lsan_probe PRIVATE
      MOUSE_TEST_RUNTIME_PATH="$<TARGET_FILE_DIR:mouse_lsan_probe>/${MOUSE_ORT_RUNTIME_FILENAME}")
    mouse_target_options(mouse_lsan_probe)
    mouse_copy_ort_runtime(mouse_lsan_probe)
    add_test(NAME mouse.toolchain.leak-detection COMMAND "${CMAKE_COMMAND}"
      "-DMOUSE_LSAN_PROBE=$<TARGET_FILE:mouse_lsan_probe>"
      -P "${PROJECT_SOURCE_DIR}/tests/leak_detection_test.cmake")
    set_tests_properties(mouse.toolchain.leak-detection PROPERTIES LABELS "cpu;toolchain;regression" TIMEOUT 60)
    add_test(NAME mouse.toolchain.ubsan-fatal COMMAND "${CMAKE_COMMAND}"
      "-DMOUSE_UBSAN_PROBE=$<TARGET_FILE:mouse_ubsan_probe>"
      "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/ubsan-policy"
      -P "${PROJECT_SOURCE_DIR}/tests/sanitizer_fatal_test.cmake")
    set_tests_properties(mouse.toolchain.ubsan-fatal PROPERTIES LABELS "cpu;toolchain;regression" TIMEOUT 60)
  endif()
  add_library(mouse_wrong_ort SHARED tests/util/wrong_ort_library.cpp)
  target_include_directories(mouse_wrong_ort SYSTEM PRIVATE "${MOUSE_ORT_INCLUDE_DIR}")
  mouse_target_options(mouse_wrong_ort)
  set_target_properties(mouse_wrong_ort PROPERTIES PREFIX "" WINDOWS_EXPORT_ALL_SYMBOLS ON)

  add_executable(mouse_tests tests/util/runtime_library_test.cpp tests/util/sha256_test.cpp tests/util/hash128_test.cpp tests/gpu/runtime_gpu_test.cpp
    tests/chess/attacks_test.cpp tests/chess/position_test.cpp tests/chess/perft_suite_test.cpp tests/nn/encoder_test.cpp tests/nn/policy_map_test.cpp tests/nn/history_test.cpp tests/nn/model_metadata_test.cpp tests/nn/model_artifact_test.cpp tests/nn/commands_test.cpp tests/nn/backend_test.cpp tests/nn/cache_test.cpp tests/nn/cached_backend_test.cpp tests/nn/legal_cache_test.cpp
    tests/search/tree_test.cpp tests/search/proof_test.cpp tests/search/proof_search_test.cpp tests/search/moves_left_test.cpp tests/search/prior_test.cpp
    tests/search/selection_test.cpp tests/search/pruning_test.cpp tests/search/collisions_test.cpp tests/search/rules_test.cpp tests/search/commands_test.cpp tests/search/search_test.cpp tests/search/pipeline_test.cpp tests/uci/commands_test.cpp tests/uci/score_test.cpp tests/uci/options_test.cpp tests/uci/session_test.cpp
        tests/uci/report_test.cpp tests/uci/time_test.cpp tests/uci/input_test.cpp tests/uci/input_request_test.cpp tests/uci/bench_test.cpp)
  target_link_libraries(mouse_tests PRIVATE mouse_util mouse_chess mouse_nn mouse_search mouse_uci Catch2::Catch2WithMain)
  mouse_target_options(mouse_tests)
  mouse_copy_ort_runtime(mouse_tests)
  add_dependencies(mouse_tests mouse_wrong_ort)
  # Explicit CPU microbenchmark for the NN cache; no timing threshold in CTest.
  add_executable(mouse_cache_benchmark tests/nn/cache_benchmark.cpp)
  # Real-model cache equality and NPS check; invoked locally with T1.
  add_executable(mouse_cache_equivalence tests/nn/cache_equivalence.cpp)
  target_link_libraries(mouse_cache_equivalence PRIVATE mouse_search)
  mouse_target_options(mouse_cache_equivalence)
  mouse_copy_ort_runtime(mouse_cache_equivalence)
  target_link_libraries(mouse_cache_benchmark PRIVATE mouse_nn)
  target_include_directories(mouse_cache_benchmark SYSTEM PRIVATE "${MOUSE_ORT_INCLUDE_DIR}")
  mouse_target_options(mouse_cache_benchmark)
  add_executable(mouse_encoder_probe tests/nn/encoder_probe.cpp)
  target_link_libraries(mouse_encoder_probe PRIVATE mouse_nn)
  mouse_target_options(mouse_encoder_probe)

  target_compile_definitions(mouse_tests PRIVATE
    MOUSE_TEST_RUNTIME_PATH="$<TARGET_FILE_DIR:mouse_tests>/${MOUSE_ORT_RUNTIME_FILENAME}"
    MOUSE_TEST_WRONG_RUNTIME_PATH="$<TARGET_FILE:mouse_wrong_ort>"
    MOUSE_TEST_PERFT_SUITE_PATH="${PROJECT_SOURCE_DIR}/tests/data/perft.epd"
    MOUSE_TEST_METADATA_DIRECTORY="${PROJECT_SOURCE_DIR}/tests/data/model-metadata"
    MOUSE_TEST_BACKEND_DIRECTORY="${PROJECT_SOURCE_DIR}/tests/data/backend"
    MOUSE_TEST_BENCH_DIRECTORY="${PROJECT_SOURCE_DIR}/tests/data/bench"
    MOUSE_TEST_MODEL_PATH="${PROJECT_SOURCE_DIR}/tools/tests/data/toolchain.onnx")
  add_executable(mouse_root_reset_allocation_test tests/search/root_reset_allocation_test.cpp)
  target_link_libraries(mouse_root_reset_allocation_test PRIVATE mouse_search)
  mouse_target_options(mouse_root_reset_allocation_test)
  if(MOUSE_SANITIZER STREQUAL "thread" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # The allocation auditor supplies instrumented new/delete replacements.
    # Static tsan_cxx defines them strongly; shared TSan permits normal ELF
    # interposition while keeping core race and malloc/free instrumentation.
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --print-file-name=libclang_rt.tsan-x86_64.so
      OUTPUT_VARIABLE mouse_tsan_library OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
    if(NOT IS_ABSOLUTE "${mouse_tsan_library}" OR NOT EXISTS "${mouse_tsan_library}")
      message(FATAL_ERROR "Clang's shared TSan runtime is required: ${mouse_tsan_library}")
    endif()
    get_filename_component(mouse_tsan_directory "${mouse_tsan_library}" DIRECTORY)
    target_link_options(mouse_root_reset_allocation_test PRIVATE -shared-libsan)
    set_property(TARGET mouse_root_reset_allocation_test APPEND PROPERTY BUILD_RPATH
      "${mouse_tsan_directory}")
  endif()
  add_test(NAME mouse.search.root-reset-allocation COMMAND mouse_root_reset_allocation_test)
  set_tests_properties(mouse.search.root-reset-allocation PROPERTIES LABELS "cpu;search;regression" TIMEOUT 60)
  add_test(NAME mouse.unit COMMAND mouse_tests "~[gpu]~[.long]~[perft-gate]")
  set_tests_properties(mouse.unit PROPERTIES LABELS "cpu;unit" TIMEOUT 120)
  if(WIN32)
    find_program(MOUSE_POWERSHELL_EXECUTABLE NAMES powershell.exe REQUIRED)
    add_test(NAME mouse.toolchain.developer-shell COMMAND "${MOUSE_POWERSHELL_EXECUTABLE}"
      -NoProfile -ExecutionPolicy Bypass -File "${PROJECT_SOURCE_DIR}/scripts/test-developer-shell.ps1")
    set_tests_properties(mouse.toolchain.developer-shell PROPERTIES LABELS "cpu;toolchain" TIMEOUT 120)
  endif()
  set(MOUSE_ALLOW_UNKNOWN_IDENTITY "${MOUSE_SOURCE_IS_ARCHIVE}")
  if(MOUSE_GIT_HASH STREQUAL "unknown")
    set(MOUSE_ALLOW_UNKNOWN_IDENTITY TRUE)
  endif()
  foreach(case IN ITEMS version help invalid missing wrong)
    add_test(NAME "mouse.cli.${case}" COMMAND "${CMAKE_COMMAND}"
      "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
      "-DMOUSE_CASE=${case}"
      "-DMOUSE_ALLOW_UNKNOWN_IDENTITY=${MOUSE_ALLOW_UNKNOWN_IDENTITY}"
      "-DMOUSE_RUNTIME_FILENAME=${MOUSE_ORT_RUNTIME_FILENAME}"
      "-DMOUSE_WRONG_RUNTIME=$<TARGET_FILE:mouse_wrong_ort>"
      "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/cli-tests"
      -P "${PROJECT_SOURCE_DIR}/tests/cli_test.cmake")
    set_tests_properties("mouse.cli.${case}" PROPERTIES LABELS "cpu;cli" TIMEOUT 60)
  endforeach()
  add_test(NAME mouse.cli.perft COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>" -P "${PROJECT_SOURCE_DIR}/tests/perft_cli_test.cmake")
  set_tests_properties(mouse.cli.perft PROPERTIES LABELS "cpu;cli;chess" TIMEOUT 120)
  add_test(NAME mouse.cli.describe-net COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
    "-DMOUSE_FIXTURE_DIRECTORY=${PROJECT_SOURCE_DIR}/tests/data/model-metadata"
    -P "${PROJECT_SOURCE_DIR}/tests/describe_net_cli_test.cmake")
  set_tests_properties(mouse.cli.describe-net PROPERTIES LABELS "cpu;cli;nn" TIMEOUT 120)
  add_test(NAME mouse.cli.backend COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
    "-DMOUSE_FIXTURE_DIRECTORY=${PROJECT_SOURCE_DIR}/tests/data/backend"
    "-DMOUSE_INPUTS=${PROJECT_SOURCE_DIR}/tests/data/nn/backend-parity-v1/planes.bin"
    "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/cli-tests"
    -P "${PROJECT_SOURCE_DIR}/tests/backend_cli_test.cmake")
  set_tests_properties(mouse.cli.backend PROPERTIES LABELS "cpu;cli;nn" TIMEOUT 180)
  add_test(NAME mouse.cli.search COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
    "-DMOUSE_FIXTURE_DIRECTORY=${PROJECT_SOURCE_DIR}/tests/data/backend"
    -P "${PROJECT_SOURCE_DIR}/tests/search_cli_test.cmake")
  set_tests_properties(mouse.cli.search PROPERTIES LABELS "cpu;cli;search" TIMEOUT 300)
  add_test(NAME mouse.cli.bench COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
    "-DMOUSE_FIXTURE_DIRECTORY=${PROJECT_SOURCE_DIR}/tests/data/backend"
    -P "${PROJECT_SOURCE_DIR}/tests/bench_cli_test.cmake")
  set_tests_properties(mouse.cli.bench PROPERTIES LABELS "cpu;cli;uci;bench" TIMEOUT 180)
  find_package(Python3 3.12...<3.13 REQUIRED COMPONENTS Interpreter)
  add_test(NAME mouse.uci.protocol COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
    "-DMOUSE_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
    "-DMOUSE_SOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
    "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/uci-protocol"
    -P "${PROJECT_SOURCE_DIR}/tests/uci_protocol_test.cmake")
  set_tests_properties(mouse.uci.protocol PROPERTIES
    LABELS "cpu;uci;golden;integration" TIMEOUT 1500)
  add_test(NAME mouse.uci.options-docs COMMAND "${Python3_EXECUTABLE}" -I -c
    "import sys; sys.path.insert(0, sys.argv[1]); from mouse_tools.uci_options import main; raise SystemExit(main(sys.argv[2:]))"
    "${PROJECT_SOURCE_DIR}/tools/src"
    --binary "$<TARGET_FILE:mouse>" --readme "${PROJECT_SOURCE_DIR}/README.md" --check)
  set_tests_properties(mouse.uci.options-docs PROPERTIES LABELS "cpu;uci;docs" TIMEOUT 30)
  add_test(NAME mouse.chess.perft.full COMMAND mouse_tests "[perft-gate]")
  set_tests_properties(mouse.chess.perft.full PROPERTIES LABELS "cpu;chess;perft;gate" TIMEOUT 14400)
  add_test(NAME mouse.ort-runtime.packaging COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_SOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
    "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/integration-tests"
    "-DMOUSE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
    -P "${PROJECT_SOURCE_DIR}/tests/onnx_runtime_packaging_test.cmake")
  set_tests_properties(mouse.ort-runtime.packaging PROPERTIES LABELS "cpu;packaging" TIMEOUT 120)
  add_test(NAME mouse.build-identity.validation COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_SOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
    "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/integration-tests"
    -P "${PROJECT_SOURCE_DIR}/tests/build_identity_test.cmake")
  set_tests_properties(mouse.build-identity.validation PROPERTIES LABELS "cpu;metadata" TIMEOUT 120)
  if(EXISTS "${PROJECT_SOURCE_DIR}/.git")
    add_test(NAME mouse.source-archive COMMAND "${CMAKE_COMMAND}"
      "-DMOUSE_SOURCE_DIRECTORY=${PROJECT_SOURCE_DIR}"
      "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/integration-tests"
      "-DMOUSE_GIT_EXECUTABLE=${GIT_EXECUTABLE}"
      "-DMOUSE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
      "-DMOUSE_ORT_ROOT=${MOUSE_ORT_ROOT}"
      "-DMOUSE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DMOUSE_MAKE_PROGRAM=${CMAKE_MAKE_PROGRAM}"
      "-DMOUSE_TRIPLET=${VCPKG_TARGET_TRIPLET}"
      "-DMOUSE_VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR}"
      "-DMOUSE_EXECUTABLE_SUFFIX=${CMAKE_EXECUTABLE_SUFFIX}"
      "-DMOUSE_PYTHON_EXECUTABLE=${Python3_EXECUTABLE}"
      -P "${PROJECT_SOURCE_DIR}/tests/source_archive_test.cmake")
    set_tests_properties(mouse.source-archive PROPERTIES LABELS "cpu;integration" TIMEOUT 600 RUN_SERIAL TRUE)
  endif()
  if(MOUSE_ENABLE_GPU_TESTS)
    add_test(NAME mouse.gpu.providers COMMAND mouse_tests "[gpu]")
    set_tests_properties(mouse.gpu.providers PROPERTIES
      LABELS "gpu" TIMEOUT 600 RUN_SERIAL TRUE RESOURCE_LOCK mouse_gpu)
    add_test(NAME mouse.gpu.unavailable COMMAND "${CMAKE_COMMAND}"
      "-DMOUSE_PROGRAM=$<TARGET_FILE:mouse>"
      "-DMOUSE_MODEL=${PROJECT_SOURCE_DIR}/tests/data/backend/valid.onnx"
      "-DMOUSE_INPUTS=${PROJECT_SOURCE_DIR}/tests/data/nn/backend-parity-v1/planes.bin"
      "-DMOUSE_TEST_DIRECTORY=${PROJECT_BINARY_DIR}/gpu-unavailable"
      -P "${PROJECT_SOURCE_DIR}/tests/gpu/unavailable_gpu_cli_test.cmake")
    set_tests_properties(mouse.gpu.unavailable PROPERTIES
      LABELS "gpu;errors" TIMEOUT 300 RUN_SERIAL TRUE RESOURCE_LOCK mouse_gpu)
  endif()
endif()

install(TARGETS mouse RUNTIME DESTINATION .)
install(FILES ${MOUSE_ORT_RUNTIME_FILES} DESTINATION .)
install(FILES "${MOUSE_ORT_ROOT}/LICENSE" DESTINATION licenses RENAME ONNXRuntime-LICENSE)
install(FILES "${MOUSE_ORT_ROOT}/ThirdPartyNotices.txt" DESTINATION licenses)

install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/README.md"
  "${PROJECT_SOURCE_DIR}/CHANGELOG.md" DESTINATION .)
install(FILES "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/tl-expected/copyright"
  DESTINATION licenses RENAME tl-expected-LICENSE)
