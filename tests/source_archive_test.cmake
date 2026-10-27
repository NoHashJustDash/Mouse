# SPDX-License-Identifier: GPL-3.0-or-later
foreach(required IN ITEMS MOUSE_SOURCE_DIRECTORY MOUSE_TEST_DIRECTORY MOUSE_GIT_EXECUTABLE
    MOUSE_TOOLCHAIN_FILE MOUSE_ORT_ROOT MOUSE_CXX_COMPILER MOUSE_MAKE_PROGRAM MOUSE_TRIPLET
    MOUSE_VCPKG_INSTALLED_DIR)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
if(NOT DEFINED MOUSE_SOURCE_REF)
  set(MOUSE_SOURCE_REF HEAD)
endif()
get_filename_component(cmake_binary_directory "${CMAKE_COMMAND}" DIRECTORY)
find_program(MOUSE_CTEST_COMMAND NAMES ctest HINTS "${cmake_binary_directory}" NO_DEFAULT_PATH REQUIRED)
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef unique)
set(test_root "${MOUSE_TEST_DIRECTORY}/source-archive-${unique}")
set(source "${test_root}/source")
set(binary "${test_root}/build")
file(MAKE_DIRECTORY "${source}")
function(run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE result
    OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    string(JOIN " " command ${ARGN})
    message(FATAL_ERROR "Command failed (${result}): ${command}\n${output}\n${error}")
  endif()
  set(last_output "${output}" PARENT_SCOPE)
endfunction()
function(build)
  run("${CMAKE_COMMAND}" --build "${binary}" --target mouse --parallel 2)
endfunction()
function(verify_version version hash)
  run("${binary}/mouse${MOUSE_EXECUTABLE_SUFFIX}" --version)
  string(REPLACE "\r" "" actual "${last_output}")
  foreach(expected IN ITEMS "Mouse ${version}\n" "git: ${hash}\n" "ONNX Runtime: 1.22.0\n")
    string(FIND "${actual}" "${expected}" index)
    if(index LESS 0)
      message(FATAL_ERROR "Expected '${expected}' in version output:\n${actual}")
    endif()
  endforeach()
  message(STATUS "Export/identity verified: version=${version} git=${hash}")
endfunction()
function(verify_unknown_cli_contract state)
  run("${MOUSE_CTEST_COMMAND}" --test-dir "${binary}" -R "^mouse.cli.version$" --output-on-failure)
  # Unknown must still fail when the caller has not explicitly enabled that contract.
  execute_process(COMMAND "${CMAKE_COMMAND}"
    "-DMOUSE_PROGRAM=${binary}/mouse${MOUSE_EXECUTABLE_SUFFIX}"
    "-DMOUSE_CASE=version" "-DMOUSE_ALLOW_UNKNOWN_IDENTITY=FALSE"
    -P "${source}/tests/cli_test.cmake"
    RESULT_VARIABLE strict_result OUTPUT_VARIABLE strict_output ERROR_VARIABLE strict_error)
  if(strict_result EQUAL 0 OR NOT strict_error MATCHES "Version output lacks git:")
    message(FATAL_ERROR "Default checkout contract accepted unknown or failed unexpectedly: ${strict_output}${strict_error}")
  endif()
  message(STATUS "Explicit unknown ${state}: configured CLI CTest passed; strict contract rejected sentinel")
endfunction()
set(configure "${CMAKE_COMMAND}" -S "${source}" -B "${binary}" -G Ninja
  "-DCMAKE_BUILD_TYPE=Release" "-DBUILD_TESTING=OFF"
  "-DCMAKE_CXX_COMPILER=${MOUSE_CXX_COMPILER}"
  "-DCMAKE_MAKE_PROGRAM=${MOUSE_MAKE_PROGRAM}"
  "-DCMAKE_TOOLCHAIN_FILE=${MOUSE_TOOLCHAIN_FILE}"
  "-DVCPKG_TARGET_TRIPLET=${MOUSE_TRIPLET}"
  "-DVCPKG_INSTALLED_DIR=${MOUSE_VCPKG_INSTALLED_DIR}"
  "-DMOUSE_ORT_ROOT=${MOUSE_ORT_ROOT}"
  "-DPython3_EXECUTABLE=${MOUSE_PYTHON_EXECUTABLE}")
# Write binary data directly to disk: piping archives through PowerShell corrupts them.
run("${MOUSE_GIT_EXECUTABLE}" -C "${MOUSE_SOURCE_DIRECTORY}" archive --format=tar
  "--output=${test_root}/source.tar" "${MOUSE_SOURCE_REF}")
# cmake -E tar extracts in its working directory, so use execute_process explicitly.
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar xf "${test_root}/source.tar"
  WORKING_DIRECTORY "${source}" RESULT_VARIABLE extract_result)
if(NOT extract_result EQUAL 0 OR EXISTS "${source}/.git")
  message(FATAL_ERROR "Cannot create a source archive fixture without .git")
endif()
run(${configure} "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE")
build()
verify_version(unknown unknown)
run(${configure} "-DCMAKE_DISABLE_FIND_PACKAGE_Git=TRUE"
  "-DMOUSE_VERSION=1.2.3-rc.4+archive.5" "-DMOUSE_GIT_HASH=0123456789abcdef")
build()
verify_version("1.2.3-rc.4+archive.5" 0123456789abcdef)

# A fresh fixture checkout lets the build itself prove tracked/untracked dirty
# changes are reflected without another configure step or touching the real repo.
run("${MOUSE_GIT_EXECUTABLE}" -c init.templateDir= -C "${source}" init --initial-branch=identity-test)
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" add --all)
file(MAKE_DIRECTORY "${test_root}/empty-hooks")
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" -c user.name=MouseBuildTest
  -c user.email=build-test@example.invalid "-c" "core.hooksPath=${test_root}/empty-hooks"
  -c commit.gpgsign=false commit --quiet -m "Identity regression fixture")
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" rev-parse --short=12 HEAD)
string(STRIP "${last_output}" fixture_hash)
run(${configure} "-DCMAKE_DISABLE_FIND_PACKAGE_Git=FALSE"
  "-DGIT_EXECUTABLE=${MOUSE_GIT_EXECUTABLE}" "-DMOUSE_VERSION=" "-DMOUSE_GIT_HASH=")
build()
file(READ "${source}/CMakeLists.txt" project_cmake)
string(REGEX MATCH "project\\(Mouse VERSION ([0-9]+\\.[0-9]+\\.[0-9]+)" match "${project_cmake}")
set(development_version "${CMAKE_MATCH_1}-dev")
verify_version("${development_version}" "${fixture_hash}")
file(APPEND "${source}/README.md" "\nBuild identity regression fixture.\n")
build()
verify_version("${development_version}" "${fixture_hash}-dirty")
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" restore -- README.md)
build()
verify_version("${development_version}" "${fixture_hash}")
file(WRITE "${source}/identity-untracked.txt" "Build identity regression fixture.\n")
build()
verify_version("${development_version}" "${fixture_hash}-dirty")
file(REMOVE "${source}/identity-untracked.txt")
build()
verify_version("${development_version}" "${fixture_hash}")
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" -c tag.gpgsign=false tag v9.8.7-rc.1+fixture.2)
build()
verify_version("9.8.7-rc.1+fixture.2" "${fixture_hash}")
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" tag --delete v9.8.7-rc.1+fixture.2)
build()
verify_version("${development_version}" "${fixture_hash}")
file(TIMESTAMP "${binary}/generated/engine/util/build_info.h" before "%Y-%m-%dT%H:%M:%S.%f" UTC)
build()
file(TIMESTAMP "${binary}/generated/engine/util/build_info.h" after "%Y-%m-%dT%H:%M:%S.%f" UTC)
if(NOT before STREQUAL after)
  message(FATAL_ERROR "An unchanged build rewrote the identity header")
endif()
# Explicit unknown is a supported packager override inside a checkout, too.
# Run the actual configured CLI CTest in both clean and dirty fixture states.
run(${configure} "-DCMAKE_DISABLE_FIND_PACKAGE_Git=FALSE"
  "-DGIT_EXECUTABLE=${MOUSE_GIT_EXECUTABLE}" "-DBUILD_TESTING=ON"
  "-DMOUSE_VERSION=" "-DMOUSE_GIT_HASH=unknown")
build()
verify_version("${development_version}" unknown)
verify_unknown_cli_contract(clean)
file(APPEND "${source}/README.md" "\nExplicit unknown checkout regression fixture.\n")
build()
verify_version("${development_version}" unknown-dirty)
verify_unknown_cli_contract(dirty)
run("${MOUSE_GIT_EXECUTABLE}" -C "${source}" restore -- README.md)
message(STATUS "Explicit unknown checkout override passed clean and dirty CLI CTest regressions")
message(STATUS "Source archive and dirty-build regressions passed; evidence fixture: ${test_root}")
