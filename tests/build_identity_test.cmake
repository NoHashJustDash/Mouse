# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT DEFINED MOUSE_SOURCE_DIRECTORY OR NOT DEFINED MOUSE_TEST_DIRECTORY)
  message(FATAL_ERROR "MOUSE_SOURCE_DIRECTORY and MOUSE_TEST_DIRECTORY are required")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef unique)
set(test_root "${MOUSE_TEST_DIRECTORY}/metadata-validation-${unique}")
file(MAKE_DIRECTORY "${test_root}/source")
function(check version hash expected_success)
  file(WRITE "${test_root}/config.cmake"
    "set(MOUSE_SOURCE_DIRECTORY [==[${test_root}/source]==])\n"
    "set(MOUSE_TEMPLATE [==[${MOUSE_SOURCE_DIRECTORY}/cmake/build_info.h.in]==])\n"
    "set(MOUSE_OUTPUT_HEADER [==[${test_root}/build_info.h]==])\n"
    "set(MOUSE_VERSION_OVERRIDE [==[${version}]==])\n"
    "set(MOUSE_GIT_HASH_OVERRIDE [==[${hash}]==])\n")
  execute_process(COMMAND "${CMAKE_COMMAND}" "-DMOUSE_IDENTITY_CONFIG=${test_root}/config.cmake"
    -P "${MOUSE_SOURCE_DIRECTORY}/cmake/GenerateBuildInfo.cmake"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(expected_success AND NOT result EQUAL 0)
    message(FATAL_ERROR "Valid metadata was rejected: ${version}, ${hash}\n${output}${error}")
  elseif(NOT expected_success AND result EQUAL 0)
    message(FATAL_ERROR "Invalid metadata was accepted: ${version}, ${hash}")
  endif()
endfunction()
check("" "" TRUE)
file(READ "${test_root}/build_info.h" generated)
if(NOT generated MATCHES "version = \"unknown\"" OR NOT generated MATCHES "git_hash = \"unknown\"")
  message(FATAL_ERROR "Archive defaults must be unknown")
endif()
check("1.2.3-rc.1+source.5" "0123456789abcdef-dirty" TRUE)
check("unknown" "unknown" TRUE)
check("1.2.3+build-01" "ABCDEF01234567" TRUE)
check("0.0.0" "1234567" TRUE)
foreach(version IN ITEMS "v1.2.3" "1.2" "01.2.3" "1.2.3-" "1.2.3-01" "1.2.3+" "1.2.3+bad..build")
  check("${version}" "0123456789ab" FALSE)
endforeach()
foreach(hash IN ITEMS "abc" "not-a-commit" "abcdefg" "1234567-dirty-dirty")
  check("1.2.3" "${hash}" FALSE)
endforeach()
message(STATUS "Build metadata validation: defaults and valid overrides accepted; 11 invalid cases rejected")
