# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT DEFINED MOUSE_IDENTITY_CONFIG)
  message(FATAL_ERROR "MOUSE_IDENTITY_CONFIG is required")
endif()
include("${MOUSE_IDENTITY_CONFIG}")

include("${CMAKE_CURRENT_LIST_DIR}/ValidateBuildIdentity.cmake")

mouse_validate_version("${MOUSE_VERSION_OVERRIDE}")
mouse_validate_hash("${MOUSE_GIT_HASH_OVERRIDE}")
set(MOUSE_IDENTITY_VERSION "${MOUSE_VERSION_OVERRIDE}")
set(MOUSE_IDENTITY_HASH "${MOUSE_GIT_HASH_OVERRIDE}")

# An archive may live below an unrelated checkout. Never discover that parent.
if(EXISTS "${MOUSE_SOURCE_DIRECTORY}/.git")
  if(NOT EXISTS "${MOUSE_GIT_EXECUTABLE}")
    message(FATAL_ERROR "Git is needed to identify this checkout; source archives do not require it")
  endif()
  execute_process(COMMAND "${MOUSE_GIT_EXECUTABLE}" -C "${MOUSE_SOURCE_DIRECTORY}" rev-parse --short=12 HEAD
    RESULT_VARIABLE result OUTPUT_VARIABLE commit ERROR_VARIABLE error OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot identify the source checkout: ${error}")
  endif()
  if(MOUSE_IDENTITY_HASH STREQUAL "")
    set(MOUSE_IDENTITY_HASH "${commit}")
  endif()
  if(MOUSE_IDENTITY_VERSION STREQUAL "")
    execute_process(COMMAND "${MOUSE_GIT_EXECUTABLE}" -C "${MOUSE_SOURCE_DIRECTORY}" describe --tags --exact-match HEAD
      RESULT_VARIABLE result OUTPUT_VARIABLE tag ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(result EQUAL 0 AND tag MATCHES "^v[0-9]")
      string(SUBSTRING "${tag}" 1 -1 candidate)
      mouse_validate_version("${candidate}")
      set(MOUSE_IDENTITY_VERSION "${candidate}")
    else()
      set(MOUSE_IDENTITY_VERSION "${MOUSE_PROJECT_VERSION}-dev")
    endif()
  endif()
  execute_process(COMMAND "${MOUSE_GIT_EXECUTABLE}" -C "${MOUSE_SOURCE_DIRECTORY}"
    status --porcelain=v1 --untracked-files=normal
    RESULT_VARIABLE result OUTPUT_VARIABLE changes ERROR_VARIABLE error)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect source checkout state: ${error}")
  endif()
  if(NOT changes STREQUAL "" AND NOT MOUSE_IDENTITY_HASH MATCHES "-dirty$")
    string(APPEND MOUSE_IDENTITY_HASH "-dirty")
  endif()
else()
  if(MOUSE_IDENTITY_VERSION STREQUAL "")
    set(MOUSE_IDENTITY_VERSION "unknown")
  endif()
  if(MOUSE_IDENTITY_HASH STREQUAL "")
    set(MOUSE_IDENTITY_HASH "unknown")
  endif()
endif()

# configure_file preserves the header timestamp when its contents do not change,
# so checking identity on every build does not force unnecessary recompilation.
configure_file("${MOUSE_TEMPLATE}" "${MOUSE_OUTPUT_HEADER}" @ONLY)
