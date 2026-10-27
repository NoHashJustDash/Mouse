# SPDX-License-Identifier: GPL-3.0-or-later
include_guard(GLOBAL)
set(MOUSE_ORT_VERSION "1.22.0")
if(DEFINED ENV{MOUSE_ONNXRUNTIME_ROOT} AND NOT "$ENV{MOUSE_ONNXRUNTIME_ROOT}" STREQUAL "")
  set(MOUSE_ORT_ROOT "$ENV{MOUSE_ONNXRUNTIME_ROOT}" CACHE PATH "Pinned ONNX Runtime SDK")
elseif(WIN32)
  set(MOUSE_ORT_ROOT "${PROJECT_SOURCE_DIR}/.toolchain/sdks/onnxruntime/onnxruntime-win-x64-gpu-${MOUSE_ORT_VERSION}"
    CACHE PATH "Pinned ONNX Runtime SDK")
else()
  set(MOUSE_ORT_ROOT "${PROJECT_SOURCE_DIR}/.toolchain/sdks/onnxruntime/onnxruntime-linux-x64-${MOUSE_ORT_VERSION}"
    CACHE PATH "Pinned ONNX Runtime SDK")
endif()
cmake_path(ABSOLUTE_PATH MOUSE_ORT_ROOT NORMALIZE)
set(MOUSE_ORT_INCLUDE_DIR "${MOUSE_ORT_ROOT}/include")
if(NOT EXISTS "${MOUSE_ORT_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "Pinned ORT SDK not found at '${MOUSE_ORT_ROOT}'. Run scripts/build.ps1 or scripts/bootstrap-linux.sh.")
endif()
file(STRINGS "${MOUSE_ORT_ROOT}/VERSION_NUMBER" sdk_version LIMIT_COUNT 1)
string(STRIP "${sdk_version}" sdk_version)
if(NOT sdk_version STREQUAL MOUSE_ORT_VERSION)
  message(FATAL_ERROR "ORT SDK version '${sdk_version}' differs from required ${MOUSE_ORT_VERSION}")
endif()
if(WIN32)
  set(MOUSE_ORT_RUNTIME_FILENAME "onnxruntime.dll")
  set(ort_provider_prefix "")
  set(ort_provider_suffix ".dll")
else()
  set(MOUSE_ORT_RUNTIME_FILENAME "libonnxruntime.so.${MOUSE_ORT_VERSION}")
  set(ort_provider_prefix "lib")
  set(ort_provider_suffix ".so")
endif()
set(MOUSE_ORT_RUNTIME_FILES "${MOUSE_ORT_ROOT}/lib/${MOUSE_ORT_RUNTIME_FILENAME}")
# CPU SDKs need no GPU providers. Preserve every provider supplied by a GPU SDK
# for both build-tree staging and installation; backend selection remains explicit.
set(ort_has_gpu_provider FALSE)
foreach(provider IN ITEMS shared cuda tensorrt)
  set(provider_path "${MOUSE_ORT_ROOT}/lib/${ort_provider_prefix}onnxruntime_providers_${provider}${ort_provider_suffix}")
  if(EXISTS "${provider_path}")
    list(APPEND MOUSE_ORT_RUNTIME_FILES "${provider_path}")
    if(NOT provider STREQUAL "shared")
      set(ort_has_gpu_provider TRUE)
    endif()
  endif()
endforeach()
foreach(runtime IN LISTS MOUSE_ORT_RUNTIME_FILES)
  if(NOT EXISTS "${runtime}" OR IS_DIRECTORY "${runtime}")
    message(FATAL_ERROR "Required ONNX Runtime library is missing: ${runtime}")
  endif()
endforeach()
set(shared_provider "${MOUSE_ORT_ROOT}/lib/${ort_provider_prefix}onnxruntime_providers_shared${ort_provider_suffix}")
if(ort_has_gpu_provider AND NOT EXISTS "${shared_provider}")
  message(FATAL_ERROR "Required ONNX Runtime shared provider is missing: ${shared_provider}")
endif()

# Serialize staging even when several consumers share a directory. Target file
# paths also support per-target and multi-configuration output directories.
# TARGET_FILE_DIR does not add a dependency under CMP0112 NEW (CMake>=3.28),
# so consumers can depend on this always-run repair target without a cycle.
function(mouse_copy_ort_runtime target)
  if(NOT TARGET mouse_ort_runtime)
    add_custom_target(mouse_ort_runtime ALL)
  endif()
  get_target_property(consumer_type ${target} TYPE)
  if(consumer_type STREQUAL "UTILITY")
    # Utility targets have no target file; use their configured global output
    # directory. Packaging checks can then run without a target-platform compiler.
    if(CMAKE_RUNTIME_OUTPUT_DIRECTORY)
      set(destination "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
    else()
      set(destination "${PROJECT_BINARY_DIR}")
    endif()
  else()
    set(destination "$<TARGET_FILE_DIR:${target}>")
  endif()
  add_custom_command(TARGET mouse_ort_runtime POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${destination}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different ${MOUSE_ORT_RUNTIME_FILES} "${destination}"
    COMMENT "Staging pinned ONNX Runtime libraries beside ${target}"
    COMMAND_EXPAND_LISTS VERBATIM)
  add_dependencies(${target} mouse_ort_runtime)
endfunction()
