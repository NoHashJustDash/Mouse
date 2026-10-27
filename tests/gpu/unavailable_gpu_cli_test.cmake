# SPDX-License-Identifier: GPL-3.0-or-later
# Actual GPU-driver process regression. Never included in CPU acceptance.
foreach(required IN ITEMS MOUSE_PROGRAM MOUSE_MODEL MOUSE_INPUTS MOUSE_TEST_DIRECTORY)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
set(directories "$ENV{MOUSE_GPU_DLL_DIRECTORIES}")
if(directories STREQUAL "")
  message(FATAL_ERROR "Set explicit MOUSE_GPU_DLL_DIRECTORIES before GPU error tests")
endif()
set(sdk_arguments)
foreach(directory IN LISTS directories)
  if(NOT IS_ABSOLUTE "${directory}" OR NOT IS_DIRECTORY "${directory}")
    message(FATAL_ERROR "GPU SDK directory must exist and be absolute: ${directory}")
  endif()
  list(APPEND sdk_arguments --gpu-dll-dir "${directory}")
endforeach()
# Retain every invocation's output, including failures, in a new private child.
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef unique)
set(test_root "${MOUSE_TEST_DIRECTORY}/${unique}")
file(MAKE_DIRECTORY "${test_root}")
foreach(provider IN ITEMS cuda tensorrt)
  if(provider STREQUAL "cuda")
    set(label CUDA)
  else()
    set(label TensorRT)
  endif()
  foreach(scenario IN ITEMS masked ordinal)
    if(scenario STREQUAL "masked")
      set(visible -1)
      set(ordinal 0)
      set(expected_name CUDA_ERROR_NO_DEVICE)
      set(expected_status 100)
    else()
      # Exactly one visible physical device makes ordinal1023 nonexistent,
      # independent of the machine's total number of installed GPUs.
      set(visible 0)
      set(ordinal 1023)
      set(expected_name CUDA_ERROR_INVALID_DEVICE)
      set(expected_status 101)
    endif()
    set(stem "${test_root}/${provider}-${scenario}")
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PATH="
      "CUDA_VISIBLE_DEVICES=${visible}" "${MOUSE_PROGRAM}" backend-eval
      --weights "${MOUSE_MODEL}" --backend "${provider}" --precision fp32
      --inputs "${MOUSE_INPUTS}" --buckets 1 --device "${ordinal}"
      --nncache-mb 0 --cache-dir "${test_root}/c" ${sdk_arguments}
      RESULT_VARIABLE status OUTPUT_FILE "${stem}.stdout.bin"
      ERROR_FILE "${stem}.stderr.txt" TIMEOUT 60)
    file(SIZE "${stem}.stdout.bin" output_bytes)
    file(READ "${stem}.stderr.txt" error)
    string(FIND "${error}" "Backend construction [${label}]:" provider_index)
    string(FIND "${error}" "${expected_name}" name_index)
    string(FIND "${error}" "CUDA driver status ${expected_status}" status_index)
    string(FIND "${error}" "CUDA_VISIBLE_DEVICES" visibility_index)
    string(FIND "${error}" "--device" device_index)
    # Require a nonempty driver-supplied description between name and remedy;
    # its wording belongs to the installed driver, not a copied local table.
    string(REGEX MATCH "${expected_name}[)]: [^;\r\n]+; check" description "${error}")
    if(NOT "${status}" STREQUAL "1" OR NOT output_bytes EQUAL 0 OR
       provider_index LESS 0 OR name_index LESS 0 OR status_index LESS 0 OR
       visibility_index LESS 0 OR device_index LESS 0 OR description STREQUAL "")
      message(FATAL_ERROR "${provider}/${scenario}: expected named ${expected_name}, selected provider, numeric status, remedy, exit1 and zero output; got ${status}/${output_bytes}: ${error}; evidence ${stem}")
    endif()
    message(STATUS "${provider}/${scenario}: ${expected_name}, status${expected_status}, exit1, stdout0; ${error}")
  endforeach()
endforeach()
message(STATUS "All4 actual GPU-unavailable/device-selection errors verified; evidence ${test_root}")
