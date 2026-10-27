# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT DEFINED MOUSE_LSAN_PROBE)
  message(FATAL_ERROR "MOUSE_LSAN_PROBE is required")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=ASAN_OPTIONS --unset=LSAN_OPTIONS
  "${MOUSE_LSAN_PROBE}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error
  TIMEOUT 20)
if(status EQUAL 0 OR NOT output STREQUAL "LSAN_PROBE_COMPLETED\n" OR
   NOT error MATCHES "LeakSanitizer: detected memory leaks" OR
   NOT error MATCHES "256 byte")
  message(FATAL_ERROR "Leak detection must remain active with a resident ORT image (${status}): ${output}${error}")
endif()
message(STATUS "LSan detects the intentional 256-byte leak after validated ORT owner release")
