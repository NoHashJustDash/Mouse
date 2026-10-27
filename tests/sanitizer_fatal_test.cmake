# SPDX-License-Identifier: GPL-3.0-or-later
foreach(required IN ITEMS MOUSE_UBSAN_PROBE MOUSE_TEST_DIRECTORY)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
file(MAKE_DIRECTORY "${MOUSE_TEST_DIRECTORY}")
# Compiler policy must be fatal even when an inherited runtime setting permits
# recovery. Clear optional ASan overrides so missing instrumentation is exposed.
set(child_environment "${CMAKE_COMMAND}" -E env --unset=ASAN_OPTIONS
  "UBSAN_OPTIONS=halt_on_error=0:print_stacktrace=0")
execute_process(COMMAND ${child_environment} "${MOUSE_UBSAN_PROBE}"
  RESULT_VARIABLE control_result OUTPUT_VARIABLE control_output ERROR_VARIABLE control_error
  TIMEOUT 20)
file(WRITE "${MOUSE_TEST_DIRECTORY}/control.stdout" "${control_output}")
file(WRITE "${MOUSE_TEST_DIRECTORY}/control.stderr" "${control_error}")
if(NOT control_result EQUAL 0 OR NOT control_output STREQUAL "UBSAN_PROBE_ENTERED\nUBSAN_PROBE_COMPLETED\n"
   OR NOT control_error STREQUAL "")
  message(FATAL_ERROR "Sanitizer control did not complete cleanly (${control_result}): ${control_output}${control_error}")
endif()
execute_process(COMMAND ${child_environment} "${MOUSE_UBSAN_PROBE}" overflow
  RESULT_VARIABLE overflow_result OUTPUT_VARIABLE overflow_output ERROR_VARIABLE overflow_error
  TIMEOUT 20)
file(WRITE "${MOUSE_TEST_DIRECTORY}/overflow.stdout" "${overflow_output}")
file(WRITE "${MOUSE_TEST_DIRECTORY}/overflow.stderr" "${overflow_error}")
if(NOT overflow_result MATCHES "^-?[0-9]+$" OR overflow_result EQUAL 0)
  message(FATAL_ERROR "UBSan must exit nonzero on intentional signed overflow (${overflow_result}): ${overflow_output}${overflow_error}")
endif()
if(NOT overflow_error MATCHES "runtime error: signed integer overflow"
   OR NOT overflow_output STREQUAL "UBSAN_PROBE_ENTERED\n")
  message(FATAL_ERROR "UBSan did not diagnose and stop at the overflow: ${overflow_output}${overflow_error}")
endif()
message(STATUS "Fatal UBSan policy: clean control0; signed overflow diagnosed, nonzero, no completion sentinel")
