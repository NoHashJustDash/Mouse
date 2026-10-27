# SPDX-License-Identifier: GPL-3.0-or-later
cmake_minimum_required(VERSION 3.28)
foreach(required IN ITEMS MOUSE_PROGRAM MOUSE_PYTHON_EXECUTABLE MOUSE_SOURCE_DIRECTORY MOUSE_TEST_DIRECTORY)
  if(NOT DEFINED ${required})
    message(FATAL_ERROR "${required} is required")
  endif()
endforeach()
# Every CTest rerun retains a distinct immutable report, including failures.
string(RANDOM LENGTH 24 ALPHABET 0123456789abcdef unique)
execute_process(COMMAND "${MOUSE_PYTHON_EXECUTABLE}" -I
  "${MOUSE_SOURCE_DIRECTORY}/tools/src/mouse_tools/uci_gate.py"
  --engine "${MOUSE_PROGRAM}"
  --weights "${MOUSE_SOURCE_DIRECTORY}/tests/data/backend/valid.onnx"
  --failure-weights "${MOUSE_SOURCE_DIRECTORY}/tests/data/backend/search-nonfinite.onnx"
  --spec-dir "${MOUSE_SOURCE_DIRECTORY}/tests/data/uci/protocol-v1"
  --output "${MOUSE_TEST_DIRECTORY}/${unique}.json"
  RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 1400)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "UCI protocol gate failed ${result}: ${output}
${error}")
endif()
message(STATUS "${output}")
