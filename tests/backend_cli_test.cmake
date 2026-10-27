# SPDX-License-Identifier: GPL-3.0-or-later
if(NOT DEFINED MOUSE_PROGRAM OR NOT DEFINED MOUSE_FIXTURE_DIRECTORY OR
   NOT DEFINED MOUSE_INPUTS OR NOT DEFINED MOUSE_TEST_DIRECTORY)
  message(FATAL_ERROR "Backend CLI test requires engine, fixtures, inputs and output directory")
endif()
file(MAKE_DIRECTORY "${MOUSE_TEST_DIRECTORY}")
set(result_file "${MOUSE_TEST_DIRECTORY}/backend-result.bin")
execute_process(COMMAND "${MOUSE_PROGRAM}" backend-eval
  --weights "${MOUSE_FIXTURE_DIRECTORY}/valid.onnx" --backend cpu --precision fp32
  --inputs "${MOUSE_INPUTS}" --buckets 64 --nncache-mb 1
  RESULT_VARIABLE status OUTPUT_FILE "${result_file}" ERROR_VARIABLE error TIMEOUT 60)
if(NOT status EQUAL 0)
  message(FATAL_ERROR "CPU backend-eval failed: ${status}: ${error}")
endif()
file(SIZE "${result_file}" size)
if(NOT size EQUAL 372544)
  message(FATAL_ERROR "Backend binary output length mismatch: ${size}")
endif()
file(READ "${result_file}" magic LIMIT 8 HEX)
file(READ "${result_file}" model_hash OFFSET 16 LIMIT 64 HEX)
file(READ "${result_file}" input_hash OFFSET 80 LIMIT 64 HEX)
file(SHA256 "${MOUSE_FIXTURE_DIRECTORY}/valid.onnx" expected_model)
file(SHA256 "${MOUSE_INPUTS}" expected_inputs)
# Read transport bytes as hex: CMake's text read may append/normalize newlines
# on Linux. Identity checks must compare exact bytes, without trimming them.
string(HEX "${expected_model}" expected_model_hex)
string(HEX "${expected_inputs}" expected_inputs_hex)
if(NOT magic STREQUAL "4d4f5553454f5531" OR NOT model_hash STREQUAL expected_model_hex OR
   NOT input_hash STREQUAL expected_inputs_hex)
  message(FATAL_ERROR "Backend output identities are not bound to actual artifacts")
endif()

function(check_error pattern)
  execute_process(COMMAND "${MOUSE_PROGRAM}" backend-eval ${ARGN}
    RESULT_VARIABLE error_status OUTPUT_FILE "${result_file}" ERROR_VARIABLE message TIMEOUT 60)
  file(SIZE "${result_file}" error_size)
  if(error_status EQUAL 0 OR NOT error_size EQUAL 0 OR NOT message MATCHES "${pattern}")
    message(FATAL_ERROR "Expected backend error /${pattern}/ with no stdout, got ${error_status}/${error_size}: ${message}")
  endif()
endfunction()
check_error("--weights")
check_error("--backend" --weights model --backend auto --precision fp32 --inputs "${MOUSE_INPUTS}")
check_error("MOUSEIN1" --weights "${MOUSE_FIXTURE_DIRECTORY}/valid.onnx" --backend cpu --precision fp32 --inputs "${MOUSE_FIXTURE_DIRECTORY}/valid.onnx")
check_error("open|exist" --weights "${MOUSE_FIXTURE_DIRECTORY}/valid.onnx" --backend cpu --precision fp32 --inputs "${MOUSE_TEST_DIRECTORY}/missing.bin")
check_error("finite" --weights "${MOUSE_FIXTURE_DIRECTORY}/nonfinite.onnx" --backend cpu --precision fp32 --inputs "${MOUSE_FIXTURE_DIRECTORY}/failure-after-valid.bin" --buckets 1)
execute_process(COMMAND "${MOUSE_PROGRAM}" backendbench
  --weights "${MOUSE_FIXTURE_DIRECTORY}/valid.onnx" --backend cpu --precision fp32
  --start-batch-size 1 --max-batch-size 3 --batch-step 1 --batches 2
  RESULT_VARIABLE bench_status OUTPUT_VARIABLE bench_output ERROR_VARIABLE bench_error TIMEOUT 60)
if(NOT bench_status EQUAL 0 OR NOT bench_output MATCHES "batch 1 physical_batch 1" OR
   NOT bench_output MATCHES "batch 2 physical_batch 2" OR
   NOT bench_output MATCHES "batch 3 physical_batch 3" OR NOT bench_output MATCHES "best_nps" OR NOT bench_output MATCHES "benchmark_config version" OR NOT bench_output MATCHES "device_identity name" OR NOT bench_output MATCHES "workspace_bytes")
  message(FATAL_ERROR "CPU backendbench coverage failed: ${bench_status}: ${bench_output}${bench_error}")
endif()


execute_process(COMMAND "${MOUSE_PROGRAM}" backend-eval --weights valid.onnx
  --backend cpu --precision fp32 --inputs "${MOUSE_INPUTS}" --buckets 64
  WORKING_DIRECTORY "${MOUSE_FIXTURE_DIRECTORY}"
  RESULT_VARIABLE relative_status OUTPUT_FILE "${result_file}" ERROR_VARIABLE relative_error TIMEOUT 60)
file(SIZE "${result_file}" relative_size)
if(NOT relative_status EQUAL 0 OR NOT relative_size EQUAL 372544)
  message(FATAL_ERROR "Relative backend weights path failed: ${relative_status}: ${relative_error}")
endif()
file(READ "${result_file}" relative_hash OFFSET 16 LIMIT 64 HEX)
if(NOT relative_hash STREQUAL expected_model_hex)
  message(FATAL_ERROR "Relative backend path resolved to the wrong model")
endif()

message(STATUS "Backend evaluation/benchmark CLI and relative path conformance passed")
