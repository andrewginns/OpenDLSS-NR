file(MAKE_DIRECTORY "${NR_TEST_DIR}/safe-source")
file(WRITE "${NR_TEST_DIR}/safe-source/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.24)\ninclude(\"${NR_SOURCE_DIR}/cmake/CheckArithmeticFlags.cmake\")\nnr_check_arithmetic_flags()\nproject(flag_test LANGUAGES NONE)\n")
set(case 0)
foreach(flag --use_fast_math -use_fast_math --ftz=true -ftz=true "--ftz true" "-ftz true"
             --prec-div=false -prec-div=false "--prec-div false" "-prec-div false"
             --prec-sqrt=false -prec-sqrt=false "--prec-sqrt false" "-prec-sqrt false"
             -ffast-math /fp:fast -Xcompiler=-ffast-math --compiler-options=-ffast-math
             -Xcompiler=-fPIC,-ffast-math)
  foreach(variable CMAKE_CUDA_FLAGS CMAKE_CUDA_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    math(EXPR case "${case}+1")
    execute_process(COMMAND "${CMAKE_COMMAND}" -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/bad-${case}"
      "-D${variable}=${flag}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "unsafe flags alter the compatibility arithmetic")
      message(FATAL_ERROR "Guard did not reject ${variable}=${flag}: ${output}${error}")
    endif()
  endforeach()
endforeach()
foreach(flag --ftz=false -ftz=false "--ftz false" --prec-div=true -prec-div=true
             "--prec-div true" --prec-sqrt=true -prec-sqrt=true "--prec-sqrt true"
             -lineinfo "-DNAME=--use_fast_math" "-I/path/--ftz=true" --use_fast_math_suffix
             "\"-I/path with --ftz=true\"" "\"-DNAME=--prec-div false\"")
  math(EXPR case "${case}+1")
  execute_process(COMMAND "${CMAKE_COMMAND}" -S "${NR_TEST_DIR}/safe-source" -B "${NR_TEST_DIR}/safe-${case}"
    "-DCMAKE_CUDA_FLAGS=${flag}" RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(NOT status EQUAL 0)
    message(FATAL_ERROR "Guard rejected safe/unrelated flag ${flag}: ${output}${error}")
  endif()
endforeach()
foreach(environment "CXXFLAGS=-ffast-math" "CUDAFLAGS=-use_fast_math")
  math(EXPR case "${case}+1")
  execute_process(COMMAND "${CMAKE_COMMAND}" -E env "${environment}" "${CMAKE_COMMAND}"
    -S "${NR_SOURCE_DIR}" -B "${NR_TEST_DIR}/environment-${case}"
    RESULT_VARIABLE status OUTPUT_VARIABLE output ERROR_VARIABLE error)
  if(status EQUAL 0 OR NOT "${output}${error}" MATCHES "unsafe flags alter the compatibility arithmetic")
    message(FATAL_ERROR "Guard did not reject ${environment}: ${output}${error}")
  endif()
endforeach()
message(STATUS "Compiler flag configure cases passed: ${case}")
