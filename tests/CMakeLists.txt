# Unit / integration tests, registered with CTest.
#
# Run everything:            ctest --test-dir <build>
# Only the fast unit tests:  ctest -L unit
# Verbose on failure:        ctest --output-on-failure

set(HIPARTI_TEST_DATA "${CMAKE_CURRENT_SOURCE_DIR}/../data")

set(PTI_TESTS
    test_io            unit
    test_sort_convert  unit
    test_mttkrp        integration
    test_equivalence   integration
    test_edge_cases    unit
    test_ttm_ops       unit
    test_spmatrix      unit
    test_cpd           integration
    test_hicoo_variants integration
    test_tensor_algebra unit
)

list(LENGTH PTI_TESTS _n)
math(EXPR _last "${_n} / 2 - 1")
foreach(i RANGE ${_last})
    math(EXPR _ni "${i} * 2")
    math(EXPR _li "${i} * 2 + 1")
    list(GET PTI_TESTS ${_ni} _name)
    list(GET PTI_TESTS ${_li} _label)

    add_executable(${_name} "${_name}.c")
    set_target_properties(${_name} PROPERTIES C_STANDARD 99)
    if(BUILD_STATIC)
        target_link_libraries(${_name} HiParTI_s)
    else()
        target_link_libraries(${_name} HiParTI)
    endif()

    add_test(NAME ${_name} COMMAND ${_name} "${HIPARTI_TEST_DATA}")
    set_tests_properties(${_name} PROPERTIES
        LABELS "${_label}"
        TIMEOUT 300
        # each test writes scratch files into its own working directory
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()

# GPU tests: built only with USE_CUDA, labelled so they can be selected or
# excluded (`ctest -L gpu` / `ctest -LE gpu`).  Exit code 77 = skipped (no device).
if(USE_CUDA)
    cuda_add_executable(test_gpu "test_gpu.cu")
    if(BUILD_STATIC)
        target_link_libraries(test_gpu HiParTI_s)
    else()
        target_link_libraries(test_gpu HiParTI)
    endif()
    add_test(NAME test_gpu COMMAND test_gpu "${HIPARTI_TEST_DATA}")
    set_tests_properties(test_gpu PROPERTIES
        LABELS "gpu;integration"
        TIMEOUT 600
        SKIP_RETURN_CODE 77
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()
