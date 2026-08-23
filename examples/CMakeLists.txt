cmake_minimum_required(VERSION 3.5)

# Standalone example programs; see README.md in this directory.
file(GLOB EXAMPLE_SRCS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.c")
foreach(EX_SRC IN LISTS EXAMPLE_SRCS)
    get_filename_component(EX_BIN "${EX_SRC}" NAME_WE)
    add_executable("${EX_BIN}" "${EX_SRC}")
    set_target_properties("${EX_BIN}" PROPERTIES C_STANDARD 99)
    if(BUILD_STATIC)
        target_link_libraries("${EX_BIN}" HiParTI_s)
    else()
        target_link_libraries("${EX_BIN}" HiParTI)
    endif()
endforeach()
