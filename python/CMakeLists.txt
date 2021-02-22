cmake_minimum_required(VERSION 3.4...3.18)

INCLUDE_DIRECTORIES(
    ${pyMOR_SOURCE_DIR}/DRPD/core
    ${pyMOR_SOURCE_DIR}/DRPD/ProjUtils
)

SET(PYTHON_SRCS
    main.cpp
    pyMOR.cpp
)

SET(PYTHON_HDRS
    pyMOR.h
)

link_libraries(hrpd projutils)

pybind11_add_module(pymor ${PYTHON_SRCS} ${PYTHON_HDRS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
set(EXAMPLE_VERSION_INFO "0.0.1")
target_compile_definitions(pymor PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
