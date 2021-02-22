cmake_minimum_required(VERSION 3.4...3.18)

INCLUDE_DIRECTORIES(
        #${pyVegaFEM_SOURCE_DIR}/VegaFEM/isotropicHyperelasticFEM
        ${pyVegaFEM_SOURCE_DIR}/VegaFEM/minivector
        ${pyVegaFEM_SOURCE_DIR}/VegaFEM/ElasticForceFEM
)

SET(PYTHON_SRCS
    main.cpp
    pyElasticForceFEM.cpp
)

SET(PYTHON_HDRS
    pyElasticForceFEM.h
)

link_libraries(minivector ElasticForceFEM)

pybind11_add_module(pyvegafem ${PYTHON_SRCS} ${PYTHON_HDRS})

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
set(EXAMPLE_VERSION_INFO "0.0.1")
target_compile_definitions(pyvegafem PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
