cmake_minimum_required(VERSION 3.4...3.18)
project(pyMOR)

INCLUDE_DIRECTORIES(
    ${pyMOR_SOURCE_DIR}
    ${pyMOR_SOURCE_DIR}/HRPD
    ${pyMOR_SOURCE_DIR}/external
    ${pyMOR_SOURCE_DIR}/python
)

# hrpd dir
set(MOR_SOURCE_DIR 
    ${pyMOR_SOURCE_DIR}/MOR 
    ${pyMOR_SOURCE_DIR}/HRPD 
    )

add_subdirectory(pybind11)

subdirs(HRPD)
subdirs(python)