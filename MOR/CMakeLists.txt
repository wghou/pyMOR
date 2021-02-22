# CMake 最低版本号要求
cmake_minimum_required(VERSION 2.4.3)

INCLUDE_DIRECTORIES(
        ${pyVegaFEM_SOURCE_DIR}/VegaFEM/minivector
        ${pyVegaFEM_SOURCE_DIR}/VegaFEM/ElasticForceFEM
)

subdirs(ElasticForceFEM minivector)