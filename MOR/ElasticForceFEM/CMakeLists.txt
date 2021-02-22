# CMakeList.txt : CMake project for SrSoftBody, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

SET(ElasticForceFEM_SRCS
    elasticForceFEM.cpp
)

SET(ElasticForceFEM_HDRS
    elasticForceFEM.h
)

# Add source to this project's library
ADD_LIBRARY(ElasticForceFEM  ${ElasticForceFEM_SRCS} ${ElasticForceFEM_HDRS})

# TODO: Add tests and install targets if needed.
