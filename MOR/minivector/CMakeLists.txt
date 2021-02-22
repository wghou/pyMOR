# CMakeList.txt : CMake project for SrSoftBody, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

SET(MiniVector_SRCS
    eig3.cpp
    mat3d.cpp
    vec2d.cpp 
    vec3.cpp 
    vec3d.cpp 
    vec3i.cpp 
    vec4i.cpp
)

SET(MiniVector_HDRS
    eig3.h
    mat3d.h
    minivector.h
    vec2d.h 
    vec2i.h 
    vec3.h 
    vec3d.h 
    vec3i.h 
    vec4d.h 
    vec4i.h
)

# Add source to this project's library
ADD_LIBRARY(minivector  ${MiniVector_SRCS} ${MiniVector_HDRS})

# TODO: Add tests and install targets if needed.
