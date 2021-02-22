# CMake 最低版本号要求
cmake_minimum_required(VERSION 2.4.3)

INCLUDE_DIRECTORIES(
        ${HRPD_SOURCE_DIR}
)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

SET(PROJUTILS_SRCS
        ProjDynMeshSampler.cpp
        ProjDynTetGen.cpp
        tetgen.cxx
        STVD.cpp
        ProjDynUtil.cpp
        StopWatch.cpp
)

SET(PROJUTILS_HRDS
        ProjDynMeshSampler.h
        ProjDynTetGen.h
        tetgen.h
        STVD.h
        ProjDynUtil.h
        StopWatch.h
)

add_library(projutils ${PROJUTILS_HRDS} ${PROJUTILS_SRCS})
