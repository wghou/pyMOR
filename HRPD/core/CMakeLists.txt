# CMake 最低版本号要求
cmake_minimum_required(VERSION 2.4.3)

INCLUDE_DIRECTORIES(
        ${HRPD_SOURCE_DIR}
)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

SET(HRPD_SRCS
	predicates.cxx
	ProjDynConstraints.cpp
        ProjDynRHSInterpol.cpp
        ProjDynSimulator.cpp
)

SET(HRPD_HRDS
        ProjDynConstraints.h
        ProjDynRHSInterpol.h
        ProjDynSimulator.h    
)

add_library(hrpd ${HRPD_HRDS} ${HRPD_SRCS})
