# CMake 最低版本号要求
cmake_minimum_required(VERSION 2.4.3)

INCLUDE_DIRECTORIES(
	${HRPD_SOURCE_DIR}
)

subdirs(core projutils)