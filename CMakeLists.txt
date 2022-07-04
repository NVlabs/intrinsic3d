##
# This file is part of Intrinsic3D.
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2019, Technical University of Munich. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of NVIDIA CORPORATION nor the names of its
#      contributors may be used to endorse or promote products derived
#      from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##

# ------------------------------------------------------------------------
# CMake project file for Intrinsic3D.
# Author: Robert Maier (robert.maier@tum.de)
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.9.0 FATAL_ERROR)
PROJECT(Intrinsic3D C CXX)

# ------------------------------------------------------------------------
# CMake configuration
IF(NOT CMAKE_BUILD_TYPE)
    # CMAKE_BUILD_TYPE: RelWithDebInfo, Release or Debug
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# CMake paths
# Externals directories (from environment variable "EXT")
LIST(APPEND CMAKE_PREFIX_PATH "$ENV{EXT}")
# include files in cmake subfolder
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# executable and library output paths
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# ------------------------------------------------------------------------
# dependencies

# Boost
IF(WIN32)
    SET(Boost_USE_STATIC_LIBS ON)
ENDIF()
FIND_PACKAGE(Boost COMPONENTS filesystem graph system REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

# Eigen
FIND_PACKAGE(Eigen REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

# OpenCV
FIND_PACKAGE(OpenCV 4.0.0 COMPONENTS core highgui imgproc imgcodecs REQUIRED)
IF(OPENCV_CORE_FOUND)
    INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ENDIF()

# OpenMP
FIND_PACKAGE(OpenMP)
IF(OpenMP_CXX_FOUND)
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    IF(NOT WIN32)
        SET(GOMP_LIBRARY OpenMP::OpenMP_CXX)
    ENDIF()
ENDIF()

# Ceres
FIND_PACKAGE(Ceres REQUIRED)
IF(Ceres_FOUND)
    MESSAGE(STATUS "Ceres found (include: ${CERES_INCLUDE_DIR}, libs: ${CERES_LIBRARIES})")
    INCLUDE_DIRECTORIES(${CERES_INCLUDE_DIRS})
ENDIF()

# ------------------------------------------------------------------------
# C++ flags
IF(WIN32 AND MSVC)
    # Visual Studio preprocessor defines etc
    # on Windows, use OpenCV release libs also in RelWithDebInfo mode (to avoid crashes)
    SET_TARGET_PROPERTIES(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)
    # set required preprocessor defines
    ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
    ADD_DEFINITIONS(-DNOMINMAX)
    ADD_DEFINITIONS(-D_WINDOWS)
    # activate C++ exception handling in MSVC
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
ELSE()
    # explicitly set CXX standard to fix compile issues with Ceres-2.x.x
    SET(CMAKE_CXX_STANDARD 14)
    # flags for GCC etc
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -pthread")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable -Wno-unused-parameter")
ENDIF()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

# ------------------------------------------------------------------------
# Intrinsic3D library
ADD_SUBDIRECTORY(libintrinsic3d)
INCLUDE_DIRECTORIES(libintrinsic3d/include)

# ------------------------------------------------------------------------
# applications
ADD_SUBDIRECTORY(apps)
