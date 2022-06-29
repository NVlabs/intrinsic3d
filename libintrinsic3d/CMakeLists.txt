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
# CMake file for building Intrinsic3D library.
# Author: Robert Maier (robert.maier@tum.de)
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# glob headers and source files
FILE(GLOB_RECURSE incs "${CMAKE_CURRENT_SOURCE_DIR}/include/" "*.h")
FILE(GLOB_RECURSE srcs "${CMAKE_CURRENT_SOURCE_DIR}/src/" "*.cpp")
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
# group headers and source files
SOURCE_GROUP("Header Files" FILES ${incs})
SOURCE_GROUP("Source Files" FILES ${srcs})

# ------------------------------------------------------------------------
# Intrinsic3D library
SET(Intrinsic3D_LIBRARY Intrinsic3D)
SET(Intrinsic3D_LIB_DEPENDENCIES
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${CERES_LIBRARIES}
    ${GOMP_LIBRARY}
)
LIST(APPEND Intrinsic3D_LIBRARIES ${Intrinsic3D_LIBRARY})
#LIST(APPEND Intrinsic3D_LIBRARIES ${Intrinsic3D_LIB_DEPENDENCIES})
SET(Intrinsic3D_LIBRARIES ${Intrinsic3D_LIBRARIES} PARENT_SCOPE)

ADD_LIBRARY(${Intrinsic3D_LIBRARY} STATIC
    ${incs} ${srcs}
)
TARGET_LINK_LIBRARIES(${Intrinsic3D_LIBRARY}
    ${Intrinsic3D_LIB_DEPENDENCIES}
)
SET_TARGET_PROPERTIES(${Intrinsic3D_LIBRARY} PROPERTIES
    LINKER_LANGUAGE CXX
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
TARGET_COMPILE_OPTIONS(${Intrinsic3D_LIBRARY} PRIVATE -std=c++11)
