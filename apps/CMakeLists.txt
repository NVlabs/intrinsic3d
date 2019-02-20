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
# CMake file for building Intrinsic3D applications.
# Author: Robert Maier (robert.maier@tum.de)
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# include directory
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)

# ------------------------------------------------------------------------
# Keyframe selection app
SET(APP_KEYFRAMES AppKeyframes)
ADD_EXECUTABLE(${APP_KEYFRAMES}
    include/nv/app_keyframes.h
    src/app_keyframes.cpp
)
TARGET_LINK_LIBRARIES(${APP_KEYFRAMES} ${Intrinsic3D_LIBRARIES})
TARGET_COMPILE_OPTIONS(${APP_KEYFRAMES} PRIVATE -std=c++11)

# ------------------------------------------------------------------------
# Fusion app
SET(APP_FUSION AppFusion)
ADD_EXECUTABLE(${APP_FUSION}
    include/nv/app_fusion.h
    src/app_fusion.cpp
)
TARGET_LINK_LIBRARIES(${APP_FUSION} ${Intrinsic3D_LIBRARIES})
TARGET_COMPILE_OPTIONS(${APP_FUSION} PRIVATE -std=c++11)

# ------------------------------------------------------------------------
# Intrinsic3D app
SET(APP_INTRINSIC3D AppIntrinsic3D)
ADD_EXECUTABLE(${APP_INTRINSIC3D}
    include/nv/app_intrinsic3d.h
    src/app_intrinsic3d.cpp
)
TARGET_LINK_LIBRARIES(${APP_INTRINSIC3D} ${Intrinsic3D_LIBRARIES})
TARGET_COMPILE_OPTIONS(${APP_INTRINSIC3D} PRIVATE -std=c++11)
