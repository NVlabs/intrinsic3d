/**
* This file is part of Intrinsic3D.
*
* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
* Copyright (c) 2019, Technical University of Munich. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*    * Redistributions of source code must retain the above copyright
*      notice, this list of conditions and the following disclaimer.
*    * Redistributions in binary form must reproduce the above copyright
*      notice, this list of conditions and the following disclaimer in the
*      documentation and/or other materials provided with the distribution.
*    * Neither the name of NVIDIA CORPORATION nor the names of its
*      contributors may be used to endorse or promote products derived
*      from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS "AS IS" AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
* PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
* CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
* EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
* PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
* PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
* OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <nv/mat.h>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <nv/camera.h>


/**
 * @brief   Functionalities for processing the color images and depth maps
 *          contained in RGB-D frames
 * @author  Robert Maier <robert.maier@tum.de>
 */
namespace nv
{
    void threshold(cv::Mat &depth, float depth_min, float depth_max);

	cv::Mat computeVertexMap(const Mat3f &K, const cv::Mat &depth);

    cv::Mat computeNormals(const cv::Mat &vertex_map, float depth_threshold = 0.3f);
    cv::Mat computeNormals(const Mat3f &K, const cv::Mat &depth, float depth_threshold = 0.3f);

    cv::Mat resizeDepth(const Camera &input_cam, const cv::Mat &input_depth, const Camera &output_cam);

    cv::Mat erodeDiscontinuities(const cv::Mat &depth, int window_size, float max_depth_diff = 0.5f);

	template<typename T>
	T interpolate(const cv::Mat &img, float x, float y, int channel = 0);

	Vec3b interpolateRGB(const cv::Mat &color, float x, float y);

} // namespace nv
