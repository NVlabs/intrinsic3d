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
#include <vector>

#include <opencv2/core.hpp>


namespace nv
{

    /**
     * @brief   RGB-D frame pyramid container
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class Pyramid
	{
	public:
        Pyramid();
        Pyramid(int num_levels, const cv::Mat &color, const cv::Mat &depth);
        ~Pyramid();

        bool create(int num_levels, const cv::Mat &color, const cv::Mat &depth);

		cv::Mat color(int lvl = 0);
		cv::Mat intensity(int lvl = 0);
		cv::Mat depth(int lvl = 0);

	private:
		cv::Mat downsample(const cv::Mat &img);
		cv::Mat downsampleDepth(const cv::Mat &depth);
        std::vector<cv::Mat> createPyramid(int num_pyramid_levels, const cv::Mat &img);
        std::vector<cv::Mat> createDepthPyramid(int num_pyramid_levels, const cv::Mat &depth);

        std::vector<cv::Mat> color_pyramid_;
        std::vector<cv::Mat> intensity_pyramid_;
        std::vector<cv::Mat> depth_pyramid_;
	};

} // namespace nv
