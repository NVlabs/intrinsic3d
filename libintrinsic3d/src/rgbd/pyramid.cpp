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

#include <nv/rgbd/pyramid.h>

#include <iostream>

#include <opencv2/imgproc.hpp>
#include <nv/rgbd/processing.h>


namespace nv
{

    Pyramid::Pyramid()
	{
	}


    Pyramid::Pyramid(int num_levels, const cv::Mat &color, const cv::Mat &depth)
	{
        create(num_levels, color, depth);
	}


    Pyramid::~Pyramid()
	{
	}


    bool Pyramid::create(int num_levels, const cv::Mat &color, const cv::Mat &depth)
	{
        if (num_levels <= 0 || color.empty() || depth.empty())
			return false;

		// create color image pyramid
        color_pyramid_ = createPyramid(num_levels, color);

		// convert color image to intensity (float)
        cv::Mat color_f;
        color.convertTo(color_f, CV_32FC3, 1.0 / 255.0);
		cv::Mat lum;
        cv::cvtColor(color_f, lum, CV_BGR2GRAY);
		// create intensity image pyramid
        intensity_pyramid_ = createPyramid(num_levels, lum);

		// create depth pyramid
        depth_pyramid_ = createDepthPyramid(num_levels, depth);

		return true;
	}
	
    cv::Mat Pyramid::color(int lvl)
	{
		if (color_pyramid_.empty() || lvl < 0 || lvl >= (int)color_pyramid_.size())
			return cv::Mat();
		else
			return color_pyramid_[lvl];
	}
	
	
    cv::Mat Pyramid::intensity(int lvl)
	{
		if (intensity_pyramid_.empty() || lvl < 0 || lvl >= (int)intensity_pyramid_.size())
			return cv::Mat();
		else
			return intensity_pyramid_[lvl];
	}


    cv::Mat Pyramid::depth(int lvl)
	{
		if (depth_pyramid_.empty() || lvl < 0 || lvl >= (int)depth_pyramid_.size())
			return cv::Mat();
		else
			return depth_pyramid_[lvl];
	}


    cv::Mat Pyramid::downsample(const cv::Mat &img)
	{
        cv::Mat img_down;
        cv::pyrDown(img, img_down, cv::Size(img.cols / 2, img.rows / 2));
        return img_down;
	}


    cv::Mat Pyramid::downsampleDepth(const cv::Mat &depth)
	{
		// downscaling by averaging the depth
		int w = depth.cols / 2;
		int h = depth.rows / 2;
        cv::Mat depth_down = cv::Mat::zeros(h, w, depth.type());
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				int cnt = 0;
				float sum = 0.0f;
				float d0 = depth.at<float>(2 * y, 2 * x);
				if (d0 > 0.0f) { sum += d0; ++cnt; }
				float d1 = depth.at<float>(2 * y, 2 * x + 1);
				if (d1 > 0.0f) { sum += d1; ++cnt; }
				float d2 = depth.at<float>(2 * y + 1, 2 * x);
				if (d2 > 0.0f) { sum += d2; ++cnt; }
				float d3 = depth.at<float>(2 * y + 1, 2 * x + 1);
				if (d3 > 0.0f) { sum += d3; ++cnt; }
				if (cnt > 0)
                    depth_down.at<float>(y, x) = sum / float(cnt);
			}
		}
        return depth_down;
	}


    std::vector<cv::Mat> Pyramid::createPyramid(int num_pyramid_levels, const cv::Mat &img)
	{
        std::vector<cv::Mat> pyramid;
        pyramid.push_back(img);
		// downsample images
        for (int i = 1; i < num_pyramid_levels; ++i)
            pyramid.push_back(downsample(pyramid[i - 1]));
        return pyramid;
	}


    std::vector<cv::Mat> Pyramid::createDepthPyramid(int num_pyramid_levels, const cv::Mat &depth)
	{
        std::vector<cv::Mat> pyramid;
        pyramid.push_back(depth);
        for (int i = 1; i < num_pyramid_levels; ++i)
		{
			// downsample depth images
            cv::Mat depth_down = downsampleDepth(pyramid[i - 1]);
            pyramid.push_back(depth_down);
		}
        return pyramid;
	}

} // namespace nv
