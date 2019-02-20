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

#include <nv/rgbd/processing.h>

#include <iostream>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>

namespace nv
{

    void threshold(cv::Mat &depth, float depth_min, float depth_max)
	{
        cv::threshold(depth, depth, static_cast<double>(depth_min), 0.0, cv::THRESH_TOZERO);
        cv::threshold(depth, depth, static_cast<double>(depth_max), 0.0, cv::THRESH_TOZERO_INV);
	}


	cv::Mat computeVertexMap(const Mat3f &K, const cv::Mat &depth)
	{
		if (depth.empty() || depth.type() != CV_32FC1)
			return cv::Mat();

		float cx = K(0, 2);
		float cy = K(1, 2);
		float fx_inv = 1.0f / K(0, 0);
		float fy_inv = 1.0f / K(1, 1);

        cv::Mat vertex_map = cv::Mat::zeros(depth.size(), CV_32FC3);
		for (int y = 0; y < depth.rows; ++y)
		{
			for (int x = 0; x < depth.cols; ++x)
			{
				float d = depth.at<float>(y, x);
				float x0 = (float(x) - cx) * fx_inv;
				float y0 = (float(y) - cy) * fy_inv;
                vertex_map.at<cv::Vec3f>(y, x) = cv::Vec3f(x0 * d, y0 * d, d);
			}
		}
        return vertex_map;
	}


    cv::Mat computeNormals(const cv::Mat &vertex_map, float depth_threshold)
	{
        if (vertex_map.empty() || vertex_map.type() != CV_32FC3)
			return cv::Mat();

        int w = vertex_map.cols;
        int h = vertex_map.rows;
		cv::Mat normals = cv::Mat::zeros(h, w, CV_32FC3);
        const float* ptr_vert = reinterpret_cast<const float*>(vertex_map.data);

		// depth threshold to avoid computing properties over depth-discontinuities
        for (int y = 1; y < h - 1; ++y)
		{
            for (int x = 1; x < w - 1; ++x)
			{
                size_t off = static_cast<size_t>((y*w + x) * 3);
                Vec3f vert(ptr_vert[off], ptr_vert[off + 1], ptr_vert[off + 2]);
                if (vert[2] == 0.0f)
					continue;

				// determine tangent vectors
                size_t off_x0 = static_cast<size_t>((y*w + x - 1) * 3);
                Vec3f vert_x0(ptr_vert[off_x0], ptr_vert[off_x0 + 1], ptr_vert[off_x0 + 2]);
                size_t off_x1 = static_cast<size_t>((y*w + x + 1) * 3);
                Vec3f vert_x1(ptr_vert[off_x1], ptr_vert[off_x1 + 1], ptr_vert[off_x1 + 2]);
                size_t off_y0 = static_cast<size_t>(((y - 1)*w + x) * 3);
                Vec3f vert_y0(ptr_vert[off_y0], ptr_vert[off_y0 + 1], ptr_vert[off_y0 + 2]);
                size_t off_y1 = static_cast<size_t>(((y + 1)*w + x) * 3);
                Vec3f vert_y1(ptr_vert[off_y1], ptr_vert[off_y1 + 1], ptr_vert[off_y1 + 2]);
                if (vert_x0[2] == 0.0f || vert_x1[2] == 0.0f || vert_y0[2] == 0.0f || vert_y1[2] == 0.0f)
					continue;

                Vec3f tangent_x = vert_x1 - vert_x0;
                Vec3f tangent_y = vert_y1 - vert_y0;
                if (tangent_x.norm() < depth_threshold && tangent_y.norm() < depth_threshold)
				{
					// compute normal using cross product
                    Vec3f n = (tangent_y.cross(tangent_x)).normalized();
					normals.at<cv::Vec3f>(y, x) = cv::Vec3f(n[0], n[1], n[2]);
				}
			}
		}

		return normals;
	}


    cv::Mat computeNormals(const Mat3f &K, const cv::Mat &depth, float depth_threshold)
	{
        cv::Mat vertex_map = computeVertexMap(K, depth);
        cv::Mat normals = computeNormals(vertex_map, depth_threshold);
		return normals;
	}


    cv::Mat resizeDepth(const Camera &input_cam, const cv::Mat &input_depth, const Camera &output_cam)
	{
        if (input_depth.empty() || input_depth.type() != CV_32FC1)
			return cv::Mat();
        int w = output_cam.width();
        int h = output_cam.height();
        if (input_depth.cols == w && input_depth.rows == h)
		{
			// TODO check if intrinsics of both cameras are the same
            return input_depth.clone();
		}

        const Mat3f input_K = input_cam.intrinsics();
        const Mat3f output_K = output_cam.intrinsics();

        float in_cx = input_K(0, 2);
        float in_cy = input_K(1, 2);
        float in_fx = input_K(0, 0);
        float in_fy = input_K(1, 1);

        float out_cx = output_K(0, 2);
        float out_cy = output_K(1, 2);
        float out_fx_inv = 1.0f / output_K(0, 0);
        float out_fy_inv = 1.0f / output_K(1, 1);

        cv::Mat depth_out = cv::Mat::zeros(h, w, CV_32FC1);
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				// compute lookup coordinates for input depth
                float x0 = (float(x) - out_cx) * out_fx_inv;
                float y0 = (float(y) - out_cy) * out_fy_inv;
				// 3d point (depth and color are registered/aligned -> depth value shouldn't matter)
				Vec3f p(x0, y0, 1.0f);
				// project 3d point into depth image
				Vec2f p2;
                p2[0] = (in_fx * p[0] / p[2]) + in_cx;
                p2[1] = (in_fy * p[1] / p[2]) + in_cy;
				Vec2i p2i = (p2 + Vec2f::Constant(0.5f)).cast<int>();
                if (p2i[0] < 0 || p2i[1] < 0 || p2i[0] >= input_depth.cols || p2i[1] >= input_depth.rows)
					continue;
                // lookup depth in input depth (using linear interpolation)
                float d_in = interpolate<float>(input_depth, p2[0], p2[1], 0);
				// check depth and store it
                if (d_in == 0.0f)
					continue;
                depth_out.at<float>(y, x) = d_in;
			}
		}

        return depth_out;
	}


    cv::Mat erodeDiscontinuities(const cv::Mat &depth_in, int window_size, float max_depth_diff)
	{
        if (depth_in.empty() || depth_in.type() != CV_32FC1)
			return cv::Mat();

        if (window_size <= 0)
		{
			// no erosion -> return copy of input depth
            return depth_in.clone();
		}

		// valid depth values
        int h = depth_in.rows;
        int w = depth_in.cols;
        const float* ptr_in = reinterpret_cast<const float*>(depth_in.data);
        cv::Mat depth_out = depth_in.clone();
        float* ptr_out = reinterpret_cast<float*>(depth_out.data);

		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
                size_t idx = static_cast<size_t>(y*w + x);
                float d_ref = ptr_in[idx];
                if (d_ref == 0.0f)
				{
                    ptr_out[idx] = 0.0f;
					continue;
				}

				bool valid = true;
                for (int v = std::max(0, y - window_size); v <= std::min(y + window_size, h - 1); ++v)
				{
                    for (int u = std::max(0, x - window_size); u <= std::min(x + window_size, w - 1); ++u)
					{
                        size_t off = static_cast<size_t>(v*w + u);
                        float d = ptr_in[off];
                        if (d == 0.0f || std::abs(d - d_ref) > max_depth_diff)
						{
							valid = false;
							break;
						}
					}
					if (!valid)
						break;
				}
				if (!valid)
                    ptr_out[idx] = 0.0f;
			}
		}
        return depth_out;
	}


	template<typename T>
	T interpolate(const cv::Mat &img, float x, float y, int channel)
	{
		int w = img.cols;
		int h = img.rows;
        T val_cur = static_cast<T>(0.0);
        const T* ptr_img = reinterpret_cast<const T*>(img.data);
		int nc = img.channels();

		//bilinear interpolation
		int x0 = static_cast<int>(std::floor(x));
		int y0 = static_cast<int>(std::floor(y));
		int x1 = x0 + 1;
		int y1 = y0 + 1;

        float x1_weight = x - static_cast<float>(x0);
        float y1_weight = y - static_cast<float>(y0);
        float x0_weight = 1.0f - x1_weight;
        float y0_weight = 1.0f - y1_weight;

		if (x0 < 0 || x0 >= w)
            x0_weight = 0.0f;
		if (x1 < 0 || x1 >= w)
            x1_weight = 0.0f;
		if (y0 < 0 || y0 >= h)
            y0_weight = 0.0f;
		if (y1 < 0 || y1 >= h)
            y1_weight = 0.0f;
        float w00 = x0_weight * y0_weight;
        float w10 = x1_weight * y0_weight;
        float w01 = x0_weight * y1_weight;
        float w11 = x1_weight * y1_weight;

        float sumWeights = w00 + w10 + w01 + w11;
        float sum = 0.0f;
        if (w00 > 0.0f)
            sum += static_cast<float>(ptr_img[(y0*w + x0) * nc + channel]) * w00;
        if (w01 > 0.0f)
            sum += static_cast<float>(ptr_img[(y1*w + x0) * nc + channel]) * w01;
        if (w10 > 0.0f)
            sum += static_cast<float>(ptr_img[(y0*w + x1) * nc + channel]) * w10;
        if (w11 > 0.0f)
            sum += static_cast<float>(ptr_img[(y1*w + x1) * nc + channel]) * w11;

        if (sumWeights > 0.0f)
            val_cur = static_cast<T>(sum / sumWeights);

        return val_cur;
	}
	template unsigned char interpolate(const cv::Mat &img, float x, float y, int channel);
	template unsigned short interpolate(const cv::Mat &img, float x, float y, int channel);
	template int interpolate(const cv::Mat &img, float x, float y, int channel);
	template float interpolate(const cv::Mat &img, float x, float y, int channel);
	template double interpolate(const cv::Mat &img, float x, float y, int channel);


	Vec3b interpolateRGB(const cv::Mat &color, float x, float y)
	{
		// lookup color using bilinear interpolation
		unsigned char r = interpolate<unsigned char>(color, x, y, 2);
		unsigned char g = interpolate<unsigned char>(color, x, y, 1);
		unsigned char b = interpolate<unsigned char>(color, x, y, 0);
		return Vec3b(r, g, b);
	}

} // namespace nv
