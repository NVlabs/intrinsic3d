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

#include <opencv2/core.hpp>


/**
 * @brief   Pinhole camera model
 * @author  Robert Maier <robert.maier@tum.de>
 */

namespace nv
{

    class Camera
    {
    public:

        Camera();
        Camera(const Mat3f &K, const int width, const int height, const Vec5f &dist_coeffs = Vec5f::Zero());
        ~Camera();

        int width() const;
        void setWidth(int w);
        int height() const;
        void setHeight(int h);

        void setIntrinsics(const Mat3f &K);
        void setIntrinsics(const Vec4 &K_vec);
        Mat3f intrinsics() const;
        Vec4 intrinsicsVec() const;

        void setDistortion(const Vec5f &dist_coeffs);
        Vec5f distortion() const;

        bool project(const Vec3f &pt, Vec2f &pt2f, Vec2i &pt2i) const;
        Vec3f project2(const Vec3f& p) const;

        Vec3f unproject(float x, float y, const cv::Mat &depth) const;
        Vec3f unproject2(int ux, int uy, float depth) const;

        bool load(const std::string &filename);
        bool save(const std::string &filename) const;

        void print() const;

    private:
        Mat3f defaultIntriniscs() const;

        Mat3 convert(const Vec4 &intrinsics) const;
        Vec4f convert(const Mat3f &intrinsics) const;

        Mat3f K_;
        int width_;
        int height_;
        Vec5f dist_coeffs_;
    };


    template <typename T>
    struct CameraT
    {

        inline bool project(const T p[3], T p2d[2]) const
        {
            // compute normalized 2D point (project 3d point onto image plane)
            T x = p[0] / p[2];
            T y = p[1] / p[2];
            // apply radial distortion to normalized point to get distorted point
            const T r2 = x * x + y * y;
            const T r4 = r2 * r2;
            const T r6 = r4 * r2;
            const T dist_coeff = T(1.0) + dist_coeffs[0] * r2 + dist_coeffs[1] * r4 + dist_coeffs[2] * r6;
            x = x * dist_coeff + T(2.0) * dist_coeffs[3] * x * y + dist_coeffs[4] * (r2 + T(2.0) * x * x);
            y = y * dist_coeff + T(2.0) * dist_coeffs[4] * x * y + dist_coeffs[3] * (r2 + T(2.0) * y * y);
            // convert point to pixel coordinates and apply center pixel offset
            p2d[0] = fx * x + cx;
            p2d[1] = fy * y + cy;
            // check if projected point lies within image bounds
            if (p2d[0] < T(0.0) || p2d[0] > T(w - 1) || p2d[1] < T(0.0) || p2d[1] > T(h - 1))
                return false;
            else
                return true;
        }


        T fx;
        T fy;
        T cx;
        T cy;
        const T* dist_coeffs;
        int w;
        int h;
    };

} // namespace nv
