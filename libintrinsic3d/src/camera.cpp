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

#include <nv/camera.h>

#include <iostream>
#include <fstream>


namespace nv
{

    Camera::Camera() :
        K_(Mat3f::Identity()),
        width_(640),
        height_(480),
        dist_coeffs_(Vec5f::Zero())
    {
    }


    Camera::Camera(const Mat3f &K, const int width, const int height, const Vec5f &dist_coeffs) :
        K_(K),
        width_(width),
        height_(height),
        dist_coeffs_(dist_coeffs)
    {
    }


    Camera::~Camera()
    {
    }


    int Camera::width() const
    {
        return width_;
    }


    void Camera::setWidth(int w)
    {
        width_ = w;
    }


    int Camera::height() const
    {
        return height_;
    }


    void Camera::setHeight(int h)
    {
        height_ = h;
    }


    void Camera::setIntrinsics(const Mat3f &K)
    {
        K_ = K;
    }


    void Camera::setIntrinsics(const Vec4 &K_vec)
    {
        K_ = convert(K_vec).cast<float>();
    }


    Mat3f Camera::intrinsics() const
    {
        return K_;
    }


    Vec4 Camera::intrinsicsVec() const
    {
        return convert(K_).cast<double>();
    }


    void Camera::setDistortion(const Vec5f &dist_coeffs)
    {
        dist_coeffs_ = dist_coeffs;
    }


    Vec5f Camera::distortion() const
    {
        return dist_coeffs_;
    }


    bool Camera::project(const Vec3f &pt, Vec2f &pt2f, Vec2i &pt2i) const
    {
        // shorthands for camera intrinsics
        const float fx = K_(0, 0);
        const float fy = K_(1, 1);
        const float cx = K_(0, 2);
        const float cy = K_(1, 2);

        // compute normalized 2D point (project 3d point onto image plane)
        float x = pt[0] / pt[2];
        float y = pt[1] / pt[2];
        if (!dist_coeffs_.isZero())
        {
            // apply radial distortion to normalized point to get distorted point
            const float r2 = x * x + y * y;
            const float r4 = r2 * r2;
            const float r6 = r4 * r2;
            const float dist_coeff = 1.0f + dist_coeffs_[0] * r2 + dist_coeffs_[1] * r4 + dist_coeffs_[2] * r6;
            x = x * dist_coeff + 2.0f * dist_coeffs_[3] * x * y + dist_coeffs_[4] * (r2 + 2.0f * x * x);
            y = y * dist_coeff + 2.0f * dist_coeffs_[4] * x * y + dist_coeffs_[3] * (r2 + 2.0f * y * y);
        }
        // convert point to pixel coordinates and apply center pixel offset
        pt2f[0] = fx * x + cx;
        pt2f[1] = fy * y + cy;
        // compute integer 2D image coordinates (round to nearest integer)
        pt2i = Vec2i(static_cast<int>(pt2f[0] + 0.5f), static_cast<int>(pt2f[1] + 0.5f));
        // check if points is within image bounds
        if (pt2i[0] < 0 || pt2i[0] >= width_ || pt2i[1] < 0 || pt2i[1] >= height_)
            return false;
        return true;
    }


    Vec3f Camera::project2(const Vec3f& p) const
    {
        float x = (p[0] * K_(0, 0)) / p[2] + K_(0, 2);
        float y = (p[1] * K_(1, 1)) / p[2] + K_(1, 2);
        return Vec3f(x, y, p[2]);
    }


    Vec3f Camera::unproject(float x, float y, const cv::Mat &depth) const
    {
        assert(width_ == depth.cols && height_ == depth.rows);

        const float cx = K_(0, 2);
        const float cy = K_(1, 2);
        const float fx_inv = 1.0f / K_(0, 0);
        const float fy_inv = 1.0f / K_(1, 1);

        Vec3f pt = Vec3f::Zero();
        const int u = static_cast<int>(x + 0.5f);
        const int v = static_cast<int>(y + 0.5f);
        if (u >= 0 && u < width_ && v >= 0 && v < height_)
        {
            const float d = depth.at<float>(v, u);
            if (d != 0.0f)
            {
                pt[0] = (float(x) - cx) * fx_inv * d;
                pt[1] = (float(y) - cy) * fy_inv * d;
                pt[2] = d;
            }
        }
        return pt;
    }


    Vec3f Camera::unproject2(int ux, int uy, float depth) const
    {
        if (depth == 0.0f)
            return Vec3f::Zero();

        const float x = (static_cast<float>(ux) - K_(0, 2)) / K_(0, 0);
        const float y = (static_cast<float>(uy) - K_(1, 2)) / K_(1, 1);
        return Vec3f(depth*x, depth*y, depth);
    }


    bool Camera::load(const std::string &filename)
    {
        bool loaded = false;

        std::ifstream input_file(filename.c_str());
        if (input_file.is_open())
        {
            try
            {
                //camera width and height
                input_file >> width_ >> height_;
                //camera intrinsics
                float val = 0.0f;
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        input_file >> val;
                        K_(i, j) = val;
                    }
                }
                //distortion coefficients
                for (int i = 0; i < 5; i++) {
                    input_file >> val;
                    dist_coeffs_[i] = val;
                }
                input_file.close();
                loaded = true;
            }
            catch (...)
            {
            }
        }

        if (!loaded)
        {
            std::cout << "Intrinsics file ('" << filename << "') could not be loaded! Using defaults..." << std::endl;
            K_ = defaultIntriniscs();
            dist_coeffs_ = Vec5f::Zero();
        }

        return loaded;
    }


    bool Camera::save(const std::string &filename) const
    {
        if (filename.empty())
            return false;

        // open output file for camera intrinsics
        std::ofstream out_file;
        out_file.open(filename.c_str());
        if (!out_file.is_open())
            return false;

        // write camera intrinsics into output file

        // sensor dimensions
        out_file << width_ << " " << height_ << std::endl;

        // camera matrix K
        //outFile << std::fixed << std::setprecision(3);
        out_file << K_(0, 0) << " 0 " << K_(0, 2) << std::endl;
        out_file << "0 " << K_(1, 1) << " " << K_(1, 2) << std::endl;
        out_file << "0 0 1" << std::endl;
        // lens distortion coefficients
        out_file << dist_coeffs_[0] << " " << dist_coeffs_[1] << " " << dist_coeffs_[2] << " "
                << dist_coeffs_[3] << " " << dist_coeffs_[4] << std::endl;

        // close file
        out_file.close();

        return true;
    }


    void Camera::print() const
    {
        std::cout << "Camera model:" << std::endl;
        std::cout << "   size: " << width_ << "x" << height_ << std::endl;
        std::cout << "   intrinsics: fx=" << K_(0, 0) << ", fy=" << K_(1, 1) << ", cx=" << K_(0, 2) << ", cy=" << K_(1, 2) << std::endl;
        std::cout << "   distortion: " << dist_coeffs_.transpose() << std::endl;
    }


    Mat3f Camera::defaultIntriniscs() const
    {
        // default intrinsics
        Mat3f K;
        K << 525.0f, 0.0f, 319.5f,
            0.0f, 525.0f, 239.5f,
            0.0f, 0.0f, 1.0f;
        return K;
    }


    Mat3 Camera::convert(const Vec4 &intrinsics) const
    {
        Mat3 K = Mat3::Identity();
        K(0, 0) = intrinsics[0];
        K(1, 1) = intrinsics[1];
        K(0, 2) = intrinsics[2];
        K(1, 2) = intrinsics[3];
        return K;
    }


    Vec4f Camera::convert(const Mat3f &intrinsics) const
    {
        return Vec4f(intrinsics(0, 0), intrinsics(1, 1), intrinsics(0, 2), intrinsics(1, 2));
    }

} // namespace nv
