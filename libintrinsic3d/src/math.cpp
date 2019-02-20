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

#include <nv/math.h>

#include <iostream>
#include <nv/camera.h>


namespace nv
{
namespace math
{

    float robustKernel(float val, float thres)
    {
        float div = (1.0f + thres * val);
        return 1.0f / (div * div * div);
    }


    bool withinBounds(const Vec6i& bounds, const Vec3i &v_pos)
    {
        return !(v_pos[0] < bounds[0] || v_pos[0] > bounds[1] ||
                v_pos[1] < bounds[2] || v_pos[1] > bounds[3] ||
                v_pos[2] < bounds[4] || v_pos[2] > bounds[5]);
    }


    bool withinBounds(const Vec6f& bounds, const Vec3i &v_pos)
    {
        return !(v_pos[0] < bounds[0] || v_pos[0] > bounds[1] ||
                v_pos[1] < bounds[2] || v_pos[1] > bounds[3] ||
                v_pos[2] < bounds[4] || v_pos[2] > bounds[5]);
    }


    bool withinBounds(const Vec6f& bounds, const Vec3f &p)
    {
        return !(p[0] < bounds[0] || p[0] > bounds[1] ||
                p[1] < bounds[2] || p[1] > bounds[3] ||
                p[2] < bounds[4] || p[2] > bounds[5]);
    }


    template <typename T>
    T average(const float weights[8], const T values[8])
    {
        // compute average value
        T avg;
        avg.setZero();
        float sum_weights = 0.0f;
        for (size_t i = 0; i < 8; i++)
        {
            float w = weights[i];
            if (w == 0.0f)
                continue;
            if (sum_weights == 0.0f)
                avg = w * values[i];
            else
                avg += w * values[i];
            sum_weights += w;
        }
        // normalize
        if (sum_weights != 0.0f)
            avg = avg * (1.0f / sum_weights);
        return avg;
    }
    template Vec3f average(const float weights[8], const Vec3f values[8]);
    template Vec3 average(const float weights[8], const Vec3 values[8]);
    template Eigen::VectorXf average(const float weights[8], const Eigen::VectorXf values[8]);
    template Eigen::VectorXd average(const float weights[8], const Eigen::VectorXd values[8]);


    void interpolationWeights(const Vec3f& pos, Vec3i coords[8], float weights[8])
    {
        // round down coordinates
        Vec3i v0 = floor(pos);

        // coordinates
        coords[0] = Vec3i(v0[0], v0[1], v0[2]);
        coords[1] = Vec3i(v0[0] + 1, v0[1], v0[2]);
        coords[2] = Vec3i(v0[0], v0[1] + 1, v0[2]);
        coords[3] = Vec3i(v0[0], v0[1], v0[2] + 1);
        coords[4] = Vec3i(v0[0] + 1, v0[1] + 1, v0[2]);
        coords[5] = Vec3i(v0[0], v0[1] + 1, v0[2] + 1);
        coords[6] = Vec3i(v0[0] + 1, v0[1], v0[2] + 1);
        coords[7] = Vec3i(v0[0] + 1, v0[1] + 1, v0[2] + 1);

        // weights
        Vec3f weight = pos - v0.cast<float>();
        weights[0] = (1.0f - weight[0])*(1.0f - weight[1])*(1.0f - weight[2]);
        weights[1] = weight[0] * (1.0f - weight[1])*(1.0f - weight[2]);
        weights[2] = (1.0f - weight[0])* weight[1] * (1.0f - weight[2]);
        weights[3] = (1.0f - weight[0])*(1.0f - weight[1]) * weight[2];
        weights[4] = weight[0] * weight[1] * (1.0f - weight[2]);
        weights[5] = (1.0f - weight[0]) * weight[1] * weight[2];
        weights[6] = weight[0] * (1.0f - weight[1]) * weight[2];
        weights[7] = weight[0] * weight[1] * weight[2];
    }


    void computeFrustumPoints(const Camera& cam,
                              const float depth_min, const float depth_max,
                              std::vector<Vec3f> &corner_points)
    {
        int width = cam.width();
        int height = cam.height();
        corner_points.resize(8);

        corner_points[0] = cam.unproject2(0, 0, depth_min);
        corner_points[1] = cam.unproject2(width - 1, 0, depth_min);
        corner_points[2] = cam.unproject2(width - 1, height - 1, depth_min);
        corner_points[3] = cam.unproject2(0, height - 1, depth_min);

        corner_points[4] = cam.unproject2(0, 0, depth_max);
        corner_points[5] = cam.unproject2(width - 1, 0, depth_max);
        corner_points[6] = cam.unproject2(width - 1, height - 1, depth_max);
        corner_points[7] = cam.unproject2(0, height - 1, depth_max);
    }


    Mat4 poseVecAAToMat(const Vec6 &pose_vec_aa)
    {
        // convert pose vector (angle axis and translation) to transformation matrix
        Mat4 pose = Mat4::Identity();
        // convert angle axis representation to rotation matrix
        Vec3 vec_aa = pose_vec_aa.topRows<3>();
        Mat3 rot = Eigen::AngleAxisd(vec_aa.norm(), vec_aa.normalized()).matrix();
        //ceres::AngleAxisToRotationMatrix<double>(vec_aa.data(), rot.data());
        pose.topLeftCorner<3, 3>() = rot;
        // translation
        pose.topRightCorner<3, 1>() = pose_vec_aa.bottomRows<3>();
        return pose;
    }


    Vec6 poseMatToVecAA(const Mat4 &pose)
    {
        // convert transformation matrix to pose vector (angle axis and translation)
        Vec6 pose_vec;
        // convert rotation matrix to angle axis representation
        Mat3 R = pose.topLeftCorner<3, 3>();
        Eigen::AngleAxisd aa(R);
        Vec3 rot_aa = aa.axis() * aa.angle();
        //ceres::RotationMatrixToAngleAxis<double>(R.data(), rotAA.data());
        pose_vec.topRows<3>() = rot_aa;
        // translation
        pose_vec.bottomRows<3>() = pose.topRightCorner<3, 1>();
        return pose_vec;
    }

} // namespace math
} // namespace nv
