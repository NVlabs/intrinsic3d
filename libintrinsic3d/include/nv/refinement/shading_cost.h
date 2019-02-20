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

#include <ceres/ceres.h>

#include <nv/camera.h>
#include <nv/refinement/cost.h>
#include <nv/shading.h>
#include <nv/sparse_voxel_grid.h>


namespace nv
{

    /**
     * @brief   Class to generate Intrinsic3D data term residuals
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class ShadingCostData
    {
    public:
        ShadingCostData(const int rgbd_lvl, const double vx_size, const int w, const int h, const float* ptr_lum) :
            pyr_scale(pyramidLevelToScale(rgbd_lvl)),
            voxel_size(vx_size),
            width(w),
            height(h),
            ptr_intensity(ptr_lum)
        {
        }

        ~ShadingCostData()
        {
        }

        double pyr_scale;
        double voxel_size;
        int width;
        int height;
        const float* ptr_intensity;
    };


    class ShadingCost
	{
	public:
        ShadingCost(const Vec3i &v_pos, const Eigen::VectorXd &sh_coeffs, const ShadingCostData* data);
        ~ShadingCost();

        static VoxelResidual create(SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos, Vec6 &pose_vec, Vec4 &intrinsics,
                                    Vec5 &distortion_coeffs, const Eigen::VectorXd &sh_coeffs, const ShadingCostData* data);

		template <typename T>
		bool operator()(T const* const* params, T* residuals) const
		{
			// sdf parameters
			const T sdf_x0y0z0 = params[0][0];
			const T sdf_x0y1z0 = params[1][0];
			const T sdf_x0y2z0 = params[2][0];
			const T sdf_x0y1z1 = params[3][0];
			const T sdf_x0y0z1 = params[4][0];
			const T sdf_x0y0z2 = params[5][0];
			const T sdf_x1y0z0 = params[6][0];
			const T sdf_x1y1z0 = params[7][0];
			const T sdf_x1y0z1 = params[8][0];
			const T sdf_x2y0z0 = params[9][0];
			// albedo parameters
			const T albedo_x0y0z0 = params[10][0];
			const T albedo_x1y0z0 = params[11][0];
			const T albedo_x0y1z0 = params[12][0];
			const T albedo_x0y0z1 = params[13][0];
			// pose parameters
			// pose: angle-axis rotation
            T pose_rot_aa[3];
            pose_rot_aa[0] = params[14][0];
            pose_rot_aa[1] = params[14][1];
            pose_rot_aa[2] = params[14][2];
			// pose: translation
            T pose_t[3];
            pose_t[0] = params[14][3];
            pose_t[1] = params[14][4];
            pose_t[2] = params[14][5];
            // reference intensity image
            const float* ptr_intensity = data_->ptr_intensity;
            int w = data_->width;
            int h = data_->height;
            // camera model
            CameraT<T> cam;
            cam.w = w;
            cam.h = h;
            // camera intrinsics parameters
            const T pyr_scale = T(data_->pyr_scale);
            cam.fx = params[15][0] * pyr_scale;
            cam.fy = params[15][1] * pyr_scale;
            cam.cx = params[15][2] * pyr_scale;
            cam.cy = params[15][3] * pyr_scale;
			// radial and tangential distortion parameters (0,1,2=radial, 3,4=tangential)
            cam.dist_coeffs = &(params[16][0]);

			// compute normals (sdf gradient) for voxels
			T n_x0y0z0[3], n_x1y0z0[3], n_x0y1z0[3], n_x0y0z1[3];
			SDFOperators::computeNormal(sdf_x0y0z0, sdf_x1y0z0, sdf_x0y1z0, sdf_x0y0z1, n_x0y0z0);
			SDFOperators::computeNormal(sdf_x1y0z0, sdf_x2y0z0, sdf_x1y1z0, sdf_x1y0z1, n_x1y0z0);
			SDFOperators::computeNormal(sdf_x0y1z0, sdf_x1y1z0, sdf_x0y2z0, sdf_x0y1z1, n_x0y1z0);
			SDFOperators::computeNormal(sdf_x0y0z1, sdf_x1y0z1, sdf_x0y1z1, sdf_x0y0z2, n_x0y0z1);

			// compute 3D coords for all voxels required for gradient and
			// transform them into input view coordinate system
            Vec3i v_pos_x_plus = v_pos_ + Vec3i(1, 0, 0);
            Vec3i v_pos_y_plus = v_pos_ + Vec3i(0, 1, 0);
            Vec3i v_pos_z_plus = v_pos_ + Vec3i(0, 0, 1);
            T v_pos_iso[3], v_pos_iso_x_plus[3], v_pos_iso_y_plus[3], v_pos_iso_z_plus[3];
            // use points closest to iso-surface
            const T voxel_size = static_cast<T>(data_->voxel_size);
            transformVoxelIso(voxel_size, pose_rot_aa, pose_t, v_pos_.data(), sdf_x0y0z0, n_x0y0z0, v_pos_iso);
            transformVoxelIso(voxel_size, pose_rot_aa, pose_t, v_pos_x_plus.data(), sdf_x1y0z0, n_x1y0z0, v_pos_iso_x_plus);
            transformVoxelIso(voxel_size, pose_rot_aa, pose_t, v_pos_y_plus.data(), sdf_x0y1z0, n_x0y1z0, v_pos_iso_y_plus);
            transformVoxelIso(voxel_size, pose_rot_aa, pose_t, v_pos_z_plus.data(), sdf_x0y0z1, n_x0y0z1, v_pos_iso_z_plus);

            // project to 2D image coordinates
            T v_pos_2d[3], v_pos_x_plus_2d[3], v_pos_y_plus_2d[3], v_pos_z_plus_2d[3];
            bool valid0 = cam.project(v_pos_iso, v_pos_2d);
            bool valid1 = cam.project(v_pos_iso_x_plus, v_pos_x_plus_2d);
            bool valid2 = cam.project(v_pos_iso_y_plus, v_pos_y_plus_2d);
            bool valid3 = cam.project(v_pos_iso_z_plus, v_pos_z_plus_2d);
			if (!valid0 || !valid1 || !valid2 || !valid3)
			{
				residuals[0] = T(NV_INVALID_RESIDUAL);
				return true;
			}

			// direct voxel intensities lookup in input view
			T lum[4];
            valid0 = interpolate(ptr_intensity, w, h, v_pos_2d, &(lum[0]));
            valid1 = interpolate(ptr_intensity, w, h, v_pos_x_plus_2d, &(lum[1]));
            valid2 = interpolate(ptr_intensity, w, h, v_pos_y_plus_2d, &(lum[2]));
            valid3 = interpolate(ptr_intensity, w, h, v_pos_z_plus_2d, &(lum[3]));
			if (!valid0 || !valid1 || !valid2 || !valid3)
			{
				residuals[0] = T(NV_INVALID_RESIDUAL);
				return true;
			}

			// compute shading
			T shading[4];
            valid0 = Shading::computeShading(sh_coeffs_, n_x0y0z0, albedo_x0y0z0, &(shading[0]));
            valid1 = Shading::computeShading(sh_coeffs_, n_x1y0z0, albedo_x1y0z0, &(shading[1]));
            valid2 = Shading::computeShading(sh_coeffs_, n_x0y1z0, albedo_x0y1z0, &(shading[2]));
            valid3 = Shading::computeShading(sh_coeffs_, n_x0y0z1, albedo_x0y0z1, &(shading[3]));
			if (!valid0 || !valid1 || !valid2 || !valid3)
			{
				residuals[0] = T(NV_INVALID_RESIDUAL);
				return true;
			}

			// compute residual
            residuals[0] = Shading::computeShadingGradientDifference(lum, shading);
			// compute residual as the norm of the residual vector
			if (!isValid(residuals[0]))
			{
				residuals[0] = T(NV_INVALID_RESIDUAL);
				return true;
			}

			return true;
		}

    private:
        Vec3i v_pos_;
        const Eigen::VectorXd& sh_coeffs_;
        const ShadingCostData* data_;
	};

} // namespace nv
