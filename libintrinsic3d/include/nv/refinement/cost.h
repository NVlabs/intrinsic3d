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

#include <iostream>

#include <nv/mat.h>
#include <nv/lighting/lighting_svsh.h>
#include <nv/sdf/operators.h>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <ceres/rotation.h>


#define NV_INVALID_RESIDUAL 0.0


/**
 * @brief   Helper functions and data structures for SDF refinement
 * @author  Robert Maier <robert.maier@tum.de>
 */
namespace nv
{

    /**
     * @brief   Data structure for per-voxel residual
     * @author  Robert Maier <robert.maier@tum.de>
     */
	struct VoxelResidual
	{
		VoxelResidual()
		{
            cost = nullptr;
			weight = 0.0;
		}

		ceres::CostFunction* cost;
		std::vector<double*> params;
		double weight;
	};


	template <typename T>
	bool isValid(const T val)
	{
		return !(ceres::IsNaN(val) || ceres::IsInfinite(val));
	}


	template <typename T>
    void transform(const T pose_rot_aa[3], const T pose_t[3], const T p[3], T p_tf[3])
	{
		//rotate point using built-in angle-axis rotation function
        ceres::AngleAxisRotatePoint(pose_rot_aa, p, p_tf);
		//translate point
        p_tf[0] += pose_t[0];
        p_tf[1] += pose_t[1];
        p_tf[2] += pose_t[2];
	}


	template <typename T>
    void transformVoxelIso(T voxel_size, const T pose_rot_aa[3], const T pose_t[3], const int* v_coords, const T sdf, const T normal[3], T v_pos_tf[3])
	{
		// convert discrete 3D voxel coordinates to continuous 3D voxel position
        T v_pos[3];
        SDFOperators::voxelToWorld(v_coords, voxel_size, v_pos);

		// compute closest point to iso-surface
        T v_pos_iso[3];
        SDFOperators::voxelCenterToIso(v_pos, normal, sdf, v_pos_iso);

		// transform continuous 3D voxel position using pose (angle-axis rotation and translation)
        transform(pose_rot_aa, pose_t, v_pos_iso, v_pos_tf);
	}


	template <typename T>
    bool interpolate(const float* ptr_intensity, int w, int h, const T p2d[2], T* val_out)
	{
		// lookup interpolated intensity (automatic differentiation)
		typedef ceres::Grid2D<float, 1, true, true> ImageDataType;
        ImageDataType array(ptr_intensity, 0, h, 0, w);
		ceres::BiCubicInterpolator<ImageDataType> interpolator(array);
		T lum;
		interpolator.Evaluate(p2d[1], p2d[0], &lum);
		if (isValid(lum))
		{
            *val_out = lum;
			return true;
		}
		else
		{
            *val_out = T(0.0);
			return false;
		}
	}


    inline double computeVaryingLambda(int iteration, int num_iterations, double lambda0, double lambda1)
	{
		double lambda;
        if (num_iterations <= 1)
        {
			lambda = lambda0;
        }
		else
		{
            double step = (lambda1 - lambda0) / static_cast<double>(num_iterations - 1);
            lambda = lambda0 + step * static_cast<double>(iteration);
		}
		return lambda;
	}


    inline double pyramidLevelToScale(const int lvl)
    {
        double scale = 1.0 / std::pow(2.0, lvl);
        return scale;
    }

} // namespace nv
