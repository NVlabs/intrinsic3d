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

#include <nv/refinement/shading_cost.h>

#include <iostream>

#include <ceres/cubic_interpolation.h>

#include <nv/rgbd/processing.h>
#include <nv/sdf/operators.h>
#include <nv/lighting/subvolumes.h>


namespace nv
{

    ShadingCost::ShadingCost(const Vec3i &v_pos, const Eigen::VectorXd &sh_coeffs, const ShadingCostData* data) :
        v_pos_(v_pos),
        sh_coeffs_(sh_coeffs),
        data_(data)
	{
	}


    ShadingCost::~ShadingCost()
	{
	}
	

    VoxelResidual ShadingCost::create(SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos, Vec6 &pose_vec, Vec4 &intrinsics,
                                              Vec5 &dist_coeffs, const Eigen::VectorXd &sh_coeffs, const ShadingCostData* data)
	{
		VoxelResidual r;

		// use only voxels for which gradients can be computed
        if (!grid->exists(v_pos[0] + 2, v_pos[1], v_pos[2]) ||
            !grid->exists(v_pos[0], v_pos[1] + 2, v_pos[2]) ||
            !grid->exists(v_pos[0], v_pos[1], v_pos[2] + 2) ||
            !grid->exists(v_pos[0], v_pos[1] + 1, v_pos[2] + 1) ||
            !grid->exists(v_pos[0] + 1, v_pos[1] + 1, v_pos[2]) ||
            !grid->exists(v_pos[0] + 1, v_pos[1], v_pos[2] + 1))
			return r;

		// compute surface normal
        Vec3f n = SDFOperators::computeSurfaceNormal(grid, v_pos);
        if (n.isZero())
			return r;
		
		// valid observation
		r.weight = 1.0;

		// create cost
        //std::cout << "Intrinsic3dShadingCost size " << sizeof(Intrinsic3dShadingCost) << std::endl;
        ShadingCost* gs_cost = new ShadingCost(v_pos, sh_coeffs, data);
		// create cost function with gradients (requires more parameter blocks)
        ceres::DynamicAutoDiffCostFunction<ShadingCost, 4>* gs_cost_function = new ceres::DynamicAutoDiffCostFunction<ShadingCost, 4>(gs_cost);

        // parameter blocks
		// parameter block for voxel sdf values
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1] + 1, v_pos[2]).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1] + 2, v_pos[2]).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1] + 1, v_pos[2] + 1).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1], v_pos[2] + 1).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1], v_pos[2] + 2).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0] + 1, v_pos[1], v_pos[2]).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0] + 1, v_pos[1] + 1, v_pos[2]).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0] + 1, v_pos[1], v_pos[2] + 1).sdf_refined));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0] + 2, v_pos[1], v_pos[2]).sdf_refined));

		// parameter block for voxel albedo
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos).albedo));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0] + 1, v_pos[1], v_pos[2]).albedo));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1] + 1, v_pos[2]).albedo));
        gs_cost_function->AddParameterBlock(1);
        r.params.push_back(&(grid->voxel(v_pos[0], v_pos[1], v_pos[2] + 1).albedo));

		// parameter blocks for pose parameters
        gs_cost_function->AddParameterBlock(6);
        r.params.push_back(pose_vec.data());

		// parameter blocks for camera intrinsics parameters
        gs_cost_function->AddParameterBlock(4);
		r.params.push_back(intrinsics.data());

		// parameter blocks for radial and tangential distortion parameters
        gs_cost_function->AddParameterBlock(5);
        r.params.push_back(dist_coeffs.data());

		// number of residuals
        gs_cost_function->SetNumResiduals(1);

		// check if residual is valid
		double residual;
		//bool eval = gsCostFunction->Evaluate(&(r.params[0]), &residual, 0);
        bool eval = (*gs_cost)(&(r.params[0]), &residual);
		//std::cout << "residual " << residual << std::endl;
		if (!eval || residual == NV_INVALID_RESIDUAL)
		{
            delete gs_cost_function;
            gs_cost_function = nullptr;
            r.cost = nullptr;
		}
		else
            r.cost = gs_cost_function;

		return r;
	}

} // namespace nv
