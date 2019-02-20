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

#include <nv/refinement/surface_stab_regularizer.h>

#include <iostream>


namespace nv
{

    SurfaceStabRegularizer::SurfaceStabRegularizer(double sdf) :
		sdf_(sdf)
	{
	}


    SurfaceStabRegularizer::~SurfaceStabRegularizer()
	{
	}


    VoxelResidual SurfaceStabRegularizer::create(SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos)
	{
        SurfaceStabRegularizer* ss_cost = new SurfaceStabRegularizer(grid->voxel(v_pos).sdf);
        ceres::CostFunction* ss_cost_function = new ceres::AutoDiffCostFunction<SurfaceStabRegularizer, 1, 1>(ss_cost);

		VoxelResidual r;
        r.cost = ss_cost_function;
		r.weight = 1.0;
        r.params.push_back(&(grid->voxel(v_pos).sdf_refined));
		return r;
	}

} // namespace nv
