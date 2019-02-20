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

#include <nv/refinement/volumetric_regularizer.h>

#include <iostream>

#include <nv/sdf/algorithms.h>


namespace nv
{

    VolumetricRegularizer::VolumetricRegularizer()
	{
	}


    VolumetricRegularizer::~VolumetricRegularizer()
	{
	}


    VoxelResidual VolumetricRegularizer::create(SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos)
	{
		VoxelResidual r;

		// collect 1-ring neighborhood voxels
        std::vector<Vec3i> v_pos_neighbors = SDFAlgorithms::collectRingNeighborhood(v_pos);
		// use only voxels with valid 1-ring-neighborhood
        if (!SDFAlgorithms::checkVoxelsValid(grid, v_pos_neighbors))
			return r;

		// reduce weight if sdf differences are too large
        double w = 1.0;

		// volumetric regularizer cost Er
        VolumetricRegularizer* vr_cost = new VolumetricRegularizer();
        r.cost = new ceres::AutoDiffCostFunction<VolumetricRegularizer, 1, 1, 1, 1, 1, 1, 1, 1>(vr_cost);
		r.weight = w;
        r.params.push_back(&(grid->voxel(v_pos).sdf_refined));
        r.params.push_back(&(grid->voxel(v_pos_neighbors[0]).sdf_refined));
        r.params.push_back(&(grid->voxel(v_pos_neighbors[1]).sdf_refined));
        r.params.push_back(&(grid->voxel(v_pos_neighbors[2]).sdf_refined));
        r.params.push_back(&(grid->voxel(v_pos_neighbors[3]).sdf_refined));
        r.params.push_back(&(grid->voxel(v_pos_neighbors[4]).sdf_refined));
        r.params.push_back(&(grid->voxel(v_pos_neighbors[5]).sdf_refined));

		return r;
	}

} // namespace nv
