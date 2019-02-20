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

#include <nv/refinement/albedo_regularizer.h>

#include <iostream>


namespace nv
{

    AlbedoRegularizer::AlbedoRegularizer()
	{
	}


    AlbedoRegularizer::~AlbedoRegularizer()
	{
	}


    VoxelResidual AlbedoRegularizer::create(SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos, const Vec3i& v_pos_nb)
	{
		VoxelResidual r;
        if (!grid->valid(v_pos) || !grid->valid(v_pos_nb))
			return r;

		// retrieve voxels
        VoxelSBR& v = grid->voxel(v_pos);
        VoxelSBR& v_nb = grid->voxel(v_pos_nb);

		// compute chromacity difference
		Vec3f c = v.color.cast<float>() * (1.0f / 255.0f);
        Vec3f c_nb = v_nb.color.cast<float>() * (1.0f / 255.0f);
		float lum = intensity(v.color);
        float lum_nb = intensity(v_nb.color);
        float chroma_diff = ((c / lum) - (c_nb / lum_nb)).norm();
        float lum_diff = 1.0f;
        //chroma_diff = math::robustKernel(chroma_diff);
        chroma_diff = std::max(1.0f - chroma_diff, 0.01f);
		// compute chromacity weight by applying robust kernel on chromacity difference
        double w = static_cast<double>(chroma_diff) * static_cast<double>(lum_diff);
		if (std::isnan(w) || std::isinf(w))
			return r;

		// albedo regularizer cost Ea
        AlbedoRegularizer* ar_cost = new AlbedoRegularizer();
        ceres::CostFunction* ar_cost_function = new ceres::AutoDiffCostFunction<AlbedoRegularizer, 1, 1, 1>(ar_cost);
        r.cost = ar_cost_function;
		r.weight = w;
		// parameters
		r.params.push_back(&(v.albedo));
        r.params.push_back(&(v_nb.albedo));

		return r;
	}

} // namespace nv
