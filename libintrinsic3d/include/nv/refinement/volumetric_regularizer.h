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

#include <nv/refinement/cost.h>
#include <nv/sdf/operators.h>
#include <nv/sparse_voxel_grid.h>


namespace nv
{

    /**
     * @brief   Class to generate Intrinsic3D volumetric regularizer residuals
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class VolumetricRegularizer
	{
	public:
        VolumetricRegularizer();
        ~VolumetricRegularizer();

        static VoxelResidual create(SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos);

		template <typename T>
		bool operator()(const T* const sdf,
                        const T* const sdf_x_plus,
                        const T* const sdf_x_minus,
                        const T* const sdf_y_plus,
                        const T* const sdf_y_minus,
                        const T* const sdf_z_plus,
                        const T* const sdf_z_minus,
                        T* residual) const
		{
			// compute discrete volumetric Laplace operator
            residual[0] = SDFOperators::computeLaplacian(sdf[0], sdf_x_plus[0], sdf_x_minus[0], sdf_y_plus[0], sdf_y_minus[0], sdf_z_plus[0], sdf_z_minus[0]);
			return true;
		}

	private:

	};

} // namespace nv
