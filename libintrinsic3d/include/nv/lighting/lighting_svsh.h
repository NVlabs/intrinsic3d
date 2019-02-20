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

#include <vector>
#include <nv/mat.h>
#include <nv/sparse_voxel_grid.h>
#include <nv/lighting/subvolumes.h>

namespace nv
{

    /**
     * @brief   Lighting estimation using Spatially-Varying Spherical Harmonics
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class LightingSVSH
	{
	public:
        LightingSVSH(const SparseVoxelGrid<VoxelSBR>* grid, float subvolume_size, double lambda_reg, double thres_shell = 0.0, bool weighted = false);
        ~LightingSVSH();

        bool estimate();

        const Subvolumes& subvolumes() const;

        std::vector<Eigen::VectorXd> shCoeffs() const;

        bool interpolate(const Vec3i &v_pos, Eigen::VectorXd &sh_coeffs) const;

        bool computeVoxelShCoeffs(std::vector<Eigen::VectorXd> &voxel_coeffs) const;

    protected:

        const SparseVoxelGrid<VoxelSBR>* grid_;
        float subvolume_size_;
        double thres_shell_;
        bool weighted_;
        double lambda_reg_;
        Subvolumes subvolumes_;
        std::vector<Eigen::VectorXd> sh_coeffs_;
	};

} // namespace nv
