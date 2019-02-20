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
#include <nv/sparse_voxel_grid.h>


/**
 * @brief   SDF algorithms
 * @author  Robert Maier <robert.maier@tum.de>
 */
namespace nv
{
namespace SDFAlgorithms
{
    SparseVoxelGrid<VoxelSBR>* convert(SparseVoxelGrid<Voxel>* grid);

    std::vector<Vec3i> collectRingNeighborhood(const Vec3i &v_pos);
    std::vector<Vec3i> collectFullNeighborhood(const Vec3i &v_pos, int size = 1);

    template <class T>
    bool interpolate(const SparseVoxelGrid<T>* grid, const Vec3f& voxel_pos, T* voxel_out);

    template <class T>
    SparseVoxelGrid<T>* upsample(const SparseVoxelGrid<T>* grid);

    bool checkVoxelsValid(SparseVoxelGrid<VoxelSBR>* grid, const std::vector<Vec3i> &v_pos_neighbors);

    bool applyRefinedSdf(SparseVoxelGrid<VoxelSBR>* grid);

    template <class T>
    bool correctSDF(SparseVoxelGrid<T>* grid, unsigned int num_iter = 10);

    template <class T>
    bool clearInvalidVoxels(SparseVoxelGrid<T>* grid);

    void clearVoxelsOutsideThinShell(SparseVoxelGrid<VoxelSBR>* grid, double thres_shell);

} // namespace SDFAlgorithms
} // namespace nv
