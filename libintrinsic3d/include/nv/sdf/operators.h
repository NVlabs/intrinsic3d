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
 * @brief   SDF operators
 *          (e.g. for calculating the normal or Laplacian of the SDF etc)
 * @author  Robert Maier <robert.maier@tum.de>
 */
namespace nv
{

namespace SDFOperators
{

    template <typename T>
    inline void voxelToWorld(const int* v_coords, T voxel_size, T v_pos[3])
    {
        for (size_t i = 0; i < 3; ++i)
            v_pos[i] = static_cast<T>(v_coords[i]) * voxel_size;
    }

    Vec3f voxelCenterToIso(const SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_coord, const Vec3f &n);
    Vec3f voxelCenterToIso(const Vec3f &pt, const Vec3f &n, const float sdf);

    template <typename T>
    inline void voxelCenterToIso(const T p[3], const T n[3], const T sdf, T p_iso[3])
    {
        // compute closest point to iso-surface
        p_iso[0] = p[0] - n[0] * sdf;
        p_iso[1] = p[1] - n[1] * sdf;
        p_iso[2] = p[2] - n[2] * sdf;
    }

    Vec3f computeSurfaceNormal(const SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& voxelcoord);

    template <typename T>
    bool computeNormal(const T sdf, const T sdf_x_plus, const T sdf_y_plus, const T sdf_z_plus, T* n)
    {
        // compute surface normal using forward differences
        n[0] = sdf_x_plus - sdf;
        n[1] = sdf_y_plus - sdf;
        n[2] = sdf_z_plus - sdf;
        // normalize normal
        T n_length = sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
        if (n_length > T(0.0))
        {
            n[0] = n[0] / n_length;
            n[1] = n[1] / n_length;
            n[2] = n[2] / n_length;
        }
        return true;
    }

    float laplacian(const SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos);

    template <typename T>
    T computeLaplacian(const T sdf,
                       const T sdf_x_plus, const T sdf_x_minus,
                       const T sdf_y_plus, const T sdf_y_minus,
                       const T sdf_z_plus, const T sdf_z_minus)
    {
        // compute discrete volumetric Laplace operator
        // practical derivation:
        // - second order derivative dxx in x-direction
        //   (central differences of first order derivatives dx_forward and dx_backward):
        //   dxx = (dx_forward - dx_backward) = (d(x+1) - d(x)) - (d(x) - d(x-1)) =
        //       = d(x+1) + d(x-1) - 2*d(x)
        // - other second order derivatives dyy and dzz similarly
        // - Laplacian: L = dxx + dyy + dzz
        T dxx = sdf_x_plus + sdf_x_minus - T(2.0) * sdf;
        T dyy = sdf_y_plus + sdf_y_minus - T(2.0) * sdf;
        T dzz = sdf_z_plus + sdf_z_minus - T(2.0) * sdf;
        T lap = dxx + dyy + dzz;
        return lap;
    }

    Vec3f intensityGradient(const SparseVoxelGrid<VoxelSBR>* grid, const Vec3i& v_pos);

    double sdfToWeight(double sdf, double truncation);

} // namespace SDFOperators
} // namespace nv
