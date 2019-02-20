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

#include <nv/sdf/algorithms.h>

#include <iostream>
#include <type_traits>
#include <unordered_set>

#include <nv/color_util.h>
#include <nv/math.h>


namespace nv
{
namespace SDFAlgorithms
{

    SparseVoxelGrid<VoxelSBR>* convert(SparseVoxelGrid<Voxel>* grid)
    {
        SparseVoxelGrid<VoxelSBR>* grid_sbr = SparseVoxelGrid<VoxelSBR>::create(grid->voxelSize(), grid->depthMin(), grid->depthMax());
        if (!grid_sbr)
            return nullptr;

        for (auto itr = grid->begin(); itr != grid->end(); itr++)
        {
            // get voxel
            const Vec3i& pos_grid = itr->first;
            const Voxel& v = itr->second;
            // fill specialized voxel type
            VoxelSBR v_sbr;
            v_sbr.sdf = static_cast<double>(v.sdf);
            v_sbr.color = v.color;
            v_sbr.weight = v.weight;
            v_sbr.sdf_refined = static_cast<double>(v.sdf);
            // add voxel
            grid_sbr->setVoxel(pos_grid, v_sbr);
        }

        // clear invalid voxels
        SDFAlgorithms::clearInvalidVoxels(grid_sbr);

        return grid_sbr;
    }


    std::vector<Vec3i> collectRingNeighborhood(const Vec3i &v_pos)
    {
        std::vector<Vec3i> v_pos_neighbors(6);
        for (int nb = 0; nb < 6; ++nb)
        {
            // retrieve neighbor voxel
            Vec3i v_pos_nb = v_pos;
            if (nb == 0) v_pos_nb[0] += 1;
            else if (nb == 1) v_pos_nb[0] -= 1;
            else if (nb == 2) v_pos_nb[1] += 1;
            else if (nb == 3) v_pos_nb[1] -= 1;
            else if (nb == 4) v_pos_nb[2] += 1;
            else if (nb == 5) v_pos_nb[2] -= 1;
            v_pos_neighbors[nb] = v_pos_nb;
        }
        return v_pos_neighbors;
    }


    std::vector<Vec3i> collectFullNeighborhood(const Vec3i &v_pos, int size)
    {
        size_t nb_size = 1 + 2 * size;
        size_t num_nb = nb_size * nb_size * nb_size - 1;
        std::vector<Vec3i> v_pos_neighbors(num_nb);
        int cnt = 0;
        for (int z = -size; z <= size; z++)
        {
            for (int y = -size; y <= size; y++)
            {
                for (int x = -size; x <= size; x++)
                {
                    if (x == 0 && y == 0 && z == 0)
                        continue;
                    Vec3i v_pos_nb = v_pos + Vec3i(x, y, z);
                    v_pos_neighbors[cnt] = v_pos_nb;
                    ++cnt;
                }
            }
        }
        return v_pos_neighbors;
    }


    template <class T>
    bool interpolate(const SparseVoxelGrid<T>* grid, const Vec3f& voxel_pos, T* voxel_out)
    {
        if (!voxel_out)
            return false;

        Voxel* vx = nullptr;
        VoxelSBR* vx_sbr = nullptr;
        if (std::is_same<T, VoxelSBR>::value)
            vx_sbr = reinterpret_cast<VoxelSBR*>(voxel_out);
        else if (std::is_same<T, Voxel>::value)
            vx = reinterpret_cast<Voxel*>(voxel_out);
        if (!vx && !vx_sbr)
            return false;

        // average values
        float avg_weight = 0.0f;
        float avg_sdf = 0.0f;
        Vec3f avg_color(0.0f, 0.0f, 0.0f);
        float avg_albedo = 0.0f;
        float avg_sdf_refined = 0.0f;

        // trilinear interpolation
        Vec3i coords[8];
        float weights[8];
        math::interpolationWeights(voxel_pos, coords, weights);

        float sum_weights = 0.0f;
        int cnt_valid = 0;
        for (size_t i = 0; i < 8; i++)
        {
            if (!grid->valid(coords[i]))
                continue;
            const T& v = grid->voxel(coords[i]);

            float w = weights[i];
            avg_sdf += w * (float)v.sdf;
            avg_color += w * Vec3b(v.color).cast<float>();
            avg_weight += w * v.weight;
            if (vx_sbr)
            {
                const VoxelSBR* vx_sbr2 = reinterpret_cast<const VoxelSBR*>(&v);
                avg_albedo += w * (float)vx_sbr2->albedo;
                avg_sdf_refined += w * (float)vx_sbr2->sdf_refined;
            }
            sum_weights += w;
            ++cnt_valid;
        }
        if (sum_weights > 0.0f)
        {
            avg_sdf /= sum_weights;
            avg_color /= sum_weights;
            avg_weight /= sum_weights;
            if (vx_sbr)
            {
                avg_albedo /= sum_weights;
                avg_sdf_refined /= sum_weights;
            }
        }
        if (cnt_valid <= 4)
            avg_weight = 0.0f;

        // fill output voxel
        if (vx_sbr)
        {
            vx_sbr->sdf = avg_sdf;
            vx_sbr->color = round(avg_color).cast<unsigned char>();
            vx_sbr->weight = std::max(avg_weight, 0.0f);
            vx_sbr->albedo = avg_albedo;
            vx_sbr->sdf_refined = avg_sdf_refined;
        }
        else
        {
            vx->sdf = avg_sdf;
            vx->color = round(avg_color).cast<unsigned char>();
            vx->weight = std::max(avg_weight, 0.0f);
        }

        return true;
    }
    template bool interpolate(const SparseVoxelGrid<Voxel>* grid, const Vec3f& voxel_pos, Voxel* voxel_out);
    template bool interpolate(const SparseVoxelGrid<VoxelSBR>* grid, const Vec3f& voxel_pos, VoxelSBR* voxel_out);


    template <class T>
    SparseVoxelGrid<T>* upsample(const SparseVoxelGrid<T>* grid)
    {
        float voxel_size_up = grid->voxelSize() * 0.5f;
        SparseVoxelGrid<T>* grid_up = SparseVoxelGrid<T>::create(voxel_size_up, grid->depthMin(), grid->depthMax());
        if (!grid_up)
            return nullptr;

        // iterate over all voxels
        for (auto itr = grid->begin(); itr != grid->end(); itr++)
        {
            // get voxel
            const Vec3i& pos_grid = itr->first;
            // create eight output voxels for each input voxel
            for (int z = 0; z < 2; ++z)
            {
                for (int y = 0; y < 2; ++y)
                {
                    for (int x = 0; x < 2; ++x)
                    {
                        // convert to upsampled coordinates
                        const Vec3i pos_grid_new = 2 * pos_grid + Vec3i(x, y, z);
                        const Vec3f pos_grid_new_f = pos_grid.cast<float>() + Vec3i(x, y, z).cast<float>() * 0.5f;
                        // trilinear interpolation of neighboring voxel
                        T v;
                        interpolate<T>(grid, pos_grid_new_f, &v);
                        // add voxel
                        grid_up->setVoxel(pos_grid_new, v);
                    }
                }
            }
        }
        return grid_up;
    }
    template SparseVoxelGrid<Voxel>* upsample(const SparseVoxelGrid<Voxel>* grid);
    template SparseVoxelGrid<VoxelSBR>* upsample(const SparseVoxelGrid<VoxelSBR>* grid);


    bool checkVoxelsValid(SparseVoxelGrid<VoxelSBR>* grid, const std::vector<Vec3i> &v_pos_neighbors)
    {
        bool valid = true;
        for (size_t i = 0; i < v_pos_neighbors.size(); ++i)
            if (!grid->valid(v_pos_neighbors[i]))
                valid = false;
        return valid;
    }


    bool applyRefinedSdf(SparseVoxelGrid<VoxelSBR>* grid)
    {
        if (!grid)
            return false;
        for (auto itr = grid->begin(); itr != grid->end(); itr++)
            (itr->second).sdf = (itr->second).sdf_refined;
        return true;
    }


    template <class T>
    bool correctSDF(SparseVoxelGrid<T>* grid, unsigned int num_iter)
    {
        if (!grid)
            return false;

        // correct signed distance field using distance transform
        const int window_size = 1;
        for (unsigned int iter = 0; iter < num_iter; iter++)
        {
            size_t num_updated = 0;
            bool has_update = false;
            for (auto itr = grid->begin(); itr != grid->end(); itr++)
            {
                const Vec3i& v_pos = itr->first;
                T& v = itr->second;
                if (!grid->valid(v_pos))
                    continue;

                const Vec3f v_coord = grid->voxelToWorld(v_pos);
                double sdf = v.sdf;
                if (v.weight == 0.0f)
                {
                    // voxel unseen -> infinite SDF value for correct comparison
                    sdf = std::numeric_limits<double>::infinity();
                }
                double sgn = sdf >= 0.0 ? 1.0 : -1.0;

                // bools checks if there is a neighbor with a smaller distance (+ the dist to the current voxel);
                // if then it updates the distances
                bool voxel_updated = false;
                for (int k = -window_size; k <= window_size; k++)
                {
                    for (int j = -window_size; j <= window_size; j++)
                    {
                        for (int i = -window_size; i <= window_size; i++)
                        {
                            if (k == 0 && j == 0 && i == 0)
                                continue;

                            Vec3i v_pos_nb(v_pos[0] + i, v_pos[1] + j, v_pos[2] + k);
                            if (grid->valid(v_pos_nb))
                            {
                                const T& v_nb = grid->voxel(v_pos_nb);
                                const Vec3f v_coord_nb = grid->voxelToWorld(v_pos_nb);

                                double sdf_nb = v_nb.sdf;
                                double sgn_nb = sdf_nb >= 0.0 ? 1.0 : -1.0;

                                // voxel unseen and neighor behind surface -> invert original sign
                                if (std::isinf(sdf) && sgn_nb < 0.0)
                                    sgn = -sgn;

                                double dist_nb = sdf_nb + sgn_nb * (double)(v_coord - v_coord_nb).norm();
                                if (std::abs(dist_nb) < std::abs(sdf) && sgn == sgn_nb)
                                {
                                    v.sdf = dist_nb;
                                    v.weight = 1.0f;
                                    //v.weight = vNb.weight;
                                    has_update = true;
                                    voxel_updated = true;
                                }
                            }
                        }
                    }
                }
                if (voxel_updated)
                    ++num_updated;
            }
            //std::cout << "corrected voxels: " << numUpdated << std::endl;

            // break if no voxel has been updated any more
            if (!has_update)
                break;
        }

        return true;
    }
    template bool correctSDF(SparseVoxelGrid<Voxel>* grid, unsigned int num_iter);
    template bool correctSDF(SparseVoxelGrid<VoxelSBR>* grid, unsigned int num_iter);


    template <class T>
    bool clearInvalidVoxels(SparseVoxelGrid<T>* grid)
    {
        if (!grid)
            return false;

        // collect invalid voxels
        std::vector<Vec3i> invalid_voxels;
        for (auto itr = grid->begin(); itr != grid->end(); itr++)
        {
            const Vec3i& v_pos = itr->first;
            if (!grid->valid(v_pos))
                invalid_voxels.push_back(v_pos);
        }

        // remove invalid voxels
        for (size_t i = 0; i < invalid_voxels.size(); ++i)
            grid->remove(invalid_voxels[i]);
        //std::cout << invalid_voxels.size() << " invalid voxels removed." << std::endl;

        return true;
    }
    template bool clearInvalidVoxels(SparseVoxelGrid<Voxel>* grid);
    template bool clearInvalidVoxels(SparseVoxelGrid<VoxelSBR>* grid);


    void clearVoxelsOutsideThinShell(SparseVoxelGrid<VoxelSBR>* grid, double thres_shell)
    {
        // collect valid and invalid voxels
        std::unordered_set<Vec3i, std::hash<Vec3i> > valid_voxels;
        std::unordered_set<Vec3i, std::hash<Vec3i> > invalid_voxels;
        for (auto iter = grid->begin(); iter != grid->end(); iter++)
        {
            const Vec3i& v_pos = iter->first;
            VoxelSBR& v = iter->second;

            if (!grid->valid(v_pos) || std::abs(v.sdf_refined) > thres_shell)
                continue;

            // keep voxels that are valid and within thin shell
            if (valid_voxels.find(v_pos) == valid_voxels.end())
                valid_voxels.insert(v_pos);

            // keep voxels within valid 1-ring-neighborhood
            std::vector<Vec3i> v_pos_neighbors = collectRingNeighborhood(v_pos);
            v_pos_neighbors.push_back(Vec3i(v_pos[0] + 2, v_pos[1], v_pos[2]));
            v_pos_neighbors.push_back(Vec3i(v_pos[0], v_pos[1] + 2, v_pos[2]));
            v_pos_neighbors.push_back(Vec3i(v_pos[0], v_pos[1], v_pos[2] + 2));
            for (size_t i = 0; i < v_pos_neighbors.size(); ++i)
            {
                if (grid->exists(v_pos_neighbors[i]))
                {
                    if (valid_voxels.find(v_pos_neighbors[i]) == valid_voxels.end())
                        valid_voxels.insert(v_pos_neighbors[i]);
                }
            }
        }

        // collect invalid voxels
        for (auto iter = grid->begin(); iter != grid->end(); iter++)
        {
            const Vec3i& v_pos = iter->first;
            if (valid_voxels.find(v_pos) != valid_voxels.end())
                continue;

            // voxel is not in valid voxels
            VoxelSBR& v = iter->second;
            // check whether it is inside (<= 0.0) or outside
            bool is_negative = (v.sdf_refined < 0.0);
            // check whether it is close to iso-surface (has zero-crossing in neighborhood)
            bool has_crossing = false;
            if (is_negative)
            {
                // behind surface -> check within larger neighborhood whether voxel can be removed
                std::vector<Vec3i> v_pos_neighbors = collectFullNeighborhood(v_pos, 2);
                for (size_t i = 0; i < v_pos_neighbors.size(); ++i)
                {
                    if (grid->exists(v_pos_neighbors[i]))
                    {
                        const VoxelSBR& vNb = grid->voxel(v_pos_neighbors[i]);
                        if (vNb.sdf_refined >= 0.0)
                        {
                            has_crossing = true;
                            break;
                        }
                    }
                }
            }
            else
            {
                // in front of surface
                std::vector<Vec3i> v_pos_neighbors = collectFullNeighborhood(v_pos, 2);
                for (size_t i = 0; i < v_pos_neighbors.size(); ++i)
                {
                    if (grid->exists(v_pos_neighbors[i]))
                    {
                        const VoxelSBR& vNb = grid->voxel(v_pos_neighbors[i]);
                        if (vNb.sdf_refined < 0.0)
                        {
                            has_crossing = true;
                            break;
                        }
                    }
                }
            }
            // voxel has zero-crossing (surface) in neighborhood -> keep it
            if (has_crossing)
                continue;

            // voxel is invalid
            invalid_voxels.insert(v_pos);
        }

        // remove invalid voxels
        for (auto iter = invalid_voxels.begin(); iter != invalid_voxels.end(); iter++)
            grid->remove(*iter);
    }

} // namespace SDFAlgorithms
} // namespace nv
