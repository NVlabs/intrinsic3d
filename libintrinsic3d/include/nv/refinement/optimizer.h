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
#include <string>
#include <unordered_set>
#include <vector>
#include <opencv2/core.hpp>
#include <nv/refinement/nls_solver.h>
#include <nv/rgbd/pyramid.h>
#include <nv/refinement/shading_cost.h>
#include <nv/refinement/cost.h>
#include <nv/sdf/colorization.h>
#include <nv/sparse_voxel_grid.h>



namespace nv
{
    class NLSSolver;


    /**
     * @brief   Class for joint optimization
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class Optimizer
	{
    public:

        /**
         * @brief   Optimizer config struct for optimization and solver parameters
         * @author  Robert Maier <robert.maier@tum.de>
         */
        struct Config
        {
            int iterations = 10;
            int lm_steps = 50;
            double lambda_g = 0.2;
            double lambda_r0 = 20.0;
            double lambda_r1 = 160.0;
            double lambda_s0 = 10.0;
            double lambda_s1 = 120.0;
            double lambda_a = 0.1;
            bool fix_poses = false;
            bool fix_intrinsics = false;
            bool fix_distortion = false;

            void print() const;
        };


        /**
         * @brief   Struct for storing optimizer data
         * @author  Robert Maier <robert.maier@tum.de>
         */
        struct Data
        {
            SparseVoxelGrid<VoxelSBR>* grid = nullptr;
            double thres_shell = 0.0;
            int grid_level = 0;
            int rgbd_level = 0;
            std::vector<Eigen::VectorXd> voxel_sh_coeffs;
            std::vector<ShadingCostData> shading_cost_data;
            std::unordered_set<Vec3i, std::hash<Vec3i> > voxels_added;
        };


        /**
         * @brief   Struct for storing image formation model
         * @author  Robert Maier <robert.maier@tum.de>
         */
        struct ImageFormationModel
        {
            Vec4 intrinsics = Vec4::Zero();
            Vec5 distortion_coeffs = Vec5::Zero(5);

            std::vector<int> frame_ids;
            std::vector<Vec6> poses;
            std::vector<Pyramid> rgbd_pyr;
        };


        Optimizer(Config cfg);
        ~Optimizer();

        const Config& config() const;

        bool optimize(SDFColorization &colorization,
                      Data &data,
                      ImageFormationModel &image_formation);

	private:
        bool addVoxelResiduals(NLSSolver &solver,
                               SDFColorization &colorization,
                               Data &data,
                               ImageFormationModel &image_formation,
                               const Vec3i &v_pos, size_t voxel_idx, int pyr_lvl);

        bool buildProblem(NLSSolver &solver,
                          Data &data,
                          ImageFormationModel &image_formation);

        void fixVoxelParams(NLSSolver &solver, Data &data, bool fix_ring_neighborhood = true);

        Config cfg_;
	};

} // namespace nv
