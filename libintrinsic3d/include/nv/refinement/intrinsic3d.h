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
#include <nv/refinement/optimizer.h>
#include <nv/sdf/colorization.h>
#include <nv/sparse_voxel_grid.h>


namespace nv
{
    class Sensor;
	class KeyframeSelection;
    class LightingSVSH;


    /**
     * @brief   Intrinsic3D class for joint refinement of
     *          - geometry and albedo encoded in a Signed Distance Field
     *          - camera poses and camera intrinsics
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class Intrinsic3D
	{
    public:

        /**
         * @brief   Intrinsic3D config struct
         * @author  Robert Maier <robert.maier@tum.de>
         */
        struct Config
        {
            // sdf grid
            int num_grid_levels = 3;
            double thres_shell_factor = 2.0;
            double thres_shell_factor_final = 1.0;
            bool clear_distant_voxels = true;

            // rgbd frame sampling
            int num_rgbd_levels = 3;
            float occlusions_distance = 0.02f;
            size_t num_observations = 5;

            // svsh estimation
            float subvolume_size_sh = 0.2f;
            double sh_est_lambda_reg = 10.0;

            void print() const;
        };


        /**
         * @brief   Struct for storing information about the refinement process
         * @author  Robert Maier <robert.maier@tum.de>
         */
        struct RefinementInfo
        {
            int grid_level;
            int num_grid_levels;
            SparseVoxelGrid<VoxelSBR>* grid;
            int pyramid_level;
            int num_pyramid_levels;
        };


        /**
         * @brief   Callback class for receiving updates during the
         *          refinement process
         * @author  Robert Maier <robert.maier@tum.de>
         */
        class RefinementCallback
        {
        public:
            virtual ~RefinementCallback();
            virtual void onSDFRefined(const RefinementInfo &info) = 0;
        };


        Intrinsic3D(Config cfg,
                    Optimizer::Config opt_cfg,
                    Sensor* sensor,
                    KeyframeSelection* keyframe_selection);
        ~Intrinsic3D();

        const Config& config() const;

        const LightingSVSH* lighting() const;

        bool refine(SparseVoxelGrid<Voxel>* grid);

        void addRefinementCallback(RefinementCallback* cb);

	private:
        void notifyCallbacks();

		bool init();

        bool prepareGridLevel(int grid_lvl_coarsest);
        bool finishGridLevel();

        bool prepareRgbdLevel();
        bool finishRgbdLevel();
		
        bool recomputeColors();

        Config cfg_;
        Optimizer::Config opt_cfg_;
        Sensor* sensor_;
        KeyframeSelection* keyframe_selection_;
        SDFColorization sdf_colorization_;
        LightingSVSH* lighting_;

        std::vector<RefinementCallback*> refine_callbacks_;

        Optimizer::Data opt_data_;
        Optimizer::ImageFormationModel image_model_;
	};

} // namespace nv
