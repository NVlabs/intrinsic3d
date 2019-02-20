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
#include <nv/settings.h>
#include <nv/sparse_voxel_grid.h>


namespace nv
{
	class KeyframeSelection;
	class Sensor;
	class Subvolumes;


    /**
     * @brief   Class for re-calculating the SDF colors based on desired
     *          function, which makes it possible to directly output hidden
     *          SDF properties (like voxel albedo or forward-shading) as
     *          mesh color
     * @author  Robert Maier <robert.maier@tum.de>
     */
	class SDFVisualization
	{
	public:

        /**
         * @brief   SDF visualization config struct
         * @author  Robert Maier <robert.maier@tum.de>
         */
        struct Config
        {
            const Subvolumes* subvolumes = nullptr;
            std::vector<Eigen::VectorXd> subvolume_sh_coeffs;
            bool largest_comp_only = false;
        };

        SDFVisualization(SparseVoxelGrid<VoxelSBR>* grid, const std::string &output_mesh_prefix);
		~SDFVisualization();

        static std::vector<std::string> getOutputModes(Settings &settings, bool add_voxel_colors = true);

        bool colorize(const std::vector<std::string> &color_modes, const SDFVisualization::Config &cfg);

	private:
        static void addOutputMode(Settings &settings, const std::string &param, const std::string &color_mode, std::vector<std::string> &color_modes);
        bool exportMesh(const std::string &output_mesh_prefix, const std::string &color_mode, bool largest_comp_only = false);

        bool storeColors();
        bool restoreColors();

        void applyColorNormals();
        void applyColorLaplacian();
        void applyColorIntensity();
        void applyColorIntensityGradient();
        void applyColorAlbedo();
        void applyColorShading(const Subvolumes* subvolumes,
                               const std::vector<Eigen::VectorXd> &sh_coeffs,
                               bool constant_albedo = false);
        void applyColorChromacity();
        void applyColorSubvolumes(const Subvolumes* subvolumes);
        void applyColorSubvolumesInterp(const Subvolumes* subvolumes);

        SparseVoxelGrid<VoxelSBR>* grid_;
        std::string output_mesh_prefix_;
		std::vector<Vec3b> colors_;

	};

} // namespace nv
