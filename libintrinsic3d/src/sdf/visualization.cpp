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

#include <nv/sdf/visualization.h>

#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2/imgproc.hpp>

#include <nv/color_util.h>
#include <nv/camera.h>
#include <nv/keyframe_selection.h>
#include <nv/mesh/marching_cubes.h>
#include <nv/math.h>
#include <nv/mesh.h>
#include <nv/mesh/util.h>
#include <nv/rgbd/processing.h>
#include <nv/rgbd/pyramid.h>
#include <nv/rgbd/sensor.h>
#include <nv/sdf/colorization.h>
#include <nv/lighting/lighting_svsh.h>
#include <nv/lighting/subvolumes.h>
#include <nv/sdf/algorithms.h>
#include <nv/sdf/operators.h>
#include <nv/shading.h>


namespace nv
{

    SDFVisualization::SDFVisualization(SparseVoxelGrid<VoxelSBR>* grid, const std::string &output_mesh_prefix) :
        grid_(grid),
        output_mesh_prefix_(output_mesh_prefix)
	{
	}


	SDFVisualization::~SDFVisualization()
	{
	}


    std::vector<std::string> SDFVisualization::getOutputModes(Settings &settings, bool add_voxel_colors)
    {
        // output mesh color modes
        std::vector<std::string> color_modes;
        if (add_voxel_colors)
            color_modes.push_back("");		// voxel colors
        addOutputMode(settings, "output_mesh_normals", "normals", color_modes);
        addOutputMode(settings, "output_mesh_laplacian", "lap", color_modes);
        addOutputMode(settings, "output_mesh_intensity", "lum", color_modes);
        addOutputMode(settings, "output_mesh_intensity_grad", "lum_grad", color_modes);
        addOutputMode(settings, "output_mesh_albedo", "albedo", color_modes);
        addOutputMode(settings, "output_mesh_shading_sv", "shading_sv", color_modes);
        addOutputMode(settings, "output_mesh_shading_sv_const", "shading_sv_const", color_modes);
        addOutputMode(settings, "output_mesh_chromacity", "chroma", color_modes);
        addOutputMode(settings, "output_mesh_subvolumes", "subvol", color_modes);
        addOutputMode(settings, "output_mesh_subvolumes_interpolated", "subvol_interp", color_modes);
        return color_modes;
    }


    void SDFVisualization::addOutputMode(Settings &settings, const std::string &param, const std::string &color_mode, std::vector<std::string> &color_modes)
	{
		if (!settings.exists(param))
			return;
		if (settings.get<bool>(param))
            color_modes.push_back(color_mode);
	}


    bool SDFVisualization::colorize(const std::vector<std::string> &color_modes, const SDFVisualization::Config &cfg)
    {
        if (!grid_ || output_mesh_prefix_.empty())
            return false;

        for (size_t i = 0; i < color_modes.size(); ++i)
        {
            std::string color_mode = color_modes[i];
            std::cout << "SDF visualization and export: " << color_mode << std::endl;

            // store grid colors
            storeColors();

            // adjust voxel colors
            if (color_mode == "normals")
            {
                applyColorNormals();
            }
            else if (color_mode == "lap")
            {
                applyColorLaplacian();
            }
            else if (color_mode == "lum")
            {
                applyColorIntensity();
            }
            else if (color_mode == "lum_grad")
            {
                applyColorIntensityGradient();
            }
            else if (color_mode == "albedo")
            {
                applyColorAlbedo();
            }
            else if (color_mode == "shading_sv")
            {
				applyColorShading(cfg.subvolumes, cfg.subvolume_sh_coeffs, false);
            }
            else if (color_mode == "shading_sv_const")
            {
				applyColorShading(cfg.subvolumes, cfg.subvolume_sh_coeffs, true);
            }
            else if (color_mode == "chroma")
            {
                applyColorChromacity();
            }
            else if (color_mode == "subvol")
            {
                applyColorSubvolumes(cfg.subvolumes);
            }
            else if (color_mode == "subvol_interp")
            {
                applyColorSubvolumesInterp(cfg.subvolumes);
            }

            // generate and export mesh
            exportMesh(output_mesh_prefix_, color_mode, cfg.largest_comp_only);

            // reset grid colors
            restoreColors();
        }

        return false;
    }


    bool SDFVisualization::exportMesh(const std::string &output_mesh_prefix, const std::string &color_mode, bool largest_comp_only)
    {
        if (output_mesh_prefix.empty())
            return false;

        // generate mesh
        Mesh* mesh = MarchingCubes<VoxelSBR>::extractSurface(*grid_);
        if (!mesh)
        {
            std::cerr << "Mesh could not be generated!" << std::endl;
            return false;
        }

        if (largest_comp_only)
		{
			// remove all connected components except of largest one
            MeshUtil::removeLooseComponents(mesh);
		}

        // save mesh
        std::string mesh_file = output_mesh_prefix;
        if (!color_mode.empty())
            mesh_file += "_" + color_mode;
        mesh_file += ".ply";
        bool ok = mesh->save(mesh_file);
        if (!ok)
            std::cerr << "Mesh could not be saved!" << std::endl;

        delete mesh;
        return ok;
    }


    bool SDFVisualization::storeColors()
	{
        if (!grid_)
			return false;

		// store voxel colors
		colors_.clear();
        colors_.resize(grid_->numVoxels());
		size_t i = 0;
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++, i++)
			colors_[i] = (itr->second).color;
		return true;
	}


    bool SDFVisualization::restoreColors()
	{
        if (!grid_ || colors_.size() != grid_->numVoxels())
			return false;

		// assign stored colors back to voxels
		size_t i = 0;
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++, i++)
			(itr->second).color = colors_[i];
		return true;
	}


    void SDFVisualization::applyColorNormals()
	{
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
			// compute normal
			Vec3f c = Vec3f::Zero();
            //Vec3f n = grid_->surfaceNormal(itr->first);
            Vec3f n = SDFOperators::computeSurfaceNormal(grid_, itr->first);
			if (n.norm() != 0.0f && !std::isnan(n.norm()))
				c = 0.5f * n + Vec3f::Constant(0.5f);
			(itr->second).color = (c * 255.0f).cast<unsigned char>();
		}
	}


    void SDFVisualization::applyColorLaplacian()
	{
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
            const Vec3i& v_pos = itr->first;
            // collect 1-ring neighborhood voxels
            std::vector<Vec3i> v_pos_neighbors = SDFAlgorithms::collectRingNeighborhood(v_pos);
            // use only voxels with valid 1-ring-neighborhood
            float lap = 0.0f;
            if (SDFAlgorithms::checkVoxelsValid(grid_, v_pos_neighbors))
            {
                // compute Laplacian
                lap = 0.5f * SDFOperators::laplacian(grid_, v_pos) + 0.5f;
            }
            (itr->second).color = scalarToColor(lap, 255.0f);
		}
	}


    void SDFVisualization::applyColorIntensity()
	{
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
        {
            VoxelSBR& v = itr->second;
            const float lum = intensity(v.color);
            v.color = scalarToColor(lum, 1.0f);
		}
	}


    void SDFVisualization::applyColorIntensityGradient()
	{
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
            const Vec3i& v_pos = itr->first;
            VoxelSBR& voxel = itr->second;

            // collect 1-ring neighborhood voxels
            std::vector<Vec3i> v_pos_neighbors = SDFAlgorithms::collectRingNeighborhood(v_pos);
            // use only voxels with valid 1-ring-neighborhood
            Vec3f lum_gradient;
            if (SDFAlgorithms::checkVoxelsValid(grid_, v_pos_neighbors))
            {
                // compute intensity gradient
                lum_gradient = SDFOperators::intensityGradient(grid_, v_pos);
            }
            else
            {
                lum_gradient = Vec3f::Zero();
            }

            // scale for visualization
			// only gradient in x direction
            Vec3f c = Vec3f::Constant(lum_gradient[0] * 0.5f + 127.0f);
			// gradient in all three directions
            //Vec3f c = (lum_gradient * 0.5f) + Vec3f::Constant(127.0f);
			// gradient norm
            //Vec3f c = Vec3f::Ones() * lum_gradient.norm();

			// normalize
			c = checkRange(c, 0.0f, 255.0f);
			voxel.color = c.cast<unsigned char>();
		}
	}


    void SDFVisualization::applyColorAlbedo()
	{
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
        {
            (itr->second).color = scalarToColor((itr->second).albedo, 255.0);
		}
	}


    void SDFVisualization::applyColorShading(const Subvolumes* subvolumes,
                                             const std::vector<Eigen::VectorXd> &sh_coeffs,
                                             bool constant_albedo)
	{
		if (!subvolumes)
			return;
        if (subvolumes->count() != (int)sh_coeffs.size())
		{
			std::cerr << "Number of subvolumes and SH coeffs not equal!"  << std::endl;
			return;
		}
		//std::cout << "num subvolumes: " << subvolumes->numSubvolumes() << std::endl;

        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
            const Vec3i& v_pos = itr->first;
			VoxelSBR& v = itr->second;
            const Vec3f v_coord = grid_->voxelToWorld(v_pos);
            Vec3f n = SDFOperators::computeSurfaceNormal(grid_, itr->first);
			if (n.norm() == 0.0f || std::isnan(n.norm()))
			{
				v.color = Vec3b::Zero();
				continue;
			}
			
			// compute interpolated Spherical Harmonics for voxel
            Eigen::VectorXf avg_sh = Eigen::VectorXf::Zero(9);
            if (sh_coeffs.size() == 1)
                avg_sh = sh_coeffs[0].cast<float>();
			else
                avg_sh = subvolumes->interpolate(sh_coeffs, v_coord, true).cast<float>();

			// compute voxel shading
            float albedo = constant_albedo ? 0.7f : (float)v.albedo;
            double shad = (double)Shading::computeShading(n, avg_sh, albedo) * 255.0f;

            // assign shading to voxel color
            v.color = scalarToColor((float)shad, 1.0f);
		}
	}


    void SDFVisualization::applyColorChromacity()
	{
        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
			VoxelSBR& voxel = itr->second;
			Vec3f chrom = chromacity(voxel.color);
			// scale chromacity for visualization
			chrom = chrom * 255.0f * 0.5f;
			chrom = checkRange(chrom, 0.0f, 255.0f);
			voxel.color = chrom.cast<unsigned char>();
		}
	}


    void SDFVisualization::applyColorSubvolumes(const Subvolumes* subvolumes)
	{
        if (!subvolumes || subvolumes->count() <= 1)
			return;

        for (int subvol = 0; subvol < subvolumes->count(); ++subvol)
		{
			const Vec6i bounds = subvolumes->bounds(subvol);
			Vec3b color = subvolumes->color(subvol);

			// colorize subvolume voxels
            for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
			{
                const Vec3i& v_pos = itr->first;
                if (!math::withinBounds(bounds, v_pos))
					continue;
				itr->second.color = color;
			}
		}
	}


    void SDFVisualization::applyColorSubvolumesInterp(const Subvolumes* subvolumes)
	{
        if (!subvolumes || subvolumes->count() <= 1)
			return;

		// retrieve subvolume colors
        std::vector<Vec3f> subvol_colors;
        for (int i = 0; i < subvolumes->count(); ++i)
            subvol_colors.push_back(subvolumes->color(i).cast<float>());

        for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
            const Vec3i& v_pos = itr->first;
			VoxelSBR& v = itr->second;
            const Vec3f v_coord = grid_->voxelToWorld(v_pos);
			// interpolate subvolume color
            Vec3f avg_color = subvolumes->interpolate(subvol_colors, v_coord, true);
            avg_color = checkRange(avg_color, 0.0f, 255.0f);
            v.color = avg_color.cast<unsigned char>();
		}
	}


} // namespace nv
