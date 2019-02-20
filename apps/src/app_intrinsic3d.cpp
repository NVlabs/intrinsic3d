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

#include <nv/app_intrinsic3d.h>

#include <iostream>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <nv/filesystem.h>
#include <nv/keyframe_selection.h>
#include <nv/mesh.h>
#include <nv/mesh/marching_cubes.h>
#include <nv/rgbd/sensor.h>
#include <nv/refinement/intrinsic3d.h>
#include <nv/refinement/optimizer.h>
#include <nv/sdf/algorithms.h>
#include <nv/sdf/visualization.h>
#include <nv/lighting/lighting_svsh.h>
#include <nv/sparse_voxel_grid.h>


namespace nv
{
	AppIntrinsic3D::AppIntrinsic3D() :
        sensor_(nullptr),
        keyframe_selection_(nullptr),
        intrinsic3d_(nullptr)
	{
	}


	AppIntrinsic3D::~AppIntrinsic3D()
	{
		delete sensor_;
		delete keyframe_selection_;
		delete intrinsic3d_;
	}
	

	bool AppIntrinsic3D::run(int argc, char *argv[])
	{
        // parse command line arguments
        // command line usage:
        // <app>
        //       -s "<path>/sensor.yml"
        //       -i "<path>/intrinsic3d.yml"
        const char *keys = {
            "{s sensor| |sensor config file}"
            "{i intrinsic3d| |intrinsic3d config file}"
        };
		cv::CommandLineParser cmd(argc, argv, keys);
        const std::string sensor_cfg_file = cmd.get<std::string>("sensor");
        const std::string intrinsic3d_cfg_file = cmd.get<std::string>("intrinsic3d");

        // change working dir to sensor config file directory
        Filesystem::changeWorkingDir(sensor_cfg_file, true);
        // create default output folder
        Filesystem::createFolder("./intrinsic3d");

        // create sensor
        Settings sensor_cfg(sensor_cfg_file);
        sensor_ = Sensor::create(sensor_cfg);
        if (!sensor_)
            return false;
        // print rgbd sensor info
        sensor_->print();

        // load config
        i3d_cfg_.load(intrinsic3d_cfg_file);
        if (i3d_cfg_.empty())
            return false;

		// init keyframe selection
        std::cout << "Loading Keyframes..." << std::endl;
		keyframe_selection_ = new KeyframeSelection();
        if (!keyframe_selection_->load(i3d_cfg_.get<std::string>("keyframes")))
        {
			std::cerr << "Could not load keyframes ..." << std::endl;
        }
        std::cout << keyframe_selection_->countKeyframes() << " keyframes loaded." << std::endl;

		// create voxel grid and load sdf dump
		std::cout << "Loading SDF volume..." << std::endl;
        const std::string input_sdf_filename = i3d_cfg_.get<std::string>("input_sdf");
        SparseVoxelGrid<Voxel>* grid = SparseVoxelGrid<Voxel>::create(input_sdf_filename,
                                                                      sensor_->depthMin(),
                                                                      sensor_->depthMax());
		if (!grid)
		{
			std::cerr << "Could not load voxel grid!" << std::endl;
			return false;
		}
        grid->printInfo();

        // shading based refinement of geometry, albedo and image formation model

        // fill and print out config structs
        Intrinsic3D::Config i3d_cfg;
        Optimizer::Config opt_cfg;
        readConfig(i3d_cfg_, i3d_cfg, opt_cfg);
        i3d_cfg.print();
        opt_cfg.print();

        // create refinement
        intrinsic3d_ = new Intrinsic3D(i3d_cfg, opt_cfg, sensor_, keyframe_selection_);
		// add callback
		intrinsic3d_->addRefinementCallback(this);

		// perform SDF refinement
		if (!intrinsic3d_->refine(grid))
		{
            std::cerr << "Intrinsic3D failed!" << std::endl;
			return false;
		}

		// clean up
		delete grid;

		// close all opencv windows
		cv::destroyAllWindows();

		return true;
	}


    void AppIntrinsic3D::readConfig(const Settings &cfg,
                                    Intrinsic3D::Config &i3d_cfg,
                                    Optimizer::Config &opt_cfg) const
    {
        // number of sdf grid levels
        i3d_cfg.num_grid_levels = cfg.get<int>("num_grid_levels");
        // set thin shell size factor (in voxel sizes)
        i3d_cfg.thres_shell_factor = cfg.get<double>("thin_shell_factor");
        // set final thin shell size factor (in voxel sizes)
        i3d_cfg.thres_shell_factor_final = cfg.get<double>("thin_shell_factor_final");
        // clear voxels far from iso-surface
        i3d_cfg.clear_distant_voxels = cfg.get<bool>("clear_distant_voxels");

        // set RGB-D frame pyramid levels
        i3d_cfg.num_rgbd_levels = cfg.get<int>("num_rgbd_levels");
        // check for occlusions
        i3d_cfg.occlusions_distance = cfg.get<float>("occlusion_distance");
        // number of best observations by weight (0=all observations)
        i3d_cfg.num_observations = cfg.get<size_t>("num_observations");

        // subvolume size for SH coefficients
        i3d_cfg.subvolume_size_sh = cfg.get<float>("subvolume_size_sh");
        // subvolume SH estimation regularizer weight
        i3d_cfg.sh_est_lambda_reg = cfg.get<double>("subvolume_sh_lamda_reg");

        // set number of outer iterations
        opt_cfg.iterations = cfg.get<int>("iterations");
        // set number of Levenberg-Marquardt iterations
        opt_cfg.lm_steps = cfg.get<int>("lm_steps");
        // weight for gradient-based shading cost
        opt_cfg.lambda_g = cfg.get<double>("lambda_g");
        // weight for volumetric regularization term
        opt_cfg.lambda_r0 = cfg.get<double>("lambda_r0");
        opt_cfg.lambda_r1 = cfg.get<double>("lambda_r1");
        // weight for surface stabilization regularization term
        opt_cfg.lambda_s0 = cfg.get<double>("lambda_s0");
        opt_cfg.lambda_s1 = cfg.get<double>("lambda_s1");
        // weight for albedo regularization term (-1.0 for constant albedo)
        opt_cfg.lambda_a = cfg.get<double>("lambda_a");
        // fix parameters
        opt_cfg.fix_poses = cfg.get<bool>("fix_poses");
        opt_cfg.fix_intrinsics = cfg.get<bool>("fix_intrinsics");
        opt_cfg.fix_distortion = cfg.get<bool>("fix_distortion");
    }


    void AppIntrinsic3D::onSDFRefined(const Intrinsic3D::RefinementInfo &info)
	{
        std::string output_postfix = "_g" + std::to_string(info.grid_level) +
                                     "_p" + std::to_string(info.pyramid_level);

		// output mesh prefix
        std::string output_mesh_prefix = i3d_cfg_.get<std::string>("output_mesh_prefix");
        if (!output_mesh_prefix.empty())
		{
			// overwrite sdf values with refined sdf values
            SparseVoxelGrid<VoxelSBR>* grid_vis = info.grid->clone();
            SDFAlgorithms::applyRefinedSdf(grid_vis);

			// output mesh color modes
            std::vector<std::string> output_mesh_color_modes = SDFVisualization::getOutputModes(i3d_cfg_, true);

            // output mesh filename
            output_mesh_prefix = output_mesh_prefix + output_postfix;

			// colorize SDF and export meshes
            SDFVisualization::Config config;
            config.subvolumes = &(intrinsic3d_->lighting()->subvolumes());
            config.subvolume_sh_coeffs = intrinsic3d_->lighting()->shCoeffs();
            config.largest_comp_only = i3d_cfg_.get<bool>("output_mesh_largest_comp_only");
            SDFVisualization sdf_vis(grid_vis, output_mesh_prefix);
            sdf_vis.colorize(output_mesh_color_modes, config);

            delete grid_vis;
		}

		// save refined poses to file
        std::string output_poses_prefix = i3d_cfg_.get<std::string>("output_poses_prefix");
        if (!output_poses_prefix.empty())
		{
			// save poses
            std::string output_poses_file = output_poses_prefix + output_postfix + ".txt";
            std::cout << "Saving camera poses to file " << output_poses_file << std::endl;
            if (!sensor_->savePoses(output_poses_file))
				std::cerr << "Could not save poses..." << std::endl;
		}

		// save refined camera intrinsics to file
        std::string output_intrinsics_prefix = i3d_cfg_.get<std::string>("output_intrinsics_prefix");
        if (!output_intrinsics_prefix.empty())
		{
			// save intrinsics
            std::string output_intrinsics_file = output_intrinsics_prefix + output_postfix + ".txt";
            std::cout << "Saving camera intrinsics to file " << output_intrinsics_file << std::endl;
            if (!sensor_->colorCamera().save(output_intrinsics_file))
				std::cerr << "Could not save color camera intrinsics!" << std::endl;
        }
	}

} // namespace nv


int main(int argc, char *argv[])
{
    // run Intrinsic3D
	nv::AppIntrinsic3D app;
	app.run(argc, argv);
    
    return 0;
}
