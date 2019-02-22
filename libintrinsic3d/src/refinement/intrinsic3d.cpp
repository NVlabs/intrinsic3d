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

#include <nv/refinement/intrinsic3d.h>

#include <iostream>
#include <numeric>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <nv/keyframe_selection.h>
#include <nv/lighting/lighting_svsh.h>
#include <nv/math.h>
#include <nv/rgbd/sensor.h>
#include <nv/rgbd/processing.h>
#include <nv/sdf/algorithms.h>
#include <nv/sdf/colorization.h>


namespace nv
{

    Intrinsic3D::RefinementCallback::~RefinementCallback()
    {
    }


    void Intrinsic3D::Config::load(const Settings& cfg)
    {
        // number of sdf grid levels
        num_grid_levels = cfg.get<int>("num_grid_levels");
        // set thin shell size factor (in voxel sizes)
        thres_shell_factor = cfg.get<double>("thin_shell_factor");
        // set final thin shell size factor (in voxel sizes)
        thres_shell_factor_final = cfg.get<double>("thin_shell_factor_final");
        // clear voxels far from iso-surface
        clear_distant_voxels = cfg.get<bool>("clear_distant_voxels");

        // set RGB-D frame pyramid levels
        num_rgbd_levels = cfg.get<int>("num_rgbd_levels");
        // check for occlusions
        occlusions_distance = cfg.get<float>("occlusion_distance");
        // number of best observations by weight (0=all observations)
        num_observations = cfg.get<size_t>("num_observations");

        // subvolume size for SH coefficients
        subvolume_size_sh = cfg.get<float>("subvolume_size_sh");
        // subvolume SH estimation regularizer weight
        sh_est_lambda_reg = cfg.get<double>("subvolume_sh_lamda_reg");
    }


    void Intrinsic3D::Config::print() const
    {
        std::cout << "Intrinsic3D config:" << std::endl;
        std::cout << "   num_grid_levels: " << num_grid_levels << std::endl;
        std::cout << "   num_rgbd_levels: " << num_rgbd_levels << std::endl;
        std::cout << "   thres_shell_factor: " << thres_shell_factor << std::endl;
        std::cout << "   thres_shell_factor_final: " << thres_shell_factor_final << std::endl;
        std::cout << "   clear_distant_voxels: " << clear_distant_voxels << std::endl;
        std::cout << "   occlusions_distance: " << occlusions_distance << std::endl;
        std::cout << "   num_observations: " << num_observations << std::endl;
        std::cout << "   subvolume_size_sh: " << subvolume_size_sh << std::endl;
        std::cout << "   sh_est_lambda_reg: " << sh_est_lambda_reg << std::endl;
    }


    Intrinsic3D::Intrinsic3D(Config cfg,
                             Optimizer::Config opt_cfg,
                             Sensor* sensor,
                             KeyframeSelection* keyframe_selection) :
        cfg_(cfg),
        opt_cfg_(opt_cfg),
        sensor_(sensor),
        keyframe_selection_(keyframe_selection),
        sdf_colorization_(sensor->colorCamera()),
        lighting_(nullptr)
	{
	}


	Intrinsic3D::~Intrinsic3D()
    {
	}


    const Intrinsic3D::Config& Intrinsic3D::config() const
    {
        return cfg_;
    }


    const LightingSVSH* Intrinsic3D::lighting() const
    {
        return lighting_;
    }


    void Intrinsic3D::addRefinementCallback(RefinementCallback* cb)
    {
        refine_callbacks_.push_back(cb);
    }


    void Intrinsic3D::notifyCallbacks()
    {
        RefinementInfo info;
        info.grid_level = opt_data_.grid_level;
        info.pyramid_level = opt_data_.rgbd_level;
        info.num_grid_levels = cfg_.num_grid_levels;
        info.grid = opt_data_.grid;
        info.num_pyramid_levels = cfg_.num_rgbd_levels;

        for (auto cb : refine_callbacks_)
        {
            cb->onSDFRefined(info);
        }
    }


    bool Intrinsic3D::init()
    {
        if (!sensor_)
            return false;

        // camera intrinsics
        image_model_.intrinsics = sensor_->colorCamera().intrinsicsVec();
        // radial and tangential distortion coefficients
        image_model_.distortion_coeffs.setZero(5);

        // init SDF colorization (for visibility and observation weights)
        sdf_colorization_.reset(opt_data_.grid, sensor_->colorCamera());
        // configure recolorization
        SDFColorization::Config colorizeCfg;
        colorizeCfg.max_occlusion_distance = cfg_.occlusions_distance;
        colorizeCfg.max_num_observations = cfg_.num_observations;
        sdf_colorization_.setConfig(colorizeCfg);

        // store keyframe views
        std::cout << "   convert and store input frames ..." << std::endl;
        image_model_.frame_ids.clear();
        image_model_.poses.clear();
        image_model_.rgbd_pyr.clear();
        for (int i = 0; i < sensor_->numFrames(); ++i)
        {
            if (!keyframe_selection_->isKeyframe(i))
                continue;
            image_model_.frame_ids.push_back(i);

            // resize depth to color image size
            cv::Mat color = sensor_->color(i);
            cv::Mat depth = resizeDepth(sensor_->depthCamera(), sensor_->depth(i), sensor_->colorCamera());
            // create pyramid
            Pyramid frame(cfg_.num_rgbd_levels, color, depth);
            image_model_.rgbd_pyr.push_back(frame);

            // convert pose to vector (angle axis and translation)
            Mat4 pose_cam_to_world = sensor_->pose(i).cast<double>();
            Mat4 pose_world_to_cam = pose_cam_to_world.inverse();
            Vec6 pose_vec = math::poseMatToVecAA(pose_world_to_cam);
            image_model_.poses.push_back(pose_vec);
        }

        // recompute voxel colors
        std::cout << "   initial SDF recolorization ..." << std::endl;
        if (!recomputeColors())
        {
            std::cerr << "   initial SDF recolorization failed!" << std::endl;
            return false;
        }

        return true;
    }


    bool Intrinsic3D::refine(SparseVoxelGrid<Voxel>* grid)
    {
        if (!grid)
            return false;
        if (cfg_.num_grid_levels <= 0 || cfg_.num_rgbd_levels <= 0)
            return false;
		
        std::cout << "Intrinsic3D ..." << std::endl;

        // fill initial grid on coarsest hierarchy level
        std::cout << "   filling initial volume ..." << std::endl;
        opt_data_.grid = SDFAlgorithms::convert(grid);

		// init refinement
		std::cout << "   initializing refinement ..." << std::endl;
		if (!init())
		{
			std::cerr << "   failed to initialize refinement ..." << std::endl;
			return false;
		}

        // double-hierarchical coarse-to-fine approach
        // outer loop over grid pyramid levels
        const int grid_lvl_coarsest = cfg_.num_grid_levels - 1;
        for (int grid_lvl = grid_lvl_coarsest; grid_lvl >= 0; --grid_lvl)
        {
            std::cout << "   refinement on level " << grid_lvl << std::endl;
            std::cout << "      voxel size: " << opt_data_.grid->voxelSize() << std::endl;
            std::cout << "      num voxels: " << opt_data_.grid->numVoxels() << std::endl;

            // preparations for each grid level
            opt_data_.grid_level = grid_lvl;
            prepareGridLevel(grid_lvl_coarsest);

            // inner loop over rgb-d frame pyramid levels
            const int rgbd_lvl_coarsest = cfg_.num_rgbd_levels - 1;
            for (int rgbd_lvl = rgbd_lvl_coarsest; rgbd_lvl >= 0; --rgbd_lvl)
			{
                // use all frame pyramid levels only on coarsest grid level
                if (rgbd_lvl > 0 && grid_lvl < grid_lvl_coarsest)
					continue;
                std::cout << "   level " << grid_lvl << " (pyramid level " << rgbd_lvl << ") ..." << std::endl;
                opt_data_.rgbd_level = rgbd_lvl;

                // preparations for each rgbd level
                prepareRgbdLevel();

                // estimate lighting
                std::cout << "   estimating spherical harmonics on level " << grid_lvl << " ..." << std::endl;
                lighting_ = new LightingSVSH(opt_data_.grid, cfg_.subvolume_size_sh,
                                             cfg_.sh_est_lambda_reg, opt_data_.thres_shell, true);
                if (!lighting_->estimate())
                {
                    std::cerr << "   lighting estimation on level " << grid_lvl << " not successful!" << std::endl;
                    break;
                }
                // compute interpolated SH coefficients in advance for all voxels
                std::cout << "   compute interpolated SH coefficients for voxels ..." << std::endl;
                if (!lighting_->computeVoxelShCoeffs(opt_data_.voxel_sh_coeffs))
                {
                    std::cerr << "   computing interpolated voxel SH coefficients failed!" << std::endl;
                    return false;
                }

                // build and solve optimization problem
                std::cout << "Optimization (level " << grid_lvl << ", pyramid " << rgbd_lvl <<  ")..." << std::endl;
                Optimizer optimizer(opt_cfg_);
                if (!optimizer.optimize(sdf_colorization_, opt_data_, image_model_))
                {
                    std::cerr << "   optimization failed!" << std::endl;
                }

                // finish rgbd level
                finishRgbdLevel();

				// notify callbacks
                notifyCallbacks();

                delete lighting_;
                opt_data_.shading_cost_data.clear();
			}

            // finish grid level
            finishGridLevel();
        }

        delete opt_data_.grid;

        return true;
    }


    bool Intrinsic3D::prepareGridLevel(int grid_lvl_coarsest)
    {
        // thin shell threshold
        double thres_shell_factor = cfg_.thres_shell_factor;
        if (cfg_.thres_shell_factor_final > 0.0)
            thres_shell_factor = computeVaryingLambda(grid_lvl_coarsest - opt_data_.grid_level, cfg_.num_grid_levels,
                                                      cfg_.thres_shell_factor, cfg_.thres_shell_factor_final);
        std::cout << "      thres shell factor: " << thres_shell_factor << std::endl;
        opt_data_.thres_shell = thres_shell_factor * static_cast<double>(opt_data_.grid->voxelSize());

        if (cfg_.clear_distant_voxels)
        {
            // clear voxels far from iso-surface
            std::cout << "   remove voxels far from iso-surface ..." << std::endl;
            SDFAlgorithms::clearVoxelsOutsideThinShell(opt_data_.grid, opt_data_.thres_shell);
            std::cout << "      num voxels (sparsified): " << opt_data_.grid->numVoxels() << std::endl;
        }

        return true;
    }


    bool Intrinsic3D::finishGridLevel()
    {
        if (opt_data_.grid_level > 0)
        {
            // upsample volume for next level
            std::cout << "   upsampling grid for next level ..."<< std::endl;
            SparseVoxelGrid<VoxelSBR>* gridLvlUp = SDFAlgorithms::upsample<VoxelSBR>(opt_data_.grid);
            delete opt_data_.grid;
            // use upsampled volume for next level
            opt_data_.grid = gridLvlUp;
        }

        return true;
    }


    bool Intrinsic3D::prepareRgbdLevel()
    {
        // collect shading cost data for current rgb-d pyramid level
        for (size_t i = 0; i < image_model_.rgbd_pyr.size(); ++i)
        {
            cv::Mat lum = image_model_.rgbd_pyr[i].intensity(opt_data_.rgbd_level);
            const float* ptr_intensity = reinterpret_cast<const float*>(lum.data);
            ShadingCostData data(opt_data_.rgbd_level,
                                 static_cast<double>(opt_data_.grid->voxelSize()),
                                 lum.cols, lum.rows, ptr_intensity);
            opt_data_.shading_cost_data.push_back(data);
        }

        return true;
    }


    bool Intrinsic3D::finishRgbdLevel()
	{
        // recompute voxel colors
        std::cout << "   SDF recolorization ..." << std::endl;
        if (!recomputeColors())
        {
            std::cerr << "   SDF recolorization failed!" << std::endl;
            return false;
        }

		// update sensor poses with refined poses
        for (size_t i = 0; i < image_model_.frame_ids.size(); ++i)
        {
            // convert pose vector (angle axis and translation) to transformation matrix
            Mat4 pose_world_to_cam = math::poseVecAAToMat(image_model_.poses[i]);
            Mat4 pose_cam_to_world = pose_world_to_cam.inverse();
            sensor_->setPose(image_model_.frame_ids[i], pose_cam_to_world.cast<float>());
        }
        // update and output refined camera intrinsics
        sensor_->colorCamera().setIntrinsics(image_model_.intrinsics);
        sensor_->colorCamera().setDistortion(image_model_.distortion_coeffs.cast<float>());
        std::cout << "Updated camera model parameters:" << std::endl;
        sensor_->colorCamera().print();

		return true;
	}


	bool Intrinsic3D::recomputeColors()
	{
        if (!opt_data_.grid)
			return false;

		// recompute voxel colors
        sdf_colorization_.reset(opt_data_.grid, image_model_.intrinsics, image_model_.distortion_coeffs);

		// add views for collecting observations
        std::cout << "   collect observations ..." << std::endl;
        for (size_t i = 0; i < image_model_.poses.size(); ++i)
		{
			std::cout << "      adding frame " << i << "... " << std::endl;
			// add observations
            Mat4f pose_world_to_cam = math::poseVecAAToMat(image_model_.poses[i]).cast<float>();
            sdf_colorization_.add(static_cast<int>(i),
                                  image_model_.rgbd_pyr[i].depth(0),
                                  image_model_.rgbd_pyr[i].color(0),
                                  pose_world_to_cam);
		}

		// recompute SDF colors
        std::cout << "   recompute SDF colors from observations ..." << std::endl;
        sdf_colorization_.compute();

		return true;
	}

} // namespace nv
