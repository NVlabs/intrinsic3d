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

#include <nv/refinement/optimizer.h>

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <nv/refinement/albedo_regularizer.h>
#include <nv/refinement/cost.h>
#include <nv/refinement/nls_solver.h>
#include <nv/refinement/shading_cost.h>
#include <nv/refinement/surface_stab_regularizer.h>
#include <nv/refinement/volumetric_regularizer.h>
#include <nv/sdf/algorithms.h>


namespace nv
{

    void Optimizer::Config::load(const Settings &cfg)
    {
        // set number of outer iterations
        iterations = cfg.get<int>("iterations");
        // set number of Levenberg-Marquardt iterations
        lm_steps = cfg.get<int>("lm_steps");
        // weight for gradient-based shading cost
        lambda_g = cfg.get<double>("lambda_g");
        // weight for volumetric regularization term
        lambda_r0 = cfg.get<double>("lambda_r0");
        lambda_r1 = cfg.get<double>("lambda_r1");
        // weight for surface stabilization regularization term
        lambda_s0 = cfg.get<double>("lambda_s0");
        lambda_s1 = cfg.get<double>("lambda_s1");
        // weight for albedo regularization term (-1.0 for constant albedo)
        lambda_a = cfg.get<double>("lambda_a");
        // fix parameters
        fix_poses = cfg.get<bool>("fix_poses");
        fix_intrinsics = cfg.get<bool>("fix_intrinsics");
        fix_distortion = cfg.get<bool>("fix_distortion");
    }


    void Optimizer::Config::print() const
    {
        std::cout << "Optimizer config:" << std::endl;
        std::cout << "   iterations: " << iterations << std::endl;
        std::cout << "   lm_steps: " << lm_steps << std::endl;
        std::cout << "   lambda_g: " << lambda_g << std::endl;
        std::cout << "   lambda_r0: " << lambda_r0 << std::endl;
        std::cout << "   lambda_r1: " << lambda_r1 << std::endl;
        std::cout << "   lambda_s0: " << lambda_s0 << std::endl;
        std::cout << "   lambda_s1: " << lambda_s1 << std::endl;
        std::cout << "   lambda_a: " << lambda_a << std::endl;
        std::cout << "   fix_poses: " << fix_poses << std::endl;
        std::cout << "   fix_intrinsics: " << fix_intrinsics << std::endl;
        std::cout << "   fix_distortion: " << fix_distortion << std::endl;
    }


    Optimizer::Optimizer(Config cfg) :
        cfg_(cfg)
	{
	}


    Optimizer::~Optimizer()
    {
	}


    const Optimizer::Config& Optimizer::config() const
    {
        return cfg_;
    }


    bool Optimizer::optimize(SDFColorization &colorization,
                             Data &data,
                             ImageFormationModel &image_formation)
	{
        if (!data.grid || cfg_.iterations < 1)
			return false;

        int grid_lvl = data.grid_level;
        int rgbd_lvl = data.rgbd_level;

        for (int itr = 0; itr < cfg_.iterations; ++itr)
		{
            std::cout << "   iteration " << itr << " (grid level " << grid_lvl << ", pyramid level " << rgbd_lvl << ")" << std::endl;

            // update voxel colorization
            Vec4 intrinsics_scaled = image_formation.intrinsics * pyramidLevelToScale(rgbd_lvl);
            colorization.reset(data.grid, intrinsics_scaled, image_formation.distortion_coeffs);

			// compute cost term weights for current iteration
			// volumetric regularizer
            double lambda_r = computeVaryingLambda(itr, cfg_.iterations, cfg_.lambda_r0, cfg_.lambda_r1);
			// surface stabilization regularizer
            double lambda_s = computeVaryingLambda(itr, cfg_.iterations, cfg_.lambda_s0, cfg_.lambda_s1);
			// albedo regularizer
            double lambda_a = cfg_.lambda_a;

            // create NLS solver
            NLSSolver solver;
            solver.reset(4);
            solver.setCostWeight(0, cfg_.lambda_g);
            solver.setCostWeight(1, lambda_r);
            solver.setCostWeight(2, lambda_s);
            solver.setCostWeight(3, lambda_a);

			// add residual blocks
			std::cout << "      collecting residuals ..." << std::endl;
            size_t num_valid_voxels = 0;
            size_t voxel_idx = 0;
            for (auto iter = data.grid->begin(); iter != data.grid->end(); iter++, voxel_idx++)
			{
				// get voxel
                const Vec3i& v_pos = iter->first;
                // add residuals for voxels
                if (addVoxelResiduals(solver, colorization, data, image_formation, v_pos, voxel_idx, rgbd_lvl))
                    ++num_valid_voxels;
			}
            data.voxels_added.clear();
            std::cout << "      voxels: " << data.grid->numVoxels() << " (valid " << num_valid_voxels << ")" << std::endl;

            if (num_valid_voxels > 0)
			{
                // build problem and fix parameters
                std::cout << "      building problem ..." << std::endl;
                buildProblem(solver, data, image_formation);

				// solve problem
				std::cout << "      solving problem ..." << std::endl;
                solver.solve(cfg_.lm_steps);
			}
        }

		return true;
	}


    bool Optimizer::addVoxelResiduals(NLSSolver &solver,
                                      SDFColorization &colorization,
                                      Data &data,
                                      ImageFormationModel &image_formation,
                                      const Vec3i &v_pos, size_t voxel_idx, int pyr_lvl)
    {
        // use only observed/valid voxels
        if (!data.grid->valid(v_pos))
            return false;

        VoxelSBR& v = data.grid->voxel(v_pos);
        // use only voxels within thin shell around surface
        if (std::abs(v.sdf_refined) > data.thres_shell)
            return false;
        // use only voxels with valid surface normal
        Vec3f n = SDFOperators::computeSurfaceNormal(data.grid, v_pos);
        if (n.isZero())
            return false;

        // collect residuals/observations for voxels
        // add residuals for Eg, Er, Es and Ea

		// give higher weight to voxels close to current iso-surface
        double weight_sdf = SDFOperators::sdfToWeight(v.sdf_refined, data.grid->truncation());

        // create gradient-based shading cost Eg (data term)
        // collect voxel observations
        std::vector<VertexObservation> observations;
        colorization.collectObservations(image_formation.poses, image_formation.rgbd_pyr,
                                         v_pos, n, pyr_lvl,
                                         observations);

        // reference SH coeffs for voxel
        const Eigen::VectorXd& sh_coeffs = data.voxel_sh_coeffs[voxel_idx];

        // collect residuals for observations
        std::vector<VoxelResidual> voxel_residuals;
        double sum_obs_weights = 0.0;
        for (size_t i = 0; i < observations.size(); ++i)
        {
            VertexObservation obs = observations[i];
            if (obs.weight <= 0.0f)
                continue;
            size_t f_id = static_cast<size_t>(obs.frame);

            // add only residuals which are valid and have not been filtered out
            VoxelResidual rg = ShadingCost::create(data.grid, v_pos, image_formation.poses[f_id],
                                                   image_formation.intrinsics, image_formation.distortion_coeffs,
                                                   sh_coeffs, &(data.shading_cost_data[f_id]));
            if (rg.cost)
            {
                rg.weight = static_cast<double>(obs.weight);
                sum_obs_weights += rg.weight;
                voxel_residuals.push_back(rg);
            }
        }

        for (size_t i = 0; i < voxel_residuals.size(); ++i)
		{
            voxel_residuals[i].weight *= weight_sdf;
            solver.addResidual(0, voxel_residuals[i]);
		}

        if (cfg_.lambda_r0 > 0.0 && cfg_.lambda_r1 > 0.0)
		{
            // create volumetric regularizer cost Er
            VoxelResidual rv = VolumetricRegularizer::create(data.grid, v_pos);
			if (rv.weight > 0.0)
			{
				solver.addResidual(1, rv);
			}
		}

        if (cfg_.lambda_s0 > 0.0 && cfg_.lambda_s1 > 0.0)
		{
			// create surface stabilization cost Es
            VoxelResidual rs = SurfaceStabRegularizer::create(data.grid, v_pos);
			if (rs.weight > 0.0)
			{
				solver.addResidual(2, rs);
			}
		}

        if (cfg_.lambda_a > 0.0)
        {
            // create albedo regularizer cost Ea

            // collect 1-ring neighborhood voxels
            std::vector<Vec3i> v_pos_neighbors = SDFAlgorithms::collectRingNeighborhood(v_pos);
            if (SDFAlgorithms::checkVoxelsValid(data.grid, v_pos_neighbors))
            {
                // add albedo regularizer cost Ea for all 1-ring neighborhood voxels
                for (size_t nb = 0; nb < 6; ++nb)
                {
                    if (data.voxels_added.find(v_pos_neighbors[nb]) != data.voxels_added.end())
						continue;
                    VoxelResidual ra = AlbedoRegularizer::create(data.grid, v_pos, v_pos_neighbors[nb]);
					solver.addResidual(3, ra);
                }
            }
        }

		// keep track of added voxels to add as few albedo regularizers as possible
        data.voxels_added.insert(v_pos);

        return true;
    }


    bool Optimizer::buildProblem(NLSSolver &solver,
                                 Data &data,
                                 ImageFormationModel &image_formation)
    {
        // build problem
        solver.buildProblem(true);

        // fix voxel parameters
        fixVoxelParams(solver, data, true);

        // fix camera poses parameters
        if (cfg_.fix_poses)
        {
            for (size_t i = 0; i < image_formation.poses.size(); ++i)
                solver.fixParamBlock(image_formation.poses[i].data());
        }
        // fix camera intrinsics parameters
        if (cfg_.fix_intrinsics)
            solver.fixParamBlock(image_formation.intrinsics.data());
        // fix radial and tangential distortion parameters
        if (cfg_.fix_distortion)
            solver.fixParamBlock(image_formation.distortion_coeffs.data());

        return true;
    }


    void Optimizer::fixVoxelParams(NLSSolver &solver, Data &data, bool fix_ring_neighborhood)
    {
        // fix voxel parameters
        for (auto iter = data.grid->begin(); iter != data.grid->end(); iter++)
        {
            const Vec3i& v_pos = iter->first;
            VoxelSBR& v = iter->second;

            bool fix_sdf = false;
            bool fix_albedo = false;

            // fix parameters for voxels that are not within thin shell
            if (!data.grid->valid(v_pos) || std::abs(v.sdf_refined) > data.thres_shell)
            {
                fix_sdf = true;
                fix_albedo = true;
            }

            if (cfg_.lambda_a < 0.0)
            {
                // set albedo constant for objects with uniform albedo
                fix_albedo = true;
            }

            if (fix_ring_neighborhood)
            {
                // fix voxel parameters if 1-ring-neighborhood not valid
                std::vector<Vec3i> v_pos_neighbors = SDFAlgorithms::collectRingNeighborhood(v_pos);
                // use only voxels with valid 1-ring-neighborhood
                if (!SDFAlgorithms::checkVoxelsValid(data.grid, v_pos_neighbors))
                {
                    fix_sdf = true;
                    fix_albedo = true;
                }
            }

            // fix parameters for voxels
            if (fix_sdf)
            {
                double* ptr_sdf_refined = &(v.sdf_refined);
                solver.fixParamBlock(ptr_sdf_refined);
            }

            if (fix_albedo)
            {
                double* ptr_albedo = &(v.albedo);
                solver.fixParamBlock(ptr_albedo);
            }
        }
    }

} // namespace nv
