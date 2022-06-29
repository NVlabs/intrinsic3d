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

#include <nv/lighting/lighting_svsh.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <nv/refinement/cost.h>

#include <nv/color_util.h>
#include <nv/sdf/operators.h>
#include <nv/sdf/algorithms.h>
#include <nv/shading.h>


namespace nv
{


    LightingSVSH::LightingSVSH(const SparseVoxelGrid<VoxelSBR>* grid, float subvolume_size,
                               double lambda_reg, double thres_shell, bool weighted) :
        grid_(grid),
        subvolume_size_(subvolume_size),
        thres_shell_(thres_shell),
        weighted_(weighted),
        lambda_reg_(lambda_reg),
        subvolumes_(subvolume_size)
	{
	}


    LightingSVSH::~LightingSVSH()
    {
	}


    const Subvolumes& LightingSVSH::subvolumes() const
    {
        return subvolumes_;
    }


    std::vector<Eigen::VectorXd> LightingSVSH::shCoeffs() const
    {
        return sh_coeffs_;
    }


    bool LightingSVSH::interpolate(const Vec3i &v_pos, Eigen::VectorXd &sh_coeffs) const
    {
        if (!grid_ || !grid_->valid(v_pos))
            return false;
        const Vec3f v_coord = grid_->voxelToWorld(v_pos);
        sh_coeffs = subvolumes_.interpolate(sh_coeffs_, v_coord, true);
        return true;
    }


    bool LightingSVSH::computeVoxelShCoeffs(std::vector<Eigen::VectorXd> &voxel_coeffs) const
    {
        if (!grid_)
            return false;

        voxel_coeffs.clear();
        voxel_coeffs.resize(grid_->numVoxels(), Eigen::VectorXd());
        size_t i = 0;
        for (auto iter = grid_->begin(); iter != grid_->end(); iter++, i++)
        {
            const Vec3i& v_pos = iter->first;
            const VoxelSBR& v = iter->second;
            if (!grid_->valid(v_pos) || std::abs(v.sdf_refined) > thres_shell_)
                continue;
            interpolate(v_pos, voxel_coeffs[i]);
        }
        return true;
    }


    class SHDataCost
    {
    public:
        SHDataCost(double luminance, const Vec3f &normal, double albedo) :
            luminance_(luminance),
            albedo_(albedo),
            normal_(normal)
        {
        }

        ~SHDataCost()
        {
        }

        template <typename T>
        bool operator()(const T* const sh_coeffs, T* residuals) const
        {
            // normal
            T normal[3];
            normal[0] = T(normal_[0]);
            normal[1] = T(normal_[1]);
            normal[2] = T(normal_[2]);
            // compute shading
            T shading = T(0.0);
            Shading::computeShading(sh_coeffs, normal, T(albedo_), &shading);
            // compute residual
            residuals[0] = shading - T(luminance_);
            return true;
        }
    private:
        double luminance_;
        double albedo_;
        Vec3f normal_;
    };


    class SHRegularizerCost
    {
    public:
        SHRegularizerCost() {}
        ~SHRegularizerCost() {}

        template <typename T>
        bool operator()(const T* const sh_coeffs0, const T* const sh_coeffs1, T* residuals) const
        {
            // compute residuals
            for (size_t i = 0; i < 9; ++i)
                residuals[i] = sh_coeffs0[i] - sh_coeffs1[i];
            return true;
        }
    };


    bool LightingSVSH::estimate()
    {
        sh_coeffs_.clear();

        if (!grid_ || grid_->empty() || thres_shell_ <= 0.0)
			return false;

        // generate subvolumes for spherical harmonics estimation
        std::cout << "Generate subvolumes ..." << std::endl;
        subvolumes_.compute(grid_);
        std::cout << "number of generated SH subvolumes: " << subvolumes_.count() << std::endl;
        if (subvolumes_.count() == 0)
            return false;

        // estimate sh coefficients for subvolumes
		std::cout << "Estimating local spherical harmonics (joint estimation over all subvolumes) ..." << std::endl;

        const int iterations = 50;
        const double regularizer_weight = lambda_reg_;

		// initialize SH coefficients for subvolumes
        size_t num_subvolumes = subvolumes_.count();
        sh_coeffs_.resize(num_subvolumes);
        for (size_t i = 0; i < num_subvolumes; ++i)
            sh_coeffs_[i] = Eigen::VectorXd::Zero(9);
        std::vector<size_t> voxel_residuals_per_subvolume(num_subvolumes, 0);

		// collect voxels (and their albedos and spherical harmonics basis functions)
        std::cout << "   Collecting residuals for local SH estimation problem ..." << std::endl;
        std::vector<VoxelResidual> voxel_residuals;
        for (auto iter = grid_->begin(); iter != grid_->end(); iter++)
		{
			// get voxel
            const Vec3i& v_pos = iter->first;
			const VoxelSBR& v = iter->second;
            const Vec3f v_coord = grid_->voxelToWorld(v_pos);

			// use only valid voxels
            if (!grid_->valid(v_pos))
				continue;
			// use only voxels within thin shell around surface
            if (std::abs(v.sdf_refined) > thres_shell_)
				continue;
			
			// get per-voxel normal
            Vec3f n = SDFOperators::computeSurfaceNormal(grid_, v_pos);
			if (n.isZero() || std::isnan(n.norm()))
				continue;
			// get albedo
			double albedo = v.albedo;
			if (albedo == 0.0 || std::isnan(albedo))
				continue;

			// compute spherical harmonics basis functions
            Eigen::VectorXd sh_basis_funcs = Shading::shBasisFunctions(n).cast<double>();
            if (sh_basis_funcs.isZero())
				continue;
			// determine subvolume for voxel
            int subvol = subvolumes_.pointToSubvolume(v_coord);
			if (subvol < 0)
				continue;

			// get color and compute intensity
            double luminance = static_cast<double>(intensity(v.color) / 255.0f);

			// compute weight
			double weight = 1.0;
			if (weighted_)
			{
				// give higher weight to voxels close to iso-surface
                weight = SDFOperators::sdfToWeight(v.sdf_refined, grid_->truncation());
				// give less weight to dark areas (with low intensity/luminancy); maybe better use an exponential function
				//weight *= std::max(luminance, 0.001);
			}

			// create cost
			SHDataCost* cost = new SHDataCost(luminance, n, albedo);
            ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SHDataCost, 1, 9>(cost);

			// create voxel residual
			VoxelResidual r;
            r.cost = cost_function;
			r.weight = weight;
            r.params.push_back(sh_coeffs_[subvol].data());
            voxel_residuals.push_back(r);
            voxel_residuals_per_subvolume[subvol]++;
			//std::cout << "lum=" << luminance << ", albedo=" << albedo << ", n=" << n.transpose() << std::endl;
        }
        std::cout << "   " << voxel_residuals.size() << " voxel residuals collected." << std::endl;

        // add regularizer for smooth subvolume SH coeffs
        std::vector<VoxelResidual> regularizer_residuals;
        for (int i = 0; i < num_subvolumes; ++i)
        {
            // retrieve 1-ring neighborhood for subvolume
            Vec3i subvol_idx = subvolumes_.index(i);
            std::vector<Vec3i> neighbors1ring = SDFAlgorithms::collectRingNeighborhood(subvol_idx);

            // collect valid neighbor subvolumes
            std::vector<int> neighbor_volumes;
            for (size_t j = 0; j < neighbors1ring.size(); ++j)
            {
                Vec3i nb_idx = neighbors1ring[j];
                if (!subvolumes_.exists(nb_idx))
                    continue;
                int idx = subvolumes_.indexToSubvolume(nb_idx);
                neighbor_volumes.push_back(idx);
            }

            // add constraint for each neighbor volume
            for (size_t j = 0; j < neighbor_volumes.size(); ++j)
            {
                // create cost
                SHRegularizerCost* cost = new SHRegularizerCost();
                ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<SHRegularizerCost, 9, 9, 9>(cost);
                // create voxel residual
                VoxelResidual r;
                r.cost = cost_function;
                r.weight = 1.0;
                r.params.push_back(sh_coeffs_[i].data());
                r.params.push_back(sh_coeffs_[neighbor_volumes[j]].data());
                regularizer_residuals.push_back(r);
            }
        }
        std::cout << "   " << regularizer_residuals.size() << " regularizer residuals collected." << std::endl;

		// build problem
		std::cout << "   Building local SH estimation problem ..." << std::endl;
		ceres::Problem problem;

        // normalize cost term weight and created scaled loss
		std::cout << "   Adding voxel residuals..." << std::endl;
        double sum_vx_res_weights = 0.0;
        for (size_t i = 0; i < voxel_residuals.size(); ++i)
            sum_vx_res_weights += voxel_residuals[i].weight;
        double data_loss_weight = sum_vx_res_weights > 0.0 ? 1.0 / sum_vx_res_weights : 1.0;
		// add voxel residuals
        for (size_t i = 0; i < voxel_residuals.size(); ++i)
		{
            double w = data_loss_weight * voxel_residuals[i].weight;
            ceres::LossFunction* data_loss = new ceres::ScaledLoss(nullptr, w, ceres::TAKE_OWNERSHIP);
            problem.AddResidualBlock(voxel_residuals[i].cost, data_loss, voxel_residuals[i].params);
		}

        if (!regularizer_residuals.empty())
		{
			std::cout << "   Adding regularizer residuals..." << std::endl;
			// normalize cost term weight and created scaled loss
            double regularizer_loss_weight = regularizer_weight / (double)regularizer_residuals.size();
            ceres::LossFunction* regularizer_loss = new ceres::ScaledLoss(nullptr, regularizer_loss_weight, ceres::TAKE_OWNERSHIP);
			// add residuals for SH coeffs regularizer
            for (size_t i = 0; i < regularizer_residuals.size(); ++i)
                problem.AddResidualBlock(regularizer_residuals[i].cost, regularizer_loss, regularizer_residuals[i].params);
        }

		// solve optimization problem
		std::cout << "   Solving local SH estimation problem ..." << std::endl;

        //solver options
		ceres::Solver::Options options;
		if (iterations > 0)
			options.max_num_iterations = iterations;
		options.minimizer_progress_to_stdout = false;
		// solve linear problem using sparse solver:
		// conjugate gradients solver with jacobi preconditioner on normal equations
		options.linear_solver_type = ceres::CGNR;

		// apply also steps that do not decrease the error
		//options.use_nonmonotonic_steps = true;

		// multi-threading
		options.num_threads = 8;

		// solve problem
		ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
		std::cout << summary.FullReport() << std::endl;
		bool ok = summary.IsSolutionUsable();

		return ok;
	}

} // namespace nv
