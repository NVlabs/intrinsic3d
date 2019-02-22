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

#include <nv/refinement/nls_solver.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>

#include <ceres/ceres.h>


namespace nv
{

	NLSSolver::ProblemInfo::ProblemInfo() :
		iteration(0),
		residuals(0),
		parameters(0),
		cost(0.0),
		residual_types(0),
		time_add(0.0),
        time_build(0.0)
	{
	}


    std::string NLSSolver::ProblemInfo::toString(bool print_costs) const
	{
		std::stringstream ss;
        ss <<   "iteration " << iteration << " problem: " <<
                residuals << " residuals, " <<
                parameters << " parameters, ";
        if (print_costs)
            ss << "cost " << cost << ", ";

        ss <<   "time_add " << time_add << ", " <<
                "time_build " << time_build;

        if (print_costs)
		{
			for (size_t i = 0; i < type_residuals.size(); ++i)
			{
				ss << std::endl << "      cost type " << i << ": " <<
					type_residuals[i] << " residuals, " <<
					"cost " << type_costs[i] << ", " <<
					"weight " << type_weights[i];
			}
		}
		return ss.str();
	}


	NLSSolver::SolverInfo::SolverInfo() :
		cost_final(0.0),
		cost_change(0.0),
		inner_iterations(0),
		trust_region_radius(0.0),
        time_solve(0.0)
	{
	}


	std::string NLSSolver::SolverInfo::toString() const
	{
		std::stringstream ss;
		ss << "      iteration " << iteration << " result: " <<
			"cost " << cost << ", " <<
			"cost_final " << cost_final << ", " <<
			"cost_change " << cost_change << ", " <<
			"inner_iterations " << inner_iterations << ", " <<
			"time_solve " << time_solve;
		return ss.str();
	}


	NLSSolver::NLSSolver() :
		num_cost_types_(1),
        problem_(nullptr),
        debug_(false),
        calculate_type_costs_(false)
	{
		reset(num_cost_types_);
	}


	NLSSolver::~NLSSolver()
	{
		delete problem_;
	}


    bool NLSSolver::reset(size_t num_cost_types)
	{
        if (num_cost_types <= 0)
			return false;

		tmr_.start();

		// create problem
		delete problem_;
		problem_ = new ceres::Problem();

		// init containers
        num_cost_types_ = num_cost_types;
		residuals_.clear();
		residuals_.resize(num_cost_types_, std::vector<VoxelResidual>());
		cost_type_weights_.clear();
		cost_type_weights_.resize(num_cost_types_, 1.0);

		return true;
	}


    void NLSSolver::setCostWeight(size_t cost_id, double weight)
	{
        if (cost_id >= cost_type_weights_.size())
			return;
        cost_type_weights_[cost_id] = weight;
	}


    double NLSSolver::costWeight(size_t cost_id)
	{
        if (cost_id >= cost_type_weights_.size())
			return 0.0;
        return cost_type_weights_[cost_id];
	}


    void NLSSolver::setDebug(bool debug)
    {
        debug_ = debug;
    }


	bool NLSSolver::addResidual(VoxelResidual &residual)
	{
		return addResidual(0, residual);
	}


    bool NLSSolver::addResidual(size_t cost_id, const VoxelResidual &residual)
	{
        if (cost_id >= cost_type_weights_.size())
			return false;

		if (!residual.cost || residual.weight == 0.0)
		{
			delete residual.cost;
			return false;
		}
		else
		{
            residuals_[cost_id].push_back(residual);
			return true;
		}
	}


    bool NLSSolver::buildProblem(bool use_normalized_weights)
	{
		tmr_.stop();
        double time_add_residuals = tmr_.elapsed();

		tmr_.start();
		// problem info for iteration
		ProblemInfo info;
        info.iteration = problem_info_.size();
		info.residual_types = num_cost_types_;
		info.type_residuals.resize(num_cost_types_, 0);
		info.type_costs.resize(num_cost_types_, 0.0);
		info.type_weights.resize(num_cost_types_, 0.0);
        info.time_add = time_add_residuals;

        std::vector<double> cost_type_weights;
        if (use_normalized_weights)
		{
			// compute normalized cost term weights
			std::cout << "      compute normalized cost term weights ..." << std::endl;
            cost_type_weights = normalizeCostTermWeights();
		}
		else
		{
			// directly use the set cost type weights
            cost_type_weights = cost_type_weights_;
		}

		// add residuals to problem
		std::cout << "      add residuals to problem ..." << std::endl;
        for (size_t cost_id = 0; cost_id < num_cost_types_; ++cost_id)
		{
            auto& cur_cost_residuals = residuals_[cost_id];
            info.type_residuals[cost_id] = cur_cost_residuals.size();
            double cur_cost_weight = cost_type_weights[cost_id];
            info.type_weights[cost_id] = cur_cost_weight;

			// iterate over residual for current residual type
            for (size_t i = 0; i < cur_cost_residuals.size(); ++i)
			{
                VoxelResidual& r = cur_cost_residuals[i];

				// apply cost term weight
                r.weight *= cur_cost_weight;
				// loss for gradient-based shading cost
                ceres::LossFunction* loss = new ceres::ScaledLoss(nullptr, r.weight, ceres::TAKE_OWNERSHIP);

				// add residual
				problem_->AddResidualBlock(r.cost, loss, r.params);

                if (calculate_type_costs_)
                {
                    // compute result cost
                    double residual = 0.0;
                    bool eval = r.cost->Evaluate(&(r.params[0]), &residual, nullptr);
                    if (eval && residual != NV_INVALID_RESIDUAL)
                    {
                        // compute squared cost after applying loss function
                        double residuals_loss[3];
                        loss->Evaluate(residual * residual, residuals_loss);
                        info.type_costs[cost_id] += residuals_loss[0];
                    }
                }

                // clear parameters to immediately reduce memory usage
                r.params.clear();
			}

			// add cost for current residual type to overall cost
            info.cost += 0.5 * info.type_costs[cost_id];
			// clear residuals for current residual type
            cur_cost_residuals.clear();
		}
		tmr_.stop();
		info.time_build = tmr_.elapsed();

		// number of residuals
        info.residuals = static_cast<size_t>(problem_->NumResiduals());
		// number of parameters
        info.parameters = static_cast<size_t>(problem_->NumParameters());
		
		// store problem info
        std::cout << "      " << info.toString(calculate_type_costs_) << std::endl;
		problem_info_.push_back(info);

		return true;
	}

	
	struct SuccessfulStepCallback : public ceres::IterationCallback
	{
	public:
		SuccessfulStepCallback()
		{
		}

		virtual ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary)
		{
			if (summary.iteration > 0 && summary.step_is_successful)
				return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
			else
				return ceres::SOLVER_CONTINUE;
		}
	};


    bool NLSSolver::solve(int lm_steps)
	{
		tmr_.start();
		//solver options
		ceres::Solver::Options options;
        options.max_num_iterations = lm_steps;
        options.minimizer_progress_to_stdout = debug_;
        if (!debug_)
			options.logging_type = ceres::LoggingType::SILENT;
		// solve linear problem using sparse solver:
		// conjugate gradients solver with jacobi preconditioner on normal equations
        options.linear_solver_type = ceres::CGNR;

        // set number of local PCG iterations
        //int lin_solver_iterations = 10;
        //options.min_linear_solver_iterations = lin_solver_iterations;
        //options.max_linear_solver_iterations = lin_solver_iterations;

        //options.sparse_linear_algebra_library_type = ceres::SUITE_SPARSE;
        //options.sparse_linear_algebra_library_type = ceres::CX_SPARSE;
		//options.dense_linear_algebra_library_type = ceres::EIGEN;

		// apply also steps that do not decrease the error
		//options.use_nonmonotonic_steps = true;

		// set initial trust region radius to previous radius for finding a decreasing step faster
		if (!solver_info_.empty())
			options.initial_trust_region_radius = solver_info_[solver_info_.size() - 1].trust_region_radius;

		// callback for stopping in the case of a successful step
		// allows to re-assign the valid correspondences again
		// necessary, because otherwise (if there would be another step after a successful step) a
		// wrong gradient of a changed residual would be computed and used!
        SuccessfulStepCallback successful_step_cb;
        options.callbacks.push_back(&successful_step_cb);

		// multi-threading
		options.num_threads = 8;
		options.num_linear_solver_threads = 8;

		// solve problem
		ceres::Solver::Summary summary;
		ceres::Solve(options, problem_, &summary);
        if (debug_)
			std::cout << summary.FullReport() << std::endl;
		tmr_.stop();
        double time_solve_problem = tmr_.elapsed();

		if (!summary.iterations.empty())
		{
			// store trust region radius for fast convergence
			if (summary.termination_type == ceres::USER_SUCCESS || summary.termination_type == ceres::CONVERGENCE)
			{
                ceres::IterationSummary &it_summary = summary.iterations[summary.iterations.size() - 1];

				// problem info for iteration
				SolverInfo info;
                info.iteration = solver_info_.size();
				info.cost = summary.initial_cost;
				info.cost_final = summary.final_cost;
				info.cost_change = info.cost - info.cost_final;
                info.inner_iterations = summary.iterations.size();
                info.trust_region_radius = it_summary.trust_region_radius;
				info.report = summary.FullReport();
                info.time_solve = time_solve_problem;
				//summary.IsSolutionUsable();
				solver_info_.push_back(info);
				std::cout << "      " << info.toString() << std::endl;
			}
		}

		return true;
	}


	bool NLSSolver::fixParamBlock(double* ptr)
	{
		if (!problem_ || !problem_->HasParameterBlock(ptr))
			return false;
		problem_->SetParameterBlockConstant(ptr);
		return true;
	}


	std::vector<double> NLSSolver::normalizeCostTermWeights()
	{
        double scale_factor = 1000.0;	//1.0;
        std::vector<double> cost_type_weights(num_cost_types_, 0.0);
		for (size_t j = 0; j < num_cost_types_; ++j)
		{
			// sum up residual weights
            double sum_weights = 0.0;
			for (size_t i = 0; i < residuals_[j].size(); ++i)
                sum_weights += residuals_[j][i].weight;
			// normalize cost term weights
            if (sum_weights != 0.0)
                cost_type_weights[j] = (cost_type_weights_[j] / sum_weights) * scale_factor;
		}
        return cost_type_weights;
	}


	void NLSSolver::removeInvalidResiduals()
	{
		// TODO does it actually work correctly?

		// overall cost
        double cost_overall = 0.0;
        ceres::Problem::EvaluateOptions eval_opt;
        //eval_opt.apply_loss_function = false;
        if (!problem_->Evaluate(eval_opt, &cost_overall, nullptr, nullptr, nullptr))
			std::cout << "         Could not evaluate problem!" << std::endl;
        std::cout << "         overall cost: " << cost_overall << std::endl;

		// compute number of valid residuals
        std::vector<ceres::ResidualBlockId> invalid_residuals_blocks;
        std::vector<ceres::ResidualBlockId> problem_residuals_blocks;
        problem_->GetResidualBlocks(&problem_residuals_blocks);
        int num_valid_residuals = 0;
        for (size_t i = 0; i < problem_residuals_blocks.size(); ++i)
		{
            const ceres::CostFunction* cost_function = problem_->GetCostFunctionForResidualBlock(problem_residuals_blocks[i]);
			std::vector<double*> parameters;
            problem_->GetParameterBlocksForResidualBlock(problem_residuals_blocks[i], &parameters);
			double residual = 0.0;
            bool eval = cost_function->Evaluate(&(parameters[0]), &residual, nullptr);
			if (eval && residual != NV_INVALID_RESIDUAL)
                ++num_valid_residuals;
			else
			{
                invalid_residuals_blocks.push_back(problem_residuals_blocks[i]);
			}
		}
        std::cout << "         num valid residuals: " << num_valid_residuals << std::endl;
        for (size_t i = 0; i < invalid_residuals_blocks.size(); ++i)
		{
			// TODO remove loss functions?
            problem_->RemoveResidualBlock(invalid_residuals_blocks[i]);
		}
		std::cout << "         num valid residuals (cleaned): " << problem_->NumResidualBlocks() << std::endl;

		// evaluate cost again
        cost_overall = 0.0;
        if (!problem_->Evaluate(eval_opt, &cost_overall, nullptr, nullptr, nullptr))
			std::cout << "         Could not evaluate problem!" << std::endl;
        std::cout << "         overall cost (cleaned): " << cost_overall << std::endl;
	}

} // namespace nv
