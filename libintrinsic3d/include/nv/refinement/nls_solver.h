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
#include <vector>

#include <ceres/problem.h>

#include <nv/refinement/cost.h>
#include <nv/timer.h>

namespace nv
{

    /**
     * @brief   Nonlinear Least Squares solver class, which
     *          which encapsulates Ceres Solver for easier and
     *          more flexible usage.
     * @author  Robert Maier <robert.maier@tum.de>
     */
	class NLSSolver
	{
	public:

		struct ProblemInfo
		{
			ProblemInfo();

			int iteration;
			size_t residuals;
			size_t parameters;
			double cost;
            int residual_types;
            std::vector<size_t> type_residuals;
            std::vector<double> type_costs;
            std::vector<double> type_weights;
            double time_add;
            double time_build;

            std::string toString(bool print_cost_types = true) const;
		};


		struct SolverInfo
		{
			SolverInfo();

			int iteration;
			double cost;
            double cost_final;
            double cost_change;
            int inner_iterations;
            double trust_region_radius;
			std::string report;
            double time_solve;

			std::string toString() const;
		};


		NLSSolver();
		~NLSSolver();

        bool reset(int num_cost_types = 1);

		bool addResidual(VoxelResidual &residual);
        bool addResidual(int cost_type, const VoxelResidual &residual);
        void setCostWeight(int cost_type, double weight);
        double costWeight(int cost_type);

        void setDebug(bool debug);

        bool buildProblem(bool use_normalized_weights = false);
        bool solve(int lm_steps);

		bool fixParamBlock(double* ptr);

	private:
		std::vector<double> normalizeCostTermWeights();

		void removeInvalidResiduals();

        int num_cost_types_;
		ceres::Problem* problem_;
		std::vector< std::vector<VoxelResidual> > residuals_;
        std::vector<double> cost_type_weights_;

        std::vector<ProblemInfo> problem_info_;
        std::vector<SolverInfo> solver_info_;
		Timer tmr_;
        bool debug_;
	};

} // namespace nv
