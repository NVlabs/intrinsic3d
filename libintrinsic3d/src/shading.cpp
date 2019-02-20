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

#include <nv/shading.h>

#include <iostream>
#include <vector>


namespace nv
{
namespace Shading
{

    Eigen::VectorXf shBasisFunctions(const Vec3f &n)
    {
        // compute spherical harmonics basis functions
        Eigen::VectorXf sh(NUM_SPHERICAL_HARMONICS);
        if (!shBasisFunctions(n.data(), sh.data()))
            sh.setZero();

        bool sh_valid = true;
        for (int i = 0; i < NUM_SPHERICAL_HARMONICS; ++i)
            if (std::isnan(sh[i]) || std::isinf(sh[i]))
                sh_valid = false;
        if (!sh_valid)
            sh.setZero(NUM_SPHERICAL_HARMONICS);

        return sh;
    }


    float computeShading(const Vec3f &normal, const Eigen::VectorXf &sh_coeffs, float albedo)
    {
        float shad = 0.0f;
        if (normal.norm() != 0.0f && !std::isnan(normal.norm()))
        {
            Eigen::VectorXf sh_basis_funcs = Shading::shBasisFunctions(normal);
            if (sh_basis_funcs.norm() != 0.0f && albedo != 0.0f && !std::isnan(albedo))
            {
                shad = albedo * sh_coeffs.dot(sh_basis_funcs);
            }
        }
        return shad;
    }

} // namespace Shading
} // namespace nv
