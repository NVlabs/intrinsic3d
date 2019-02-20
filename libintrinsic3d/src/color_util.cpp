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

#include <nv/color_util.h>

#include <iostream>
#include <ctime>


namespace nv
{

	float intensity(unsigned char r, unsigned char g, unsigned char b)
	{
        return 0.299f * static_cast<float>(r) +
                0.587f * static_cast<float>(g) +
                0.114f * static_cast<float>(b);
	}


	float intensity(const Vec3b &color)
	{
		return intensity(color[0], color[1], color[2]);
	}


	float intensity(const Vec3f &color)
	{
		return 0.299f * color[0] + 0.587f * color[1] + 0.114f * color[2];
	}


	Vec3f chromacity(const Vec3b &color)
	{
		Vec3f c = color.cast<float>();
		float lum = intensity(color);
		Vec3f chrom = c * (1.0f / std::max(lum, 0.001f));
		return chrom;
	}


    template <typename T>
    Vec3b scalarToColor(const T val, const T scale)
    {
		const T min_val = static_cast<T>(0.0);
        const T max_val = static_cast<T>(255.0);
        const T color_val = std::min(std::max(val * scale, min_val), max_val);
        unsigned char c = static_cast<unsigned char>(color_val);
        return Vec3b(c, c, c);
    }
    template Vec3b scalarToColor(const float val, const float scale);
    template Vec3b scalarToColor(const double val, const double scale);


    Vec3f checkRange(const Vec3f &vec, const float min_val, const float max_val)
	{
		Vec3f v;
        v[0] = std::min(std::max(vec[0], min_val), max_val);
        v[1] = std::min(std::max(vec[1], min_val), max_val);
        v[2] = std::min(std::max(vec[2], min_val), max_val);
		return v;
	}


	double rand()
	{
		static bool init = false;
		if (!init)
		{
            init = true;
            std::srand(std::clock());
		}

		double r = double(std::rand()) / RAND_MAX;
		return r;
	}


    Vec3b randomColor()
	{
		Vec3f c;
        for (int i = 0; i < 3; ++i)
            c[i] = static_cast<float>(rand());
        c = checkRange(c * 255.0f);
		return c.cast<unsigned char>();
	}

} // namespace nv
