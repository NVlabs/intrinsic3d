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
#include <unordered_map>
#include <nv/sparse_voxel_grid.h>

namespace nv
{


    /**
     * @brief   Subvolumes generator from a Sparse Signed Distance Field
     * @author  Robert Maier <robert.maier@tum.de>
     */
	class Subvolumes
	{
	public:
		Subvolumes(float size);
		~Subvolumes();

        void clear();
        bool compute(const SparseVoxelGrid<VoxelSBR>* grid);

		float subvolumeSize() const;

        size_t count() const;

        Vec3i index(int subvol) const;
        Vec6i bounds(int subvol) const;
        Vec3b color(int subvol) const;

        bool exists(int subvol) const;
		bool exists(const Vec3i &idx) const;

		Vec3f pointToIndexCoord(const Vec3f &pt) const;
        int pointToSubvolume(const Vec3f &p) const;
        int indexToSubvolume(const Vec3i &idx) const;

		template <class T>
		T interpolate(const std::vector<T> &values, const Vec3f &pt, bool linear = true) const;

	private:
        void generate(const SparseVoxelGrid<VoxelSBR>* grid);

		int indexToVoxel(int idx) const;
		Vec6i indexToBounds(const Vec3i& idx) const;

		float pointToIndexFloat(const float pt) const;
		Vec3f pointToIndexFloat(const Vec3f &pt) const;
		int pointToIndex(const float pt) const;
		Vec3i pointToIndex(const Vec3f &pt) const;

		float size_;
        float voxel_size_;
		std::unordered_map<Vec3i, int, std::hash<Vec3i> > subvolumes_;
		std::vector<Vec3i> indices_;
		std::vector<Vec6i> bounds_;
		std::vector<Vec3b> colors_;
	};

} // namespace nv
