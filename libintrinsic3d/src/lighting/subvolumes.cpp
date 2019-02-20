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

#include <nv/lighting/subvolumes.h>

#include <iostream>

#include <nv/color_util.h>
#include <nv/math.h>


namespace nv
{

	Subvolumes::Subvolumes(float size) :
		size_(size),
		voxel_size_(0.0f)
	{
	}


	Subvolumes::~Subvolumes()
	{
	}


    void Subvolumes::clear()
    {
        subvolumes_.clear();
        indices_.clear();
        bounds_.clear();
        colors_.clear();
    }


    bool Subvolumes::compute(const SparseVoxelGrid<VoxelSBR>* grid)
	{
		if (!grid)
			return false;

        voxel_size_ = grid->voxelSize();
        clear();

		if (size_ <= 0.0f)
		{
			// only a single volume
			Vec6i bounds = Vec6i::Zero();
			//bounds = math::computeBounds(grid);
			bounds_.push_back(bounds);
			indices_.push_back(Vec3i::Zero());
			subvolumes_[Vec3i::Zero()] = 0;
		}
		else
		{
			// split volume into subvolumes in all three dimensions
			generate(grid);
		}

		// generate random subvolume color
		for (size_t i = 0; i < bounds_.size(); ++i)
		{
			Vec3b c = randomColor();
			colors_.push_back(c);
		}

		return true;
	}


	float Subvolumes::subvolumeSize() const
	{
		return size_;
	}


    size_t Subvolumes::count() const
	{
        return bounds_.size();
	}


    Vec3i Subvolumes::index(int subvol) const
	{
		assert(subvol >= 0 && subvol < indices_.size());
		return indices_[subvol];
	}


    Vec6i Subvolumes::bounds(int subvol) const
	{
		assert(subvol >= 0 && subvol < indices_.size());
		return bounds_[subvol];
	}


    Vec3b Subvolumes::color(int subvol) const
	{
		assert(subvol >= 0 && subvol < indices_.size());
		return colors_[subvol];
	}


    bool Subvolumes::exists(int subvol) const
	{
		return subvol >= 0 && subvol < indices_.size();
	}


	bool Subvolumes::exists(const Vec3i& idx) const
	{
		return exists(indexToSubvolume(idx));
	}


    int Subvolumes::pointToSubvolume(const Vec3f& p) const
	{
		// convert point to subvolume index
		Vec3i idx = pointToIndex(p);
		// get subvolume for index
		return indexToSubvolume(idx);
	}


    int Subvolumes::indexToSubvolume(const Vec3i& idx) const
	{
		if (indices_.size() == 1 && bounds_[0].isZero())
			return 0;

		int subvol = -1;
		if (subvolumes_.find(idx) != subvolumes_.end())
			subvol = subvolumes_.find(idx)->second;
		return subvol;
	}


	template <class T>
	T Subvolumes::interpolate(const std::vector<T> &values, const Vec3f &pt, bool linear) const
	{
		// initialize result
		T avg;
		if (!values.empty())
			avg = values[0];
		avg.setZero();

		if (linear)
		{
			// get subvolume index
            Vec3f subvol_idx = pointToIndexCoord(pt);
			// get interpolation weights for neighboring subvolumes
			Vec3i coords[8];
			float weights[8];
            math::interpolationWeights(subvol_idx, coords, weights);

			// tri-linear interpolation
            T val_neighbors[8];
			for (size_t i = 0; i < 8; i++)
			{
				if (exists(coords[i]))
				{
					int subvol = indexToSubvolume(coords[i]);
                    val_neighbors[i] = values[subvol];
				}
				else
					weights[i] = 0.0f;
			}
            avg = math::average(weights, val_neighbors);
		}
		else
		{
			// nearest neighbor
			int subvol = pointToSubvolume(pt);
			if (subvol >= 0)
				avg = values[subvol];
		}

		return avg;
	}
	template Eigen::VectorXd Subvolumes::interpolate(const std::vector<Eigen::VectorXd> &values, const Vec3f &pt, bool linear) const;
	template Vec3f Subvolumes::interpolate(const std::vector<Vec3f> &values, const Vec3f &pt, bool linear) const;
	template Vec3 Subvolumes::interpolate(const std::vector<Vec3> &values, const Vec3f &pt, bool linear) const;


    void Subvolumes::generate(const SparseVoxelGrid<VoxelSBR> *grid)
	{
		// determine subvolumes to be allocated
		for (auto iter = grid->begin(); iter != grid->end(); iter++)
		{
            const Vec3i& v_pos = iter->first;
            const Vec3f v_coord = grid->voxelToWorld(v_pos);
			// convert point to subvolume index
            Vec3i idx = pointToIndex(v_coord);
			// insert subvolume
			if (subvolumes_.find(idx) == subvolumes_.end())
				subvolumes_[idx] = 0;
		}

		// generate subvolumes
		int cnt = 0;
		for (auto iter = subvolumes_.begin(); iter != subvolumes_.end(); iter++)
		{
			const Vec3i& idx = iter->first;
            Vec6i voxel_bounds = indexToBounds(idx);

			subvolumes_[idx] = cnt;
			indices_.push_back(idx);
            bounds_.push_back(voxel_bounds);
			++cnt;
		}

		//std::cout << "number of generated subvolumes: " << bounds_.size() << std::endl;
	}


	int Subvolumes::indexToVoxel(int idx) const
	{
		assert(voxel_size_ != 0.0f);
        float voxel_idx_f = (float)idx * size_ / voxel_size_;
        voxel_idx_f = std::round(voxel_idx_f);
        //voxel_idx_f = std::floor(voxelIdxF + 0.5f);
        int voxel_idx = (int)voxel_idx_f;
        return voxel_idx;
	}


	Vec6i Subvolumes::indexToBounds(const Vec3i& idx) const
	{
		Vec6i subvolume;
		subvolume[0] = indexToVoxel(idx[0]);
		subvolume[1] = indexToVoxel(idx[0] + 1) - 1;
		subvolume[2] = indexToVoxel(idx[1]);
		subvolume[3] = indexToVoxel(idx[1] + 1) - 1;
		subvolume[4] = indexToVoxel(idx[2]);
		subvolume[5] = indexToVoxel(idx[2] + 1) - 1;
		return subvolume;
	}

		
	float Subvolumes::pointToIndexFloat(const float pt) const
	{
		return pt * (1.0f / size_);
	}


	Vec3f Subvolumes::pointToIndexFloat(const Vec3f &pt) const
	{
		Vec3f idx;
		idx[0] = pointToIndexFloat(pt[0]);
		idx[1] = pointToIndexFloat(pt[1]);
		idx[2] = pointToIndexFloat(pt[2]);
		return idx;
	}

	int Subvolumes::pointToIndex(const float pt) const
	{
        float idx_f = pointToIndexFloat(pt);
        return static_cast<int>(std::floor(idx_f));
	}


	Vec3i Subvolumes::pointToIndex(const Vec3f &pt) const
	{
		Vec3i idx;
		idx[0] = pointToIndex(pt[0]);
		idx[1] = pointToIndex(pt[1]);
		idx[2] = pointToIndex(pt[2]);
		return idx;
	}


	Vec3f Subvolumes::pointToIndexCoord(const Vec3f &pt) const
	{
        Vec3f subvol_idx = pointToIndexFloat(pt);
		// subtract 0.5 offset to account for subvolume center
        subvol_idx -= Vec3f::Ones() * 0.5f;
        return subvol_idx;
	}

} // namespace nv
