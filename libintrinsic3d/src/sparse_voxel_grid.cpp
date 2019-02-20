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

#include <nv/sparse_voxel_grid.h>


#include <fstream>
#include <nv/camera.h>
#include <nv/math.h>


namespace nv
{

	template <class T>
    SparseVoxelGrid<T>::SparseVoxelGrid(float voxelSize, float depth_min, float depth_max) :
        voxel_size_(voxelSize),
        depth_min_(depth_min),
        depth_max_(depth_max),
		truncation_(voxelSize * 5.0f),
        integration_weight_sample_(10.0f),
        clip_bounds_(Vec6f::Zero())
	{
		data_.reserve(64);
		data_.max_load_factor(0.6f);
	}


	template <class T>
    SparseVoxelGrid<T>* SparseVoxelGrid<T>::create(float voxel_size, float depth_min, float depth_max)
	{
        if (voxel_size <= 0.00001f)
            return nullptr;
		// create sparse voxel grid
        return new SparseVoxelGrid<T>(voxel_size, depth_min, depth_max);
	}


	template <class T>
    SparseVoxelGrid<T>* SparseVoxelGrid<T>::create(const std::string &filename, float depth_min, float depth_max)
	{
		if (filename.empty())
            return nullptr;

		// create sparse voxel grid with default values (values are overridden in loading function)
        SparseVoxelGrid<T>* grid = SparseVoxelGrid<T>::create(0.01f, depth_min, depth_max);
		if (!grid)
            return nullptr;

		// load volume from file
		if (!grid->load(filename))
		{
			delete grid;
            grid = nullptr;
		}
		return grid;
	}


	template <class T>
	SparseVoxelGrid<T>::~SparseVoxelGrid()
	{
	}


	template <class T>
	typename SparseVoxelGrid<T>::iterator SparseVoxelGrid<T>::begin()
	{
		return data_.begin();
	}


	template <class T>
	typename SparseVoxelGrid<T>::iterator SparseVoxelGrid<T>::end()
	{
		return data_.end();
	}


	template <class T>
	typename SparseVoxelGrid<T>::const_iterator SparseVoxelGrid<T>::begin() const
	{
		return data_.begin();
	}


	template <class T>
	typename SparseVoxelGrid<T>::const_iterator SparseVoxelGrid<T>::end() const
	{
		return data_.end();
	}


	template <class T>
	void SparseVoxelGrid<T>::setIntegrationWeightSample(float weight)
	{
        integration_weight_sample_ = weight;
	}


	template <class T>
    void SparseVoxelGrid<T>::setClipBounds(const Vec6f &clip_bounds)
	{
        clip_bounds_ = clip_bounds;
	}


	template <class T>
	float SparseVoxelGrid<T>::voxelSize() const
	{
        return voxel_size_;
	}


	template <class T>
    float SparseVoxelGrid<T>::truncation() const
	{
        return truncation_;
	}


	template <class T>
	float SparseVoxelGrid<T>::depthMin() const
	{
        return depth_min_;
	}


	template <class T>
	float SparseVoxelGrid<T>::depthMax() const
	{
        return depth_max_;
	}



	template <class T>
    T& SparseVoxelGrid<T>::voxel(const Vec3i &voxel_pos)
	{
        //assert(exists(voxel_pos));
        return data_.find(voxel_pos)->second;
	}


	template <class T>
    T& SparseVoxelGrid<T>::voxel(const Vec3f& world_pos)
	{
        Vec3i voxel_pos = worldToVoxel(world_pos);
        return voxel(voxel_pos[0], voxel_pos[1], voxel_pos[2]);
	}


	template <class T>
	T& SparseVoxelGrid<T>::voxel(int x, int y, int z)
	{
		return voxel(Vec3i(x, y, z));
	}


	template <class T>
    const T& SparseVoxelGrid<T>::voxel(const Vec3i &voxel_pos) const
	{
        //assert(exists(voxel_pos));
        return data_.find(voxel_pos)->second;
	}


	template <class T>
    const T& SparseVoxelGrid<T>::voxel(const Vec3f& world_pos) const
	{
        return voxel(worldToVoxel(world_pos));
	}


	template <class T>
	const T& SparseVoxelGrid<T>::voxel(int x, int y, int z) const
	{
		return voxel(Vec3i(x, y, z));
	}


    template <class T>
    Vec3i SparseVoxelGrid<T>::worldToVoxel(const Vec3f& p) const
    {
        return round(worldToVoxelFloat(p));
    }


    template <class T>
    Vec3f SparseVoxelGrid<T>::worldToVoxelFloat(const Vec3f& p) const
    {
        return p * (1.0f / voxel_size_);
    }


    template <class T>
    Vec3f SparseVoxelGrid<T>::voxelToWorld(const Vec3i& v) const
    {
        return v.cast<float>() * voxel_size_;
    }


	template <class T>
    bool SparseVoxelGrid<T>::exists(int x, int y, int z) const
	{
        return exists(Vec3i(x ,y, z));
	}


	template <class T>
    bool SparseVoxelGrid<T>::exists(const Vec3i &voxel_pos) const
	{
        return (data_.find(voxel_pos) != data_.end());
	}


	template <class T>
    bool SparseVoxelGrid<T>::valid(int x, int y, int z) const
	{
        return valid(Vec3i(x, y, z));
	}


	template <class T>
    bool SparseVoxelGrid<T>::valid(const Vec3i &voxel_pos) const
	{
        bool valid = exists(voxel_pos);
		if (valid)
            valid = voxel(voxel_pos).weight > 0.0f;
		return valid;
	}


	template <class T>
	size_t SparseVoxelGrid<T>::numVoxels() const
	{
		return data_.size();
	}


	template <class T>
    void SparseVoxelGrid<T>::setVoxel(const Vec3i &voxel_pos, const T &voxel)
	{
        data_[voxel_pos] = voxel;
	}


	template <class T>
	bool SparseVoxelGrid<T>::empty() const
	{
		return numVoxels() == 0;
	}


	template <class T>
	void SparseVoxelGrid<T>::clear()
	{
		data_.clear();
	}


	template <class T>
    bool SparseVoxelGrid<T>::remove(const Vec3i &voxel_pos)
	{
        if (!exists(voxel_pos))
			return false;
        data_.erase(voxel_pos);
		return true;
	}


	template <class T>
    bool SparseVoxelGrid<T>::integrate(const Camera &depth_cam, Camera &color_cam,
                                       const cv::Mat &depth, const cv::Mat &color,
                                       const cv::Mat &normals, const Mat4f &pose_camera_to_world)
	{
        const Mat4f pose_world_to_camera = pose_camera_to_world.inverse();

        // compute frustum bounds to avoid unneccessary allocation and integration
        const Vec6i voxel_bounds = computeFrustumBounds(depth_cam, pose_camera_to_world);

		// first allocate voxels
        alloc(depth_cam, depth, pose_camera_to_world, voxel_bounds);

		// integrate
        const int bucket_cnt = static_cast<int>(data_.bucket_count());
#pragma omp parallel for
        for (int i = 0; i < bucket_cnt; ++i)
		{
			for (auto iter = data_.begin(i); iter != data_.end(i); ++iter)
			{
                const Vec3i& pos_grid = iter->first;
                if (!math::withinBounds(voxel_bounds, pos_grid))
					continue;

                const Vec3f pos_world = voxelToWorld(pos_grid);
				T& v = iter->second;

                // transform to current frame
                const Vec3f p = pose_world_to_camera.topLeftCorner<3, 3>() * pos_world + pose_world_to_camera.topRightCorner<3, 1>();
				if (p[2] < 0.0f)
					continue;
                // project into depth image
                Vec3i pi = round(depth_cam.project2(p));
                if (pi[0] < 0 || pi[1] < 0 || pi[0] >= depth.cols || pi[1] >= depth.rows)
                    continue;
                //c heck for a valid depth range
                const float d = depth.at<float>(pi[1], pi[0]);
                if (d <= 0.0f)
                    continue;

                // compute signed distance; positive in front of the observation
                const float sdf = d - p[2];
                if (sdf <= -truncation_)
                    continue;
                const float tsdf = sdf >= 0.0f ? std::min(truncation_, sdf) : std::max(-truncation_, sdf);

                float weight_update = 1.0f;
                if (integration_weight_sample_ > 0)
                {
                    float w_normal = 1.0f;
                    if (!normals.empty())
                    {
                        cv::Vec3f normal = normals.at<cv::Vec3f>(pi[1], pi[0]);
                        Vec3f n(normal.val[0], normal.val[1], normal.val[2]);
                        w_normal = 1.0f - std::abs(p.normalized().dot(n));
                        //w_normal = std::abs(n.dot(-p));
                        //w_normal = std::abs(n[2]);
                        w_normal = std::max(std::min(w_normal, 1.0f), 0.0f);
                        w_normal = std::max(integration_weight_sample_ *
                                            math::robustKernel(w_normal), 1.0f);
                    }

                    float w_dist = std::max(integration_weight_sample_ *
                                            math::robustKernel(2.0f * std::abs(tsdf) / truncation_), 1.0f);

                    float d_norm = (d - depth_min_) / (depth_max_ - depth_min_);
                    float w_depth = std::max(integration_weight_sample_ * (1.0f - d_norm), 1.0f);

                    weight_update = std::max((w_normal + w_dist + w_depth) / 3.0f, 3.0f);
                }

                // update sdf
                const float w_old = v.weight;
                const float w_new = w_old + weight_update;
                v.sdf = (v.sdf * w_old + sdf * weight_update) / w_new;

                // update color
                // depth and color images have different size
                // project 3d point into color image
                pi = round(color_cam.project2(p));
                if (pi[0] >= 0 && pi[1] >= 0 && pi[0] < color.cols && pi[1] < color.rows)
                {
                    const cv::Vec3b c = color.at<cv::Vec3b>(pi[1], pi[0]);
                    const Vec3f c_old = v.color.template cast<float>();
                    const Vec3f c_new(c[2], c[1], c[0]);
                    const Vec3f c_avg = (c_old * v.weight + c_new * weight_update) / w_new;
                    v.color = c_avg.cast<unsigned char>();
                }

                // update weight
                v.weight = w_new;
            }
		}

		return true;
	}


	template <class T>
    void SparseVoxelGrid<T>::alloc(const Camera &cam, const cv::Mat &depth,
                                   const Mat4f &pose_camera_to_world,
                                   const Vec6i &voxel_bounds)
    {
        const float ray_step_size = voxel_size_ * 0.25f;
        const bool use_clip_bounds = clip_bounds_.norm() > 0.0f;
        const int block_size = 1;
        const Mat3f R_camera_to_world = pose_camera_to_world.topLeftCorner<3, 3>();
        const Vec3f t_camera_to_world = pose_camera_to_world.topRightCorner<3, 1>();
        const T v_init;

		for (int y = 0; y < depth.rows; ++y)
		{
			for (int x = 0; x < depth.cols; ++x)
			{
                const float d = depth.at<float>(y, x);
                if (d == 0.0f)
                    continue;

                // unproject point
                const Vec3f pos_camera = cam.unproject2(x, y, 1.0f);

                // conservative allocation (way slower, but conservative...)
                Vec3i pos_grid_last = Vec3i::Zero();
                for (float d_off = -truncation_; d_off <= truncation_; d_off += ray_step_size)
                {
                    // determine voxel to allocate along ray within truncation
                    const Vec3f pos_camera_ray = pos_camera * (d + d_off);
                    const Vec3f pos_world0 = R_camera_to_world * pos_camera_ray + t_camera_to_world;
                    const Vec3i pos_grid = worldToVoxel(pos_world0);

                    // check whether voxel position has changed
                    if (pos_grid == pos_grid_last)
                        continue;
                    pos_grid_last = pos_grid;

                    // check whether out of bounds defined through min/max depth
                    if (!math::withinBounds(voxel_bounds, pos_grid))
                        continue;

                    if (use_clip_bounds)
                    {
                        // check whether out of clip bounds
                        const Vec3f pos_world = voxelToWorld(pos_grid);
                        if (!math::withinBounds(clip_bounds_, pos_world))
                            continue;
                    }

                    // allocate block around voxel
                    for (int z = -block_size; z <= block_size; z++)
                    {
                        for (int y = -block_size; y <= block_size; y++)
                        {
                            for (int x = -block_size; x <= block_size; x++)
                            {
                                const Vec3i pos_grid_block = pos_grid + Vec3i(x, y, z);
                                // check whether voxel is already allocated
                                if (exists(pos_grid_block))
                                    continue;
                                setVoxel(pos_grid_block, v_init);
                            }
                        }
                    }
				}
			}
		}

        //std::cout << "allocated voxels: " << numVoxels() << std::endl;
	}
	

    template <class T>
    void SparseVoxelGrid<T>::printInfo() const
    {
        std::cout << "SDF volume info:" << std::endl;
        std::cout << "   voxel size: " << voxel_size_ << std::endl;
        std::cout << "   truncation: " << truncation_ << std::endl;
        std::cout << "   integration depth min: " << depth_min_ << std::endl;
        std::cout << "   integration depth max: " << depth_max_ << std::endl;
        //std::cout << "   integration sample weight: " << integration_weight_sample_ << std::endl;
        //std::cout << "   clip bounds: " << clip_bounds_.transpose() << std::endl;
    }


	template <class T>
	bool SparseVoxelGrid<T>::save(const std::string& filename) const
	{
		if (filename.empty())
			return false;

		// open file
        std::ofstream file_out(filename, std::ios::binary);
        if (!file_out.is_open())
			return false;

		// dump sdf volume information
        file_out.write((const char*)&voxel_size_, sizeof(float));
        file_out.write((const char*)&truncation_, sizeof(float));
        file_out.write((const char*)&integration_weight_sample_, sizeof(float));

		// dump sdf data
		size_t size = data_.size();
        float max_load_factor = data_.max_load_factor();
        file_out.write((const char*)&size, sizeof(size_t));
        file_out.write((const char*)&max_load_factor, sizeof(float));
		for (auto iter = begin(); iter != end(); iter++)
		{
			const Vec3i first = iter->first;
			const T second = iter->second;
            file_out.write((const char*)&first, sizeof(Vec3i));
            file_out.write((const char*)&second, sizeof(T));
		}

		// close file
        file_out.close();

		return true;
	}


	template <class T>
	SparseVoxelGrid<T>* SparseVoxelGrid<T>::clone() const
	{
        SparseVoxelGrid<T>* grid = SparseVoxelGrid<T>::create(voxel_size_, depth_min_, depth_max_);
		if (!grid)
            return nullptr;
        grid->integration_weight_sample_ = integration_weight_sample_;
        grid->data_ = data_;
		return grid;
	}


	template <class T>
	bool SparseVoxelGrid<T>::load(const std::string& filename)
	{
		if (filename.empty())
			return false;

		// reset
		data_.clear();

		// open file
        std::ifstream file_in(filename, std::ios::binary);
        if (!file_in.is_open())
			return false;

		// read sdf volume information
        file_in.read((char*)&voxel_size_, sizeof(float));
        file_in.read((char*)&truncation_, sizeof(float));
        file_in.read((char*)&integration_weight_sample_, sizeof(float));

		// read sdf data
		size_t size; 
        float max_load_factor;
        file_in.read((char*)&size, sizeof(size_t));
        file_in.read((char*)&max_load_factor, sizeof(float));
		for (size_t i = 0; i < size; i++)
		{
			Vec3i first; T second;
            file_in.read((char*)&first, sizeof(Vec3i));
            assert(file_in.good());
            file_in.read((char*)&second, sizeof(T));
            assert(file_in.good());
			data_[first] = second;
		}

		// close file
        file_in.close();

		return true;
	}


    template <class T>
    Vec6i SparseVoxelGrid<T>::computeFrustumBounds(const Camera& cam, const Mat4f& pose) const
    {
        std::vector<Vec3f> corner_points;
        math::computeFrustumPoints(cam, depth_min_, depth_max_, corner_points);

        const int min_val = std::numeric_limits<int>::min();
        const int max_val = std::numeric_limits<int>::max();
        Vec6i bbox;
        bbox << max_val, min_val, max_val, min_val, max_val, min_val;
        for (unsigned int i = 0; i < 8; i++)
        {
            Vec3f pt = pose.topLeftCorner<3, 3>() * corner_points[i] + pose.topRightCorner<3, 1>();
            Vec3i pl = worldToVoxel(floor(pt).cast<float>());
            Vec3i pu = worldToVoxel(ceil(pt).cast<float>());

            bbox[0] = std::min(bbox[0], pl[0]);
            bbox[0] = std::min(bbox[0], pu[0]);
            bbox[1] = std::max(bbox[1], pl[0]);
            bbox[1] = std::max(bbox[1], pu[0]);
            bbox[2] = std::min(bbox[2], pl[1]);
            bbox[2] = std::min(bbox[2], pu[1]);
            bbox[3] = std::max(bbox[3], pl[1]);
            bbox[3] = std::max(bbox[3], pu[1]);
            bbox[4] = std::min(bbox[4], pl[2]);
            bbox[4] = std::min(bbox[4], pu[2]);
            bbox[5] = std::max(bbox[5], pl[2]);
            bbox[5] = std::max(bbox[5], pu[2]);
        }
        return bbox;
    }

	// explicit instantiations
	template class SparseVoxelGrid<Voxel>;
    template class SparseVoxelGrid<VoxelSBR>;

} // namespace nv
