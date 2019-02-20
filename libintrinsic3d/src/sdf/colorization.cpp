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

#include <nv/sdf/colorization.h>

#include <iostream>
#include <numeric>

#include <nv/camera.h>
#include <nv/rgbd/processing.h>
#include <nv/sdf/operators.h>
#include <nv/math.h>


namespace nv
{

    bool VertexObservation::operator<(const VertexObservation &obs) const
    {
        return weight < obs.weight;
    }


    SDFColorization::SDFColorization(const Camera &cam,
                                     SparseVoxelGrid<VoxelSBR>* grid) :
        cam_(cam),
        grid_(grid)
	{
        reset(grid_);
	}


	SDFColorization::~SDFColorization()
	{
	}


    bool SDFColorization::reset(SparseVoxelGrid<VoxelSBR>* grid)
	{
		if (!grid || grid->empty())
			return false;

		// store grid
		grid_ = grid;

		// initialize voxel observations
		voxel_observations_.clear();
		for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
			voxel_observations_.push_back(std::vector<VertexObservation>());

		return true;
	}


    bool SDFColorization::reset(SparseVoxelGrid<VoxelSBR>* grid, const Vec4 &intrinsics, const Vec5 &dist_coeffs)
    {
        cam_.setIntrinsics(intrinsics);
        cam_.setDistortion(dist_coeffs.cast<float>());
        return reset(grid);
    }


    bool SDFColorization::reset(SparseVoxelGrid<VoxelSBR>* grid, const Camera &cam)
    {
        cam_ = cam;
        return reset(grid);
    }


    void SDFColorization::setConfig(const Config &cfg)
    {
        cfg_ = cfg;
    }


    const SDFColorization::Config& SDFColorization::config() const
    {
        return cfg_;
    }


    bool SDFColorization::add(int id, const cv::Mat &depth, const cv::Mat &color, const Mat4f &pose_world_to_cam)
	{
		if (!grid_ || grid_->empty())
			return false;

		// color and depth image are supposed to have the same size
		if (depth.rows != color.rows || depth.cols != color.cols)
		{
			std::cerr << "color and depth image sizes do not match!" << std::endl;
			return false;
		}

		// set depth to 0.0f close to depth discontinuities
        cv::Mat depth_map = erodeDiscontinuities(depth, cfg_.discont_distance);

        // collect observations
        size_t num_frame_obs = 0;
        size_t vx_id = 0;
		for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
			// get voxel
            const Vec3i& pos_grid = itr->first;
			bool valid = true;

			// get normal
            Vec3f normal = SDFOperators::computeSurfaceNormal(grid_, pos_grid);
            if (normal.isZero())
				valid = false;

            if (valid)
            {
                // get voxel observation in view
                VertexObservation obs = computeObservation(pos_grid, normal, pose_world_to_cam, color, depth);
				if (obs.weight > 0.0f)
				{
                    obs.frame = id;
                    voxel_observations_[vx_id].push_back(obs);
                    ++num_frame_obs;
				}
            }

            ++vx_id;
		}
        //std::cout << "num obs: " << num_frame_obs << std::endl;

		return true;
	}


	bool SDFColorization::compute()
    {
        if (!grid_ || grid_->empty() || voxel_observations_.empty())
			return false;
		if (grid_->numVoxels() != voxel_observations_.size())
			return false;

		size_t i = 0;
		for (auto itr = grid_->begin(); itr != grid_->end(); itr++)
		{
            if (!voxel_observations_[i].empty())
			{
                // filter observations
                if (cfg_.max_num_observations > 0)
                    SDFColorization::filter(voxel_observations_[i], cfg_.max_num_observations);

				// compute average color
				Vec3f c = computeColor(voxel_observations_[i]);
				voxel_observations_[i].clear();
				// update vertex color
				VoxelSBR& v = itr->second;
				v.color = c.cast<unsigned char>();
			}
			++i;
		}

		return true;
	}


    std::vector<VertexObservation> SDFColorization::collectObservations(const std::vector<Vec6> &poses, std::vector<Pyramid> &frames_pyr,
                                                                           const Vec3i &v_pos, const Vec3f &n, int pyr_lvl) const
    {
        size_t num_poses = poses.size();
        std::vector<VertexObservation> voxel_observations(num_poses);
        for (size_t f_id = 0; f_id < num_poses; ++f_id)
        {
            // get voxel observation in view
            Mat4f pose_world_to_camera = math::poseVecAAToMat(poses[f_id]).cast<float>();
            cv::Mat color = frames_pyr[f_id].color(pyr_lvl);
            cv::Mat depth = frames_pyr[f_id].depth(pyr_lvl);
            VertexObservation obs = computeObservation(v_pos, n, pose_world_to_camera, color, depth);
            obs.frame = f_id;
            voxel_observations[f_id] = obs;
        }

        // keep only best n observations
        SDFColorization::filter(voxel_observations, cfg_.max_num_observations);

        return voxel_observations;
    }


    VertexObservation SDFColorization::computeObservation(const Vec3i &v_pos, const Vec3f &normal,
                                                          const Mat4f &pose_world_to_cam, const cv::Mat &color,
                                                          const cv::Mat &depth) const
    {
        VertexObservation obs;

        const VoxelSBR& v = grid_->voxel(v_pos);
        bool valid = true;

        // transform point onto iso-surface
        Vec3f pt = SDFOperators::voxelCenterToIso(grid_, v_pos, normal);
        // transform voxel back into input view
        Vec3f pt_tf = pose_world_to_cam.topLeftCorner(3, 3) * pt + pose_world_to_cam.topRightCorner(3, 1);

        // project 3d point into input view
        Vec2f pt2f;
        Vec2i pt2i;
        if (valid)
            valid = cam_.project(pt_tf, pt2f, pt2i);
        if (valid)
            valid = isVoxelVisible(pt_tf, depth, pt2i[0], pt2i[1]);
        // check voxel visibility by comparing signed distance
        if (valid)
        {
            // compute weight
            Vec3f n = pose_world_to_cam.topLeftCorner(3, 3) * normal;
            float w = computeWeight(depth, n, pt2i[0], pt2i[1], pt_tf);
            if (w > 0.0f)
            {
                // lookup color using bilinear interpolation
                obs.color = interpolateRGB(color, pt2f[0], pt2f[1]);
                // store weight
                obs.weight = w;
            }
        }

        return obs;
    }


	bool SDFColorization::isVoxelVisible(const Vec3f& pt, const cv::Mat& depth, int x, int y) const
	{
        if (cfg_.max_occlusion_distance <= 0.0f)
            return true;

		// check voxel visibility by comparing signed distance
        bool visible = false;
        const float d = depth.at<float>(y, x);
        if (d > 0.0f)
		{
			// compute signed distance
			float sdf = d - pt[2];
            if (std::abs(sdf) <= cfg_.max_occlusion_distance)
                visible = true;
		}
        return visible;
    }



    float SDFColorization::computeWeight(const cv::Mat &depth, const Vec3f &n, const int x, const int y, const Vec3f &v) const
    {
        float w = 0.0f;

        // lookup depth
        const float d = depth.at<float>(y, x);
        if (d <= 0.0f)
            return w;

        // computed normal based weighting
        float w_normal = 0.0f;
        if (!n.isZero())
        {
            // use cosine between normal and viewing direction
            // this version seems to give best colors
            w_normal = 1.0f - std::abs(v.normalized().dot(n));
            w_normal = std::max(std::min(w_normal, 1.0f), 0.0f);
            w_normal = std::max(math::robustKernel(w_normal), 0.001f);
        }

        // scale depth between min and max depth
        const float d_min = 0.01f;
        const float d_max = 5.0f;
        float dw = std::max(std::min(d_max, d), d_min);

        // computed depth based weighting
        float w_depth = 0.0f;
#if 0
        // uncertainty increases quadratically with depth
        w_depth = 1.0f / (dw * dw);
#else
        const float depth_normalized = (dw - d_min) / (d_max - d_min);
        w_depth = std::max(1.0f - depth_normalized, 1.0f);
#endif
        w_depth = std::max(std::min(w_depth, 5.0f), 0.001f);

        // compose overall weighting
        w = w_normal * w_depth;
        //w = (w_normal + w_depth) / 2.0f;

        return w;
    }


    Vec3f SDFColorization::computeColor(const std::vector<VertexObservation> &observations) const
    {
        if (observations.empty())
            return cfg_.color_unobserved.cast<float>();

        const float scale_color = 1.0f / 255.0f;
        const size_t num_obs = observations.size();

        // collect observations for vertex
        std::vector<unsigned char> r_values(num_obs, 0);
        std::vector<unsigned char> g_values(num_obs, 0);
        std::vector<unsigned char> b_values(num_obs, 0);
        std::vector<float> weights(num_obs, 0.0f);
        for (size_t j = 0; j < num_obs; ++j)
        {
            VertexObservation obs = observations[j];
            r_values[j] = obs.color[0];
            g_values[j] = obs.color[1];
            b_values[j] = obs.color[2];
            weights[j] = obs.weight;
        }

        // compute new color using weighted average
        Vec3f c = Vec3f::Zero();
        float weight_sum = 0.0f;
        for (size_t t = 0; t < num_obs; ++t)
        {
            c[0] += r_values[t] * (weights[t] * scale_color);
            c[1] += g_values[t] * (weights[t] * scale_color);
            c[2] += b_values[t] * (weights[t] * scale_color);
            weight_sum = weight_sum + weights[t];
        }
        // mean color
        if (weight_sum > 0.0f)
            c = c * (255.0f / weight_sum);
        return c;
    }


    void SDFColorization::filter(std::vector<VertexObservation> &observations, size_t n)
    {
        size_t num_obs = observations.size();
        if (n == 0 || n >= num_obs)
            return;

        // sort voxel observations by weight
        std::sort(observations.begin(), observations.end());
        // select best n observations
        size_t start_id_best_obs = (num_obs - n);
        for (size_t i = 0; i < observations.size(); ++i)
            if (i < start_id_best_obs)
                observations[i].weight = 0.0f;
    }

} // namespace nv
