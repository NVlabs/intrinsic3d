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

#include <nv/rgbd/sensor_i3d.h>

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <fstream>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


namespace nv
{

    SensorI3d::SensorI3d() :
		Sensor(),
		data_folder_("")
	{
	}


    SensorI3d::~SensorI3d()
	{
	}


    bool SensorI3d::init(const std::string &dataset)
	{
		// reset data
		depth_images_.clear();
		depth_timestamps_.clear();
		color_images_.clear();
		color_timestamps_.clear();
        poses_cam_to_world_.clear();
		poses_timestamps_.clear();
		data_folder_ = dataset;

        if (data_folder_.empty())
            return false;

        // load camera intrinsics
        loadIntrinsics(data_folder_ + "/depthIntrinsics.txt", cam_depth_);
        //std::cout << "depth intrinsics " << std::endl << depth_K_ << std::endl;
        loadIntrinsics(data_folder_ + "/colorIntrinsics.txt", cam_color_);
        //std::cout << "color intrinsics " << std::endl << color_K_ << std::endl;

        // load frame filenames
        std::vector<double> timestamps_depth;
        std::vector<std::string> files_depth;
        std::vector<double> timestamps_color;
        std::vector<std::string> files_color;
        std::vector<std::string> files_poses;
        if (!listFiles(timestamps_depth, files_depth, timestamps_color, files_color, files_poses))
        {
            std::cout << "filenames could not be retrieved!" << std::endl;
            return false;
        }
        std::cout << files_depth.size() << " filenames loaded." << std::endl;

        num_frames_ = static_cast<int>(files_depth.size());

        // load rgb-d frames
        for (size_t i = 0; i < files_depth.size(); ++i)
        {
            if (i % 50 == 0)
                std::cout << "Loading frame " << i << "... " << std::endl;

            // load depth image and color image
            std::vector<unsigned char> buf_color, buf_depth;
            if (!loadFrame(files_depth[i], files_color[i], buf_depth, buf_color))
                break;
            // check if input depth and color are empty
            if (buf_depth.empty() || buf_color.empty())
                continue;

            double time_depth = timestamps_depth[i];
            double time_color = timestamps_color[i];

            // get pose
            Mat4f pose_cam_to_world = Mat4f::Identity();
            double time_pose = time_depth;
            if (!loadPose(files_poses[i], pose_cam_to_world))
                break;

            // store rgb-d frame data
            depth_images_.push_back(buf_depth);
            depth_timestamps_.push_back(time_depth);

            color_images_.push_back(buf_color);
            color_timestamps_.push_back(time_color);

            poses_cam_to_world_.push_back(pose_cam_to_world);
            poses_timestamps_.push_back(time_pose);

            if (i == 0)
            {
                // frame dimensions
                cv::Mat c = color(0);
                cam_color_.setWidth(c.cols);
                cam_color_.setHeight(c.rows);
                cv::Mat d = depth(0);
                cam_depth_.setWidth(d.cols);
                cam_depth_.setHeight(d.rows);
            }

            if (num_frames_max_ > 0 && depth_images_.size() >= num_frames_max_)
                break;
        }

		return true;
	}


    bool SensorI3d::loadIntrinsics(const std::string &filename, Camera& cam) const
    {
        bool loaded = false;

        std::ifstream input_file(filename.c_str());
        if (input_file.is_open())
        {
            Mat4f K_file;
            try
            {
                //camera intrinsics
                float val = 0.0f;
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        input_file >> val;
                        K_file(i, j) = val;
                    }
                }
                Mat3f K = K_file.topLeftCorner<3,3>();
                cam.setIntrinsics(K);
                input_file.close();
                loaded = true;
            }
            catch (...)
            {
            }
        }

        if (!loaded)
        {
            std::cerr << "Intrinsics file ('" << filename << "') could not be loaded!" << std::endl;
        }

        return loaded;
    }


    bool SensorI3d::listFiles(std::vector<double> &timestamps_depth,
                                          std::vector<std::string> &files_depth,
                                          std::vector<double> &timestamps_color,
                                          std::vector<std::string> &files_color,
                                          std::vector<std::string> &files_poses) const
    {
        if (data_folder_.empty())
			return false;

        // get frame filenames
        for (size_t i = 0; i < 999999; ++i)
        {
            std::stringstream ss;
            ss << data_folder_ << "/" << "frame-" << std::setfill('0') << std::setw(6) << i;
            std::string filename_base = ss.str();

            // add depth map
            std::string filename_depth = filename_base + ".depth.png";
            if (!std::ifstream(filename_depth.c_str()).is_open())
                break;
            files_depth.push_back(filename_depth);
            double timestamp_depth = static_cast<double>(i);
            timestamps_depth.push_back(timestamp_depth);

            // add color image
            std::string filename_color = filename_base + ".color.png";
            files_color.push_back(filename_color);
            double timestamp_color = timestamp_depth;
            timestamps_color.push_back(timestamp_color);

            // add pose file
            std::string filename_pose = filename_base + ".pose.txt";
            files_poses.push_back(filename_pose);
        }

		return true;
	}


    bool SensorI3d::loadFile(const std::string &filename, std::vector<unsigned char> &data) const
	{
		// read file
        unsigned char* data_file = nullptr;
		size_t size = 0;
		std::ifstream file(filename.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
		if (file.is_open())
		{
			size = file.tellg();
            data_file = new unsigned char[size];
			file.seekg(0, std::ios::beg);
            file.read((char*)data_file, size);
			file.close();

			// assign data to output
            data = std::vector<unsigned char>(data_file, data_file + size);
		}
		return size != 0;
	}


    bool SensorI3d::loadFrame(const std::string &filename_depth, const std::string &filename_color,
                                  std::vector<unsigned char> &depth, std::vector<unsigned char> &color) const
	{
        if (!loadFile(filename_depth, depth))
			return false;
        if (!loadFile(filename_color, color))
			return false;

		return true;
    }


    bool SensorI3d::loadPose(const std::string &filename, Mat4f &pose) const
    {
        bool loaded = false;

        std::ifstream input_file(filename.c_str());
        if (input_file.is_open())
        {
            try
            {
                //camera intrinsics
                float val = 0.0f;
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < 4; j++)
                    {
                        input_file >> val;
                        pose(i, j) = val;
                    }
                }
                input_file.close();
                loaded = true;
            }
            catch (...)
            {
            }
        }

        if (!loaded)
        {
            std::cerr << "Poses file ('" << filename << "') could not be loaded!" << std::endl;
            pose = Mat4f::Identity();
        }

        return loaded;
    }


    void SensorI3d::setPose(int id, const Mat4f &p)
	{
		if (!frameExists(id))
			return;
        poses_cam_to_world_[id] = p;
	}


    Mat4f SensorI3d::pose(int id)
	{
        return frameExists(id) ? poses_cam_to_world_[id] : Mat4f::Identity();
	}


    cv::Mat SensorI3d::loadDepth(int id)
	{
		if (!frameExists(id))
			return cv::Mat();

		// decode depth image
        cv::Mat depth_in = cv::imdecode(cv::Mat(depth_images_[id]), cv::IMREAD_UNCHANGED);
		cv::Mat depth;
        depth_in.convertTo(depth, CV_32FC1, (1.0 / 1000.0));
        return depth;
	}


    cv::Mat SensorI3d::loadColor(int id)
	{
		if (!frameExists(id))
			return cv::Mat();

		// decode color image
		return cv::imdecode(cv::Mat(color_images_[id]), cv::IMREAD_UNCHANGED);
	}


    double SensorI3d::timePose(int id)
	{
		return frameExists(id) ? poses_timestamps_[id] : 0.0;
	}


    double SensorI3d::timeDepth(int id)
	{
		return frameExists(id) ? depth_timestamps_[id] : 0.0;
	}


    double SensorI3d::timeColor(int id)
	{
		return frameExists(id) ? color_timestamps_[id] : 0.0;
	}

} // namespace nv
