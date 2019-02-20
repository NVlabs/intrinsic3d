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

#include <nv/rgbd/sensor.h>

#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <fstream>

#include <Eigen/Geometry>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <nv/rgbd/sensor_i3d.h>


namespace nv
{

    Sensor::Sensor() :
        cam_depth_(),
        cam_color_(),
		num_frames_max_(0),
        num_frames_(0),
		depth_min_(0.0f),
		depth_max_(0.0f)
	{
	}


    Sensor::~Sensor()
	{
	}


    Sensor* Sensor::create(const std::string &dataset, Settings &cfg)
	{
        Sensor* sensor = nullptr;

        // determine sensor type from dataset folder/file

        // load dataset in TUM RGB-D benchmark format
        sensor = new SensorI3d();

		if (!cfg.empty())
		{
			// configure sensor
            sensor->setNumFramesMax(cfg.get<int>("max_frames"));
            sensor->setDepthMin(cfg.get<float>("min_depth"));
            sensor->setDepthMax(cfg.get<float>("max_depth"));
		}

		// initialize sensor
        if (!sensor->init(dataset))
		{
			std::cerr << "RGB-D sensor could not be initialized!" << std::endl;
            return nullptr;
		}

#if 0
        std::cout << "RGB-D sensor: " << sensor->numFrames() << " frames loaded..." << std::endl;
        std::cout << "max frames: " << sensor->numFramesMax() << std::endl;
        std::cout << "depth min: " << sensor->depthMin() << std::endl;
        std::cout << "depth max: " << sensor->depthMax() << std::endl;
#endif

        return sensor;
	}


    Sensor* Sensor::create(const std::string &dataset)
	{
		Settings cfg;
        return Sensor::create(dataset, cfg);
	}


    Sensor* Sensor::create(Settings &cfg)
	{
        if (cfg.empty())
            return nullptr;

		// dataset folder/file
		std::string dataset = cfg.get<std::string>("dataset");
		// create sensor
        Sensor* sensor = Sensor::create(dataset, cfg);
        return sensor;
	}
	

    const Camera& Sensor::depthCamera() const
	{
        return cam_depth_;
	}


    Camera& Sensor::depthCamera()
    {
        return cam_depth_;
    }


    const Camera& Sensor::colorCamera() const
    {
        return cam_color_;
    }


    Camera& Sensor::colorCamera()
    {
        return cam_color_;
    }


    void Sensor::setNumFramesMax(int n)
	{
		num_frames_max_ = n;
	}


    int Sensor::numFramesMax()
	{
		return num_frames_max_;
	}


    int Sensor::numFrames() const
	{
		return num_frames_;
	}


    bool Sensor::frameExists(int id) const
	{
		if (id < 0 || id >= num_frames_)
			return false;
		else
			return true;
	}


    void Sensor::setDepthMin(float d)
	{
		depth_min_ = d;
	}


    float Sensor::depthMin() const
	{
		return depth_min_;
	}


    void Sensor::setDepthMax(float d)
	{
		depth_max_ = d;
	}


    float Sensor::depthMax() const
	{
		return depth_max_;
	}


    cv::Mat Sensor::depth(int id)
	{
		cv::Mat d = loadDepth(id);
        if (!d.empty())
            thresholdDepth(d);
		return d;
	}


    cv::Mat Sensor::color(int id)
	{
		return loadColor(id);
	}


    void Sensor::thresholdDepth(cv::Mat &depth) const
	{
        if (depth.empty())
            return;

		if (depth_min_ > 0.0f)
            cv::threshold(depth, depth, static_cast<double>(depth_min_), 0.0, cv::THRESH_TOZERO);
		if (depth_max_ > 0.0f)
            cv::threshold(depth, depth, static_cast<double>(depth_max_), 0.0, cv::THRESH_TOZERO_INV);
	}


    bool Sensor::loadDepthIntrinsics(const std::string &filename)
	{
        return cam_depth_.load(filename);
	}


    bool Sensor::loadColorIntrinsics(const std::string &filename)
	{
        return cam_color_.load(filename);
	}


    bool Sensor::loadPoses(const std::string &filename, std::vector<Mat4f> &poses,
                               std::vector<double> &timestamps, bool first_pose_is_identity)
	{
		// load transformations from CVPR RGBD datasets benchmark

		// open input file for TUM RGB-D benchmark poses
		std::ifstream poses_file;
		poses_file.open(filename.c_str());
		if (!poses_file.is_open())
			return false;

		// first load all groundtruth timestamps and poses
		std::string line;
		while (std::getline(poses_file, line))
		{
			if (line.empty() || line.compare(0, 1, "#") == 0)
				continue;
			std::istringstream iss(line);
			double timestamp;
			float tx, ty, tz;
			float qx, qy, qz, qw;
			if (!(iss >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw))
				break;

			// timestamp
			timestamps.push_back(timestamp);

			// pose
			Mat4f p = Mat4f::Identity();
			p.topRightCorner<3, 1>() = Vec3f(tx, ty, tz);
			p.topLeftCorner<3, 3>() = Eigen::Quaternionf(qw, qx, qy, qz).toRotationMatrix();
			poses.push_back(p);
		}
		// close input file
		poses_file.close();

		// align all poses so that initial pose is identity matrix
        if (first_pose_is_identity && !poses.empty())
		{
            Mat4f init_pose = poses[0];
			for (int i = 0; i < poses.size(); ++i)
                poses[i] = init_pose.inverse() * poses[i];
		}
		return true;
	}


    bool Sensor::loadPoses(const std::string &filename)
	{
		// load all timestamps and poses
		std::vector<double> timestamps;
		std::vector<Mat4f> poses;
		if (!loadPoses(filename, poses, timestamps, false))
			return false;
		
		// check whether number of poses is consistent
		if (poses.size() != num_frames_)
			return false;

        bool timestamps_valid = true;
		for (int i = 0; i < num_frames_; i++)
		{
			if (std::abs(timePose(i) - timestamps[i]) > 0.030f)
			{
                timestamps_valid = false;
				break;
			}
		}

		// store poses
		for (int i = 0; i < num_frames_; i++)
		{
			setPose(i, poses[i]);
		}
		std::cout << poses.size() << " poses loaded." << std::endl;

		return true;
	}


    bool Sensor::savePoses(const std::string &filename)
	{
        if (filename.empty())
			return false;

		// open output file for TUM RGB-D benchmark poses
        std::ofstream out_file;
        out_file.open(filename.c_str());
        if (!out_file.is_open())
			return false;

		// write poses into output file
        out_file << std::fixed << std::setprecision(6);
		for (int i = 0; i < num_frames_; i++)
		{
			//timestamp
            out_file << timePose(i) << " ";

			// pose
			Mat4f p = pose(i);
			//translation
			Vec3f t = p.topRightCorner<3,1>();
            out_file << t[0] << " " << t[1] << " " << t[2];
			//rotation (quaternion)
			Eigen::Quaternionf q(p.topLeftCorner<3, 3>());
            out_file << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
		}

		// close output file
        out_file.close();

		return true;
	}


    void Sensor::print() const
	{
		std::cout << "RGB-D sensor" << std::endl;
        std::cout << "   # frames: " << num_frames_ << std::endl;
		std::cout << "   depth min: " << depth_min_ << std::endl;
		std::cout << "   depth max: " << depth_max_ << std::endl;
        std::cout << "Depth camera:"<< std::endl;
        cam_depth_.print();
        std::cout << "Color camera:"<< std::endl;
        cam_color_.print();
	}

} // namespace nv
