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


#include <string>
#include <vector>
#include <nv/mat.h>
#include <nv/rgbd/sensor.h>
#include <opencv2/core.hpp>


namespace nv
{

    /**
     * @brief   RGB-D sensor for loading RGB-D data from
     *          Intrinsic3D dataset folder
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class SensorI3d : public Sensor
	{
	public:
        SensorI3d();
        virtual ~SensorI3d();

		virtual void setPose(int id, const Mat4f &p);
		virtual Mat4f pose(int id);
		virtual double timePose(int id);
		virtual double timeDepth(int id);
		virtual double timeColor(int id);

	private:
		virtual bool init(const std::string &dataset);

        bool loadIntrinsics(const std::string &filename, Camera &cam) const;

        bool listFiles(std::vector<double> &timestamps_depth,
                       std::vector<std::string> &files_depth,
                       std::vector<double> &timestamps_color,
                       std::vector<std::string> &files_color,
                       std::vector<std::string> &files_poses) const;

		bool loadFile(const std::string &filename, std::vector<unsigned char> &data) const;
        bool loadFrame(const std::string &filename_depth, const std::string &filename_color,
                       std::vector<unsigned char> &depth, std::vector<unsigned char> &color) const;
        bool loadPose(const std::string &filename, Mat4f &pose) const;

		virtual cv::Mat loadDepth(int id);
		virtual cv::Mat loadColor(int id);

        std::string data_folder_;
        std::vector<Mat4f> poses_cam_to_world_;
        std::vector< std::vector<unsigned char> > depth_images_;
        std::vector< std::vector<unsigned char> > color_images_;
        std::vector<double> poses_timestamps_;
        std::vector<double> depth_timestamps_;
        std::vector<double> color_timestamps_;
	};

} // namespace nv
