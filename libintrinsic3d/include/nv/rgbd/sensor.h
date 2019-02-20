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
#include <nv/camera.h>
#include <nv/settings.h>
#include <opencv2/core.hpp>

namespace nv
{

    /**
     * @brief   Interface for accessing RGB-D sensors
     *          (contains already generic functionality)
     * @author  Robert Maier <robert.maier@tum.de>
     */
    class Sensor
	{
	public:
        virtual ~Sensor();

        static Sensor* create(const std::string &dataset);
        static Sensor* create(Settings &cfg);

        const Camera& depthCamera() const;
        Camera& depthCamera();
        const Camera& colorCamera() const;
        Camera& colorCamera();

		void setNumFramesMax(int n);
		int numFramesMax();
		int numFrames() const;
		bool frameExists(int id) const;

		void setDepthMin(float d);
		float depthMin() const;
		void setDepthMax(float d);
		float depthMax() const;

		cv::Mat depth(int id);
		cv::Mat color(int id);

		virtual void setPose(int id, const Mat4f &p) = 0;
		virtual Mat4f pose(int id) = 0;
		virtual double timePose(int id) = 0;
		virtual double timeDepth(int id) = 0;
		virtual double timeColor(int id) = 0;

		bool loadDepthIntrinsics(const std::string &filename);
        bool loadColorIntrinsics(const std::string &filename);

		static bool loadPoses(const std::string &filename, std::vector<Mat4f> &poses, 
                                std::vector<double> &timestamps, bool first_pose_is_identity = false);

		bool loadPoses(const std::string &filename);
		bool savePoses(const std::string &filename);

		void print() const;

	protected:
        static Sensor* create(const std::string &dataset, Settings &cfg);

        Sensor();
        Sensor(const Sensor&);
        Sensor& operator=(const Sensor&);

		virtual bool init(const std::string &dataset) = 0;

		void thresholdDepth(cv::Mat &depth) const;

		virtual cv::Mat loadDepth(int id) = 0;
		virtual cv::Mat loadColor(int id) = 0;

        Camera cam_depth_;
        Camera cam_color_;
        int num_frames_max_;
        int num_frames_;
        float depth_min_;
        float depth_max_;
	};

} // namespace nv
