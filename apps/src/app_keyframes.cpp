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

#include <nv/app_keyframes.h>

#include <iostream>

#include <nv/mat.h>
#include <opencv2/core.hpp>

#include <nv/filesystem.h>
#include <nv/keyframe_selection.h>
#include <nv/rgbd/sensor.h>


namespace nv
{
    AppKeyframes::AppKeyframes() :
        sensor_(nullptr)
	{
	}


    AppKeyframes::~AppKeyframes()
	{
		delete sensor_;
	}
	

    bool AppKeyframes::run(int argc, char *argv[])
	{
		// parse command line arguments
        // command line usage:
        // <app>
        //       -s "<path>/sensor.yml"
        //       -k "<path>/keyframes.yml"
        const char *keys = {
            "{s sensor| |sensor config file}"
            "{k keyframes| |keyframes selection config file}"
        };
		cv::CommandLineParser cmd(argc, argv, keys);
        const std::string sensor_cfg_file = cmd.get<std::string>("sensor");
        const std::string keyframes_cfg_file = cmd.get<std::string>("keyframes");

        // change working dir to sensor config file directory
        Filesystem::changeWorkingDir(sensor_cfg_file, true);
        // create default output folder
        Filesystem::createFolder("./fusion");

		// create sensor
        Settings sensor_cfg(sensor_cfg_file);
        sensor_ = Sensor::create(sensor_cfg);
		if (!sensor_)
			return false;
		// print rgbd sensor info
		sensor_->print();

        // select keyframes
        Settings keyframes_cfg(keyframes_cfg_file);
        if (!selectKeyframes(keyframes_cfg))
        {
            std::cerr << "Keyframe selection failed!" << std::endl;
            return false;
        }

		// close all opencv windows
		cv::destroyAllWindows();

		return true;
	}


    bool AppKeyframes::selectKeyframes(const Settings &cfg)
    {
        if (!sensor_ || cfg.empty())
            return false;

        const std::string keyframes_file = cfg.get<std::string>("filename");
        std::cout << "keyframes_file " << keyframes_file << std::endl;
        const int keyframe_selection_window = cfg.get<int>("window_size");
        std::cout << "keyframe_selection_window " << keyframe_selection_window << std::endl;
        if (keyframes_file.empty() || keyframe_selection_window == 0)
            return false;

        KeyframeSelection keyframe_selection(keyframe_selection_window);

        // select keyframes
        for (int i = 0; i < sensor_->numFrames(); ++i)
        {
            if (i % 50 == 0)
                std::cout << "Keyframe selection frame " << i << "... " << std::endl;
            keyframe_selection.add(sensor_->color(i));
        }
        keyframe_selection.selectKeyframes();

        // save selected keyframes in file
        keyframe_selection.save(keyframes_file);

        // show selected keyframes
        if (cfg.get<bool>("show_keyframes"))
        {
            for (int i = 0; i < sensor_->numFrames(); ++i)
            {
                if (!keyframe_selection.isKeyframe(i))
                    continue;
                cv::Mat color_score = sensor_->color(i).clone();
                keyframe_selection.drawScore(i, color_score);
                std::string window_name = "keyframe " + std::to_string(i);
                cv::imshow(window_name, color_score);
                cv::waitKey();
                cv::destroyWindow(window_name);
            }
        }

        return true;
    }

} // namespace nv


int main(int argc, char *argv[])
{
    // run keyframe selection
    nv::AppKeyframes app;
	app.run(argc, argv);
    
    return 0;
}
