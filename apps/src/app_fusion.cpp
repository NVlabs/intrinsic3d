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

#include <nv/app_fusion.h>

#include <iostream>

#include <nv/mat.h>
#include <opencv2/core.hpp>

#include <nv/filesystem.h>
#include <nv/keyframe_selection.h>
#include <nv/mesh.h>
#include <nv/mesh/marching_cubes.h>
#include <nv/rgbd/sensor.h>
#include <nv/rgbd/processing.h>
#include <nv/sdf/algorithms.h>
#include <nv/sparse_voxel_grid.h>


namespace nv
{

    AppFusion::AppFusion() :
        sensor_(nullptr)
	{
	}


    AppFusion::~AppFusion()
	{
		delete sensor_;
	}
	

    bool AppFusion::run(int argc, char *argv[])
	{
		// parse command line arguments
        // command line usage:
        // <app>
        //       -s "<path>/sensor.yml"
        //       -f "<path>/fusion.yml"
        const char *keys = {
            "{s sensor| |sensor config file}"
            "{f fusion| |fusion config file}"
        };
		cv::CommandLineParser cmd(argc, argv, keys);
        std::string sensor_cfg_file = cmd.get<std::string>("sensor");
        std::string fusion_cfg_file = cmd.get<std::string>("fusion");

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

        // generate SDF
        Settings fusion_cfg(fusion_cfg_file);
        if (!fuseSDF(fusion_cfg))
        {
            std::cerr << "SDF fusion failed!" << std::endl;
            return false;
        }

		// close all opencv windows
		cv::destroyAllWindows();

		return true;
	}


    bool AppFusion::fuseSDF(const Settings &fusion_cfg)
    {
        if (!sensor_ || fusion_cfg.empty())
            return false;

        // init keyframe selection for fusion
        const std::string keyframes_file = fusion_cfg.get<std::string>("keyframes");
        KeyframeSelection* keyframe_selection = nullptr;
        if (!keyframes_file.empty())
        {
            keyframe_selection = new KeyframeSelection();
            if (!keyframe_selection->load(keyframes_file))
                std::cerr << "Could not load keyframes ..." << std::endl;
        }

        // create voxel grid
        SparseVoxelGrid<Voxel>* grid = SparseVoxelGrid<Voxel>::create(
            fusion_cfg.get<float>("voxel_size"), sensor_->depthMin(), sensor_->depthMax());
        if (!grid)
        {
            std::cerr << "Could not create voxel grid!" << std::endl;
            return false;
        }

        // set clip bounds to limit memory usage
        Vec6f clip_bounds;
        clip_bounds[0] = fusion_cfg.get<float>("clip_x0");
        clip_bounds[1] = fusion_cfg.get<float>("clip_x1");
        clip_bounds[2] = fusion_cfg.get<float>("clip_y0");
        clip_bounds[3] = fusion_cfg.get<float>("clip_y1");
        clip_bounds[4] = fusion_cfg.get<float>("clip_z0");
        clip_bounds[5] = fusion_cfg.get<float>("clip_z1");
        if (clip_bounds.norm() > 0.0f)
            grid->setClipBounds(clip_bounds);
        // print sdf info
        grid->printInfo();

        // fuse rgb-d frames into sdf
        const int erode_size = fusion_cfg.get<int>("discont_window_size");
        std::cout << "Fusion..." << std::endl;
        for (int i = 0; i < sensor_->numFrames(); ++i)
        {
            if (keyframe_selection && !keyframe_selection->isKeyframe(i))
                continue;

            std::cout << "   integrating frame " << i << "... " << std::endl;
            cv::Mat depth = sensor_->depth(i);

            // discard depth values close to depth discontinuities
            if (erode_size > 0)
                depth = erodeDiscontinuities(depth, erode_size);
            // compute normals from depth
            cv::Mat normals = computeNormals(sensor_->depthCamera().intrinsics(), depth);
            // integrate frame into SDF
            grid->integrate(sensor_->depthCamera(),
                            sensor_->colorCamera(),
                            depth, sensor_->color(i),
                            normals, sensor_->pose(i));
        }

        // correct signed distance field using distance transform
        std::cout << "correct SDF ..." << std::endl;
        SDFAlgorithms::correctSDF(grid);

        // clear invalid voxels
        std::cout << "clear invalid voxels ..." << std::endl;
        SDFAlgorithms::clearInvalidVoxels(grid);

        // save sdf volume
        std::cout << "Saving SDF (" << grid->numVoxels() << " voxels) ..." << std::endl;
        const std::string sdf_file = fusion_cfg.get<std::string>("output_sdf");
        if (!sdf_file.empty() && !grid->save(sdf_file))
            std::cerr << "Could not save SDF volume to file ..." << std::endl;

        // extract and save mesh
        std::cout << "Saving mesh ..." << std::endl;
        const std::string output_mesh_file = fusion_cfg.get<std::string>("output_mesh");
        if (!output_mesh_file.empty())
        {
            // extract and save mesh
            Mesh* mesh = MarchingCubes<Voxel>::extractSurface(*grid);
            if (!mesh)
                std::cerr << "Mesh could not be generated!" << std::endl;
            else if (!mesh->save(output_mesh_file))
                std::cerr << "Mesh could not be saved!" << std::endl;
            delete mesh;
        }

        // clean up
        delete grid;
        delete keyframe_selection;

        return true;
    }

} // namespace nv


int main(int argc, char *argv[])
{
    // run fusion
    nv::AppFusion app;
	app.run(argc, argv);
    
    return 0;
}
