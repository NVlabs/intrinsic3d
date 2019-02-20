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

#include <nv/mesh.h>

#include <iostream>
#include <fstream>
#include <vector>

namespace nv
{

    bool Mesh::save(const std::string &filename) const
    {
        if (filename.empty() || vertices.empty())
            return false;

        std::ofstream ply_file;
        ply_file.open(filename.c_str(), std::ios::binary);
        if (!ply_file.is_open())
            return false;

        // write ply file header
        bool has_colors = !colors.empty();
        int num_pts = static_cast<int>(vertices.size());
        int num_faces = static_cast<int>(face_vertices.size());
        ply_file << "ply" << std::endl;
        ply_file << "format binary_little_endian 1.0" << std::endl;
        ply_file << "element vertex " << num_pts << std::endl;
        ply_file << "property float x" << std::endl;
        ply_file << "property float y" << std::endl;
        ply_file << "property float z" << std::endl;
        if (has_colors)
        {
            ply_file << "property uchar red" << std::endl;
            ply_file << "property uchar green" << std::endl;
            ply_file << "property uchar blue" << std::endl;
        }
        ply_file << "element face " << num_faces << std::endl;
        ply_file << "property list uchar int vertex_indices" << std::endl;
        ply_file << "end_header" << std::endl;

        // write ply data (binary format)

        // write vertices
        for (size_t i = 0; i < vertices.size(); ++i)
        {
            // write vertex
            Vec3f v = vertices[i].cast<float>();
            ply_file.write((const char*)&v[0], sizeof(float) * 3);

            if (has_colors)
            {
                // write color
                ply_file.write((const char*)&colors[i][0], sizeof(unsigned char) * 3);
            }
        }

        // write faces
        unsigned char num_indices_per_face = 3;
        for (size_t i = 0; i < face_vertices.size(); ++i)
        {
            // face indices
            ply_file.write((const char*)&num_indices_per_face, sizeof(unsigned char));
            Eigen::Matrix<int, 3, 1> face_ind = face_vertices[i].cast<int>();
            ply_file.write((const char*)&face_ind[0], num_indices_per_face*sizeof(int));
        }

        ply_file.close();

        return true;
    }

} // namespace nv
