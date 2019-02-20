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

#include <nv/mesh/util.h>

#include <iostream>

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>


namespace nv
{
namespace MeshUtil
{

    void removeLooseComponents(Mesh* mesh)
    {
        const std::vector<Vec3f> &verts = mesh->vertices;
        std::vector<Vec3i> &face_indices = mesh->face_vertices;
        size_t num_faces = face_indices.size();

        // collapse vertices that have the same 3D location and store their connected faces
        typedef boost::tuple<float, float, float> vec3;
        std::map<vec3, std::vector<size_t> > verts_collapsed;
        for (size_t i = 0; i < num_faces; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                Vec3f v = verts[face_indices[i][j]];
                vec3 p = boost::tuples::make_tuple(v[0], v[1], v[2]);
                verts_collapsed[p].push_back(i);
            }
        }

        // build graph
        typedef boost::adjacency_list <boost::vecS, boost::vecS, boost::undirectedS> Graph;
        Graph G;
        for (std::map<vec3, std::vector<size_t> >::iterator itr = verts_collapsed.begin(); itr != verts_collapsed.end(); ++itr)
        {
            std::vector<size_t> &face_list = itr->second;
            for (size_t j = 0; j < face_list.size(); ++j)
            {
                boost::add_edge(face_list[j], face_list[(j + 1) % face_list.size()], G);
            }
        }

        // compute connected components
        std::vector<int> components(boost::num_vertices(G));
        int num_components = boost::connected_components(G, &components[0]);
        std::vector<int> component_sizes;
        component_sizes.resize(num_components, 0);
        //std::cout << "Total number of components: " << numComponents << std::endl;
        for (size_t i = 0; i != components.size(); ++i)
            component_sizes[components[i]]++;
        size_t largest = std::max_element(component_sizes.begin(), component_sizes.end()) - component_sizes.begin();
        //std::cout << "largest is " << largest << " with " << component_sizes[largest] << " out of " << num_faces << " triangles" << std::endl;

        // store old face indices and clear output
        std::vector<Vec3i> face_indices_old(face_indices);
        face_indices.clear();

        // only add faces from largest component to new output faces
        for (int i = 0; i < num_faces; ++i)
            if (components[i] == largest)
                face_indices.push_back(face_indices_old[i]);
        //std::cout << "triangles before: " << face_indices_old.size() << "; triangles after: " << face_indices.size() << std::endl;

        // remove unused vertices
        removeUnusedVertices(mesh);
    }


    void removeUnusedVertices(Mesh* mesh)
    {
        std::vector<Vec3f> &verts = mesh->vertices;
        std::vector<Vec3b> &colors = mesh->colors;
        std::vector<Vec3i> &face_indices = mesh->face_vertices;
        bool has_colors = !colors.empty();
        //std::cout << "Vertices before removing unused: " << verts.size() << std::endl;

        // detect used vertices
        std::vector<bool> verts_used;
        verts_used.resize(verts.size(), false);
        for (size_t i = 0; i < face_indices.size(); i++)
        {
            for (size_t t = 0; t < 3; ++t)
            {
                unsigned int v_idx = static_cast<unsigned int>(face_indices[i][t]);
                verts_used[v_idx] = true;
            }
        }

        // count unused vertices
        size_t num_unused = 0;
        for (size_t i = 0; i < verts_used.size(); ++i)
            if (!verts_used[i])
                ++num_unused;
        //std::cout << "Unused vertices: " << num_unused << std::endl;
        if (num_unused == 0)
            return;

        // remove unused vertices
        std::vector<Vec3f> verts_new;
        std::vector<Vec3b> colors_new;
        unsigned int idx = 0;
        std::vector<unsigned int> vert_indices;
        vert_indices.resize(verts.size());
        for (size_t i = 0; i < verts.size(); ++i)
        {
            if (verts_used[i])
            {
                verts_new.push_back(verts[i]);
                if (has_colors)
                    colors_new.push_back(colors[i]);
                vert_indices[i] = idx;
                ++idx;
            }
        }
        // set updated vertices
        mesh->vertices.clear();
        mesh->colors.clear();
        for (size_t i = 0; i < verts_new.size(); ++i)
        {
            mesh->vertices.push_back(verts_new[i]);
            if (has_colors)
                mesh->colors.push_back(colors_new[i]);
        }

        // update face indices (indices may have shifted)
        for (size_t i = 0; i < face_indices.size(); ++i)
        {
            for (size_t t = 0; t < 3; ++t)
            {
                unsigned int vIdx = static_cast<unsigned int>(face_indices[i][t]);
                face_indices[i][t] = vert_indices[vIdx];
            }
        }

        //std::cout << "Vertices after removing unused: " << mesh->vertices.size() << std::endl;
    }


    bool removeDegenerateFaces(Mesh* mesh)
    {
        if (!mesh)
            return false;

        // remove degenerate faces
        const std::vector<Vec3f> &verts = mesh->vertices;
        std::vector<Vec3i> faces = mesh->face_vertices;
        mesh->face_vertices.clear();
        for (size_t i = 0; i < faces.size(); ++i)
        {
            int v0 = faces[i][0];
            int v1 = faces[i][1];
            int v2 = faces[i][2];
            // check if two vertices have the same indices
            if ((v0 == v1) || (v0 == v2) || (v1 == v2))
                continue;
            // check if triangle area is zero
            Vec3f e0 = verts[v2] - verts[v0];
            Vec3f e1 = verts[v2] - verts[v1];
            double area = static_cast<double>((e0.cross(e1)).norm());
            if (area == 0.0 || std::isnan(area) || std::isinf(area))
                continue;
            mesh->face_vertices.push_back(Vec3i(v0, v1, v2));
        }
        return true;
    }

} // namespace MeshUtil
} // namespace nv
