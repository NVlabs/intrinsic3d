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

#include <nv/mat.h>
#include <nv/mesh.h>
#include <nv/sparse_voxel_grid.h>

#include "omp.h"


namespace nv
{

    /**
     * @brief   Implementation of the Marching Cubes algorithm for
     *          extracting the Iso-Surface from a Sparse Voxel Grid
     * @author  Robert Maier <robert.maier@tum.de>
     * @author  Matthias Niessner <niessner@tum.de>
     */
	template <class T>
	class MarchingCubes
	{
	public:
		static Mesh* extractSurface(const SparseVoxelGrid<T>& grid);

	private:
		MarchingCubes();
		MarchingCubes(const MarchingCubes&);
		MarchingCubes& operator=(const MarchingCubes&);
		~MarchingCubes();

		struct Vertex
		{
			Vec3f p;
			Vec3f c;
		};

		struct Triangle
		{
			Vertex v0;
			Vertex v1;
			Vertex v2;
		};

		Mesh* extractMesh(const SparseVoxelGrid<T>& grid);
        Mesh* merge(const std::vector< std::vector<Triangle> > &results) const;

        void extractSurfaceAt(const Vec3i& voxel, const SparseVoxelGrid<T>& grid, std::vector<Triangle>& result);
        int computeLutIndex(const SparseVoxelGrid<T>& grid, int i, int j, int k, float iso_value);
        Vec3f interpolate(float tsdf0, float tsdf1, const Vec3f &val0, const Vec3f &val1, float iso_value);
        typename MarchingCubes::Vertex getVertex(const SparseVoxelGrid<T>& grid, int i1, int j1, int k1, int i2, int j2, int k2, float iso_value);
		
        const static int edge_table_[256];
        const static int triangle_table_[256][16];
	};

} // namespace nv
