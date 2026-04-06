/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		size_t scan_size;
		float* depths;
		char* scanning_space;
		bool* clamped;
		int* internal_radii;
		float2* rects2D;
		float2* means2D;
		float* cov3D;
		float* view2gaussian; // GOF
		float4* cov3D_inv = nullptr;
		float4* conic_opacity;
		float* rgb;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static GeometryState fromChunk(char*& chunk, size_t P, bool requires_cov3D_inv);
	};

	struct ImageState
	{
		uint2* ranges;
		uint2* point_ranges;
		uint32_t* tile_launch_ranges;
		uint32_t* n_contrib;
		float* accum_alpha;
		float* center_depth;
		float4* center_alphas;

		static ImageState fromChunk(char*& chunk, size_t N);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static BinningState fromChunk(char*& chunk, size_t P);
	};

	struct PointBinningState
	{
		size_t sorting_size;
		uint64_t* point_list_keys_unsorted;
		uint64_t* point_list_keys;
		uint32_t* point_list_unsorted;
		uint32_t* point_list;
		char* list_sorting_space;

		static PointBinningState fromChunk(char*& chunk, size_t PN);
	};

	struct PointState
	{
		size_t scan_size;
		float* depths;
		float2* points2D;
		char* scanning_space;
		uint32_t* point_offsets;
		uint32_t* tiles_touched;

		static PointState fromChunk(char*& chunk, size_t P);
	};

	template<typename T, typename... Args> 
	size_t required(size_t P, Args... args)
	{
		char* size = nullptr;
		T::fromChunk(size, P, args...);
		return ((size_t)size) + 128;
	}

	struct Timer 
	{
		const bool total = true;
		const int interval = 128;
		bool active = false;
		
		int counter = 0;
		int current = 0;

		std::vector<cudaEvent_t> events;
		std::vector<float> timings;
		std::vector<std::string> names;

	public:
		Timer(std::initializer_list<std::string> l, int interval = 128) : names{ l }, interval{ interval }
		{
			events.resize(names.size() + 1);
			timings.resize(names.size(), 0.0f);
			for (auto& e : events)
				cudaEventCreate(&e);
		}
		~Timer()
		{
			for (auto& e : events)
				cudaEventDestroy(e);
		}
		void addTimePoint()
		{
			if (active)
			{
				cudaEventRecord(events[current++]);
			}
		}
		void syncAddReport(std::vector<std::pair<std::string, float>>& out_timings, bool force = false)
		{
			if (active && current)
			{
				cudaEventSynchronize(events[current - 1]);
				for (size_t i = 0; i < std::min<size_t>(current - 1, names.size()); ++i) 
				{
					float t;
					cudaEventElapsedTime(&t, events[i], events[i + 1]);
					timings[i] += t;
				}
				current = 0;
				if (force || ++counter == interval)
				{
					float tsum = 0;
					float inv = 1.0f / counter;
					for (size_t i = 0; i < names.size(); ++i)
					{
						float v = timings[i] * inv;
						out_timings.push_back({names[i], v});
						tsum += v;
					}
					if (total)
					{
						out_timings.push_back({"Total", tsum});
					}
					for (auto& t : timings)
						t = 0;
					counter = 0;
				}
			}
		}
		void operator() ()
		{
			addTimePoint();
		}
		void setActive(bool active)  { this->active = active; }
	};
};