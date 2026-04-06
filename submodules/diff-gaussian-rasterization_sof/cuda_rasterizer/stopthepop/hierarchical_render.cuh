/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

#pragma once

#include "../auxiliary.h"
#include "stopthepop_common.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

template<typename T, size_t S>
__device__ void initArray(T(&arr)[S], T v = 0)
{
#pragma unroll
	for (int i = 0; i < S; ++i)
	{
		arr[i] = v;
	}
}

template<int32_t NUM, typename CG, typename KT, typename VT>
__device__ void mergeSortRegToSmem(CG& cg, KT* keys, VT* values, KT* fin_keys, VT* fin_values, KT key, VT value)
{
	// binary search to find location
	int32_t s0 = 0;
#pragma unroll
	for (int32_t i = NUM / 2; i > 0; i /= 2)
	{
		if (keys[s0 + i] <= key)
			s0 += i;
	}
	// move one ahead
	s0 += 1;
	if (keys[0] > key)
		s0 = 0;
	// how many threads of my group are ahead of me
	s0 += cg.thread_rank();

	// reverse the search
	auto store_key = keys[cg.thread_rank()];
	keys[cg.thread_rank()] = key;
	cg.sync();

	// binary search to find location
	int32_t s1 = 0;
#pragma unroll
	for (int32_t i = NUM / 2; i > 0; i /= 2)
	{
		if (keys[s1 + i] < store_key)
			s1 += i;
	}
	// move one ahead
	s1 += 1;
	if (keys[0] >= store_key)
		s1 = 0;
	// how many threads of my group are ahead of me
	s1 += cg.thread_rank();
	auto store_value = values[cg.thread_rank()];
	cg.sync();

	// write out the new order
	fin_keys[s0] = key;
	fin_keys[s1] = store_key;
	fin_values[s0] = value;
	fin_values[s1] = store_value;
	cg.sync();
}

// only works for low number of threads, (WINDOW + THREADS) must be smaller than 2 ^ (32 / THREADS)
template<int32_t THREADS, int32_t WINDOW, typename CG, typename KT, typename VT, typename AF>
__device__ void mergeSortInto(CG& cg, int32_t rank, KT key, VT value, KT* keys, VT* values, AF&& access_function)
{
	// binary search to find location, bias for end
	int loc = WINDOW;
	if (key < keys[access_function(WINDOW - 1)])
	{
		loc = WINDOW - 1;
#pragma unroll
		for (int32_t i = WINDOW / 2; i > 0; i /= 2)
		{
			if (key < keys[access_function(loc - i)])
				loc -= i;
		}
	}
	loc += rank;

	// combined information for all locations, so we can trivially relocate 
	constexpr uint32_t BITS_PER_INFO = 32 / THREADS;
	uint32_t comb_loc = loc << (BITS_PER_INFO * (THREADS - rank - 1));
#pragma unroll
	for (int i = THREADS / 2; i >= 1; i /= 2)
	{
		comb_loc += cg.shfl_xor(comb_loc, i);
	}

	constexpr uint32_t MASK = (0x1 << BITS_PER_INFO) - 1;

	int first_offset = ((comb_loc >> (BITS_PER_INFO * (THREADS - 1))) & MASK) / THREADS * THREADS;
	int move_offset = 4;
	for (int read_from = WINDOW - THREADS + rank; read_from >= first_offset; read_from -= 4)
	{
		while (move_offset > 0 && (comb_loc & MASK) >= read_from + move_offset)
		{
			--move_offset;
			comb_loc = comb_loc >> BITS_PER_INFO;
		}
		
		int read_access = access_function(read_from);
		KT key_move = keys[read_access];
		VT value_move = values[read_access];
		cg.sync();
		if (move_offset > 0)
		{
			int write_access = access_function(read_from + move_offset);
			keys[write_access] = key_move;
			values[write_access] = value_move;
		}
	}
	cg.sync();
	// write my data
	int write_access = access_function(loc);
	keys[write_access] = key;
	values[write_access] = value;
}

template<int N, typename CG, typename KT>
__device__ int shflRankingLocal(CG& cg, int rank, KT key)
{
	// quick ranking with N-1 shfl
	int count = 0;
#pragma unroll
	for (int i = 1; i < N; ++i)
	{
		int other_rank = (rank + i) % N;
		auto other_key = cg.shfl(key, other_rank);
		if (other_key < key ||
			(other_key == key && other_rank < rank))
		{
			++count;
		}
	}
	return count;
}

template<int N, typename CG, typename KT, typename VT>
__device__ void shflSortLocal2Shared(CG& cg, int rank, KT key, VT val, KT* keys, VT* vals)
{
	// quick ranking with 3 shfl
	int count = shflRankingLocal<N>(cg, rank, key);
	keys[count] = key;
	vals[count] = val;
}

// TODO: can we do a better implementation?
template<uint32_t NUM_VALS, typename CG, typename KT, typename VT>
__device__ void batcherSort(CG& cg, KT* keys, VT* vals)
{
	for (uint32_t size = 2; size <= NUM_VALS; size *= 2)
	{
		uint32_t stride = size / 2;
		uint32_t offset = cg.thread_rank() & (stride - 1);

		{
			cg.sync();
			uint32_t pos = 2 * cg.thread_rank() - (cg.thread_rank() & (stride - 1));
			if (keys[pos + 0] > keys[pos + stride])
			{
				swap(keys[pos + 0], keys[pos + stride]);
				swap(vals[pos + 0], vals[pos + stride]);
			}
			stride /= 2;
		}

		for (; stride > 0; stride /= 2)
		{
			cg.sync();
			uint32_t pos = 2 * cg.thread_rank() - (cg.thread_rank() & (stride - 1));

			if (offset >= stride)
			{
				if (keys[pos - stride] > keys[pos + 0])
				{
					swap(keys[pos - stride], keys[pos + 0]);
					swap(vals[pos - stride], vals[pos + 0]);
				}
			}
		}
	}
}



// 0x1 -> tail
// 0x2 -> mid
// 0x4 -> front
// 0x8 -> blend
// 0x10 -> counters and flow
// 0x20 -> general
// 0x40 -> culling
// 0x100 -> select block
// 0x200 -> print info along ray for pixel
#define DEBUG_HIERARCHICAL 0x0

// MID_WINDOW needs to be pow2+4, minimum 8
template <int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA, typename PF, typename SF, typename BF, typename FF>
__device__ void sortGaussiansRayHierarchicaEvaluation(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* view2gaussian,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float4* __restrict__ conic_opacity,
	DebugVisualization debugType,
	PF && prep_function,
	SF && store_function,
	BF && blend_function,
	FF && fin_function)
{
#if (DEBUG_HIERARCHICAL & 0x100) != 0
	//if (blockIdx.x != 7 || blockIdx.y != 7)
	//	return;
	constexpr uint2 target = { 426, 55 };
	if (blockIdx.x != target.x / 16 || blockIdx.y != target.y / 16)
		return;
	uint2 rem = { target.x % 16, target.y % 16 };
	if (cg::this_thread_block().thread_rank() / 32 != rem.x / 8 + (rem.y / 4) * 2)
		return;

	//if (blockIdx.x != 0 || blockIdx.y != 3)
	//	return;
	//if (threadIdx.z != 0)
	//	return;

#endif


	// block size must be: 16,4,4

	// we use the following thread setup per warp
	// 00 01 04 05 16 17 20 21
	// 02 03 06 07 18 19 22 23
	// 08 09 12 13 24 25 28 29
	// 10 11 14 15 26 27 30 31

	// and the following warp setup (which does not matter)
	// 00 01 
	// 02 03 
	// 04 05
	// 06 07

	// and the following half warp setup 
	// 00 01 02 03 
	// 04 05 06 07
	// 08 09 10 11
	// 12 13 14 15

	// every half warp (4x4) block has one smem sort window (32 elements sorted + 32 elements buffer for loading)
	// every 2x2 block has its own 8 element buffer window for local resorting
	// every thread has its own sorted head list typically 4 elements

	// block.thread_index().y/z  identifies the 4x4 tile
	//
	constexpr int PerThreadSortWindow = HEAD_WINDOW;
	constexpr int MidSortWindow = MID_WINDOW;

	// head sorting setup
	float head_depths[PerThreadSortWindow];

	// GOF: stuff
	float3 head_normals[PerThreadSortWindow];
	float4 head_ABCs[PerThreadSortWindow];

	[[maybe_unused]] const uint2 _t1{0, 0};
	decltype(store_function(_t1, 0, 0.0f, 0.0f, 0.0f)) head_stores[PerThreadSortWindow];
	int head_ids[PerThreadSortWindow];

	// mid sorting setup
	__shared__ float mid_depths[4][4][4][MidSortWindow];
	__shared__ int mid_ids[4][4][4][MidSortWindow];
	[[maybe_unused]] uint32_t mid_front = 0;
	[[maybe_unused]] auto mid_access = [&](uint32_t offset)
		{
			return (mid_front + offset) % MidSortWindow;
		};

	// tail sorting setup
	__shared__ float tail_depths[4][4][64];
	__shared__ int tail_ids[4][4][64];

	// tail viewdir is 0, mid viewdirs are 1-4
	__shared__ float3 tail_and_mid_viewdir[4][4][5];

	// global helper
	__shared__ uint2 range;

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<WARP_SIZE>(block);
	auto halfwarp = cg::tiled_partition<WARP_SIZE / 2>(block);
	auto head_group = cg::tiled_partition<4>(halfwarp);


	// initialize head structure
	initArray(head_depths, FLT_MAX);
	initArray(head_stores);
	initArray(head_ids, -1);

	uint32_t fill_counters = 0; // HEAD 8 bit, MID 8 bit, TAIL 16 bit, 
	[[maybe_unused]] constexpr uint32_t FillHeadMask = 0xFF000000;
	constexpr uint32_t FillHeadOne = 0x1000000;
	constexpr uint32_t FillMidMask = 0xFF0000;
	constexpr uint32_t FillMidOne = 0x10000;
	constexpr uint32_t FillTailMask = 0xFFFF;
	constexpr uint32_t FillTailOne = 0x1;

	// initialize ray directions
	const uint2 tile_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 tail_corner = { tile_min.x + 4 * block.thread_index().y, tile_min.y + 4 * block.thread_index().z };

	const glm::mat4 inverse_vp = loadMatrix4x4(projmatrix_inv);
	const float3 campos = *cam_pos;

	if (block.thread_index().x < 5)
	{
		// first thread computes the tail view dir, next 4 the mid view dir
		float2 pos = { tail_corner.x + 0.5f, tail_corner.y + 0.5f };
		if (block.thread_index().x == 0)
		{
			pos.x += 1.5f;
			pos.y += 1.5f;
		}
		else
		{
			pos.x += 0.5f + 2 * ((block.thread_index().x - 1) % 2);
			pos.y += 0.5f + 2 * ((block.thread_index().x - 1) / 2);
		}
		float3 dir = computeViewRay(inverse_vp, campos, pos, W, H);
		tail_and_mid_viewdir[block.thread_index().y][block.thread_index().z][block.thread_index().x] = dir;
#if (DEBUG_HIERARCHICAL & 0x2F) != 0 && (DEBUG_HIERARCHICAL & 0x100) != 0
		printf("group dir %d - %d %d %d  - pix %f %f dir %f %f %f\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
			pos.x, pos.y, dir.x, dir.y, dir.z);
#endif
	}

	const int midid = halfwarp.thread_rank() / 4;
	const int midrank = halfwarp.thread_rank() % 4;
	const int midy = midid / 2;
	const int midx = midid % 2;

	const int heady = midrank / 2;
	const int headx = midrank % 2;

	const uint2 pixpos = { tail_corner.x + midx * 2 + headx, tail_corner.y + midy * 2 + heady };
	bool active = pixpos.x < W && pixpos.y < H;

	// do it exactly as GOF (+0.5f)
	const float2 pixf = { pixpos.x + 0.5f, pixpos.y + 0.5f };
	float2 ray = { 
		(pixf.x - W/2.) / focal_x, 
		(pixf.y - H/2.) / focal_y 
	};
	const float3 viewdir = computeViewRay(inverse_vp, campos, pixf, W, H);
#if (DEBUG_HIERARCHICAL & 0x2F) != 0 && (DEBUG_HIERARCHICAL & 0x100) != 0
	printf("own dir %d - %d %d %d  - pix %d %d dir %f %f %f\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
		pixpos.x, pixpos.y, viewdir.x, viewdir.y, viewdir.z);
#endif
	// setup helpers
	const int32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	if (warp.thread_rank() == 0)
	{
		range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	}
	

	if constexpr (MidSortWindow != 8)
	{
		for (int i = 0; i < MidSortWindow; i += 4)
		{
			mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][i + warp.thread_rank() % 4] = FLT_MAX;
			mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][i + warp.thread_rank() % 4] = -1;
		}
	}

	// ensure all helpers are visible
	warp.sync();

	float3 r{ray.x, ray.y, 1.0f};
	// thread state variables
	auto blend_data = prep_function(active, pixpos, r );
	
	// lambdas controlling the behavior
	auto blend_one = [&]()
		{
			fill_counters -= FillHeadOne;

			if (!active)
				return;
			//float* view2gaussian_j = ;
			if (!blend_function(pixpos, blend_data, head_ids[0], head_stores[0], head_depths[0], &view2gaussian[head_ids[0] * 10], ray, debugType, head_normals[0], head_ABCs[0]))
			{
				active = false;
				return;
			}

#if (DEBUG_HIERARCHICAL & 0x8) != 0
#if (DEBUG_HIERARCHICAL & 0x100)
			if (pixpos.x == target.x && pixpos.y == target.y)
#endif
				printf("%d - %d %d - blending: %f %d %f (%d %d %d)\n", warp.thread_rank(), pixpos.x, pixpos.y,
					head_depths[0], head_ids[0], head_stores[0],
					(fill_counters & FillHeadMask) / FillHeadOne,
					(fill_counters & FillMidMask) / FillMidOne,
					(fill_counters & FillTailMask) / FillTailOne);
#endif

			for (int i = 1; i < PerThreadSortWindow; ++i)
			{
				head_depths[i - 1] = head_depths[i];
				head_stores[i - 1] = head_stores[i];
				head_ids[i - 1] = head_ids[i];
				head_normals[i - 1] = head_normals[i];
				head_ABCs[i - 1] = head_ABCs[i];
			}
			head_depths[PerThreadSortWindow - 1] = FLT_MAX;
		};



	auto front4OneFromMid = [&](bool checkvalid)
		{
			if (head_group.any(active))
			{
				// prepare depth and data for shfl
				float3 mid_depth_info[3];
				float4 mid_conic_opacity;
				float2 mid_point_xy;

				int load_id;
				if constexpr (MidSortWindow == 8)
				{
					load_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4];
				}
				else
				{
					load_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][mid_access(warp.thread_rank() % 4)];
				}
				if (!checkvalid || load_id != -1)
				{
					mid_depth_info[0] = make_float3(cov3Ds_inv[3 * load_id]);
					mid_depth_info[1] = make_float3(cov3Ds_inv[3 * load_id + 1]);
					mid_depth_info[2] = make_float3(cov3Ds_inv[3 * load_id + 2]);
					mid_conic_opacity = conic_opacity[load_id];
					mid_point_xy = points_xy_image[load_id];
				}

#if (DEBUG_HIERARCHICAL & 0x4) != 0
				printf("%d - %d %d - head loading %d\n", warp.thread_rank(), pixpos.x, pixpos.y, load_id);
#endif
				// inner always takes out up to four elements from mid
				for (int inner = 0; inner < 4; ++inner)
				{
					// the head is left most, so this checks for a full sort window
					if (fill_counters >= FillHeadOne * PerThreadSortWindow)
					{
						blend_one();
					}

					// take one from mid
					int coll_id;
					if constexpr (MidSortWindow == 8)
					{
						coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][inner];
					}
					else
					{
						coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][mid_access(inner)];
					}

					// every thread has the same id so this is safe
					if (checkvalid && coll_id == -1)
						continue;

					const float* V2G = view2gaussian + (coll_id * 10);
					float3 normal = {
						V2G[0] * ray.x + V2G[1] * ray.y + V2G[2], 
						V2G[1] * ray.x + V2G[3] * ray.y + V2G[4],
						V2G[2] * ray.x + V2G[4] * ray.y + V2G[5]
					};
					float4 con_o = head_group.shfl(mid_conic_opacity, inner);
					float4 ABC = {
						ray.x * normal.x + ray.y * normal.y + normal.z,
						2 * (V2G[6] * ray.x + V2G[7] * ray.y + V2G[8]),
						V2G[9],
						con_o.w
					};
					float depth = -ABC.y/(2.f * ABC.x);

#if (DEBUG_HIERARCHICAL & 0x4) != 0
					printf("%d - %d %d - %d new depth %f\n", warp.thread_rank(), pixpos.x, pixpos.y, coll_id, depth);
#endif
					if (!active || depth < NEAR_PLANE)
						continue;

					blend_data.contributor++;

					float min_value = -(ABC.y/ABC.x) * (ABC.y/4.f) + ABC.z;

					float power = -0.5f * min_value;
					if (power > 0.0f){
						power = 0.0f;
					}

					float G = exp(power);
					float alpha = min(0.99f, con_o.w * G);
					

#if (DEBUG_HIERARCHICAL & 0x4) != 0
					printf("%d - %d %d - %d %f alpha is %f\n", warp.thread_rank(), pixpos.x, pixpos.y, coll_id, depth, alpha);
#endif
					if (alpha < 1.0f / 255.0f)
						continue;

					auto store = store_function(pixpos, coll_id, G, alpha, depth);

					// push alpha and depth into per thread sorted array
#pragma unroll
					for (int s = 0; s < PerThreadSortWindow; ++s)
					{
						if (depth < head_depths[s])
						{
							swap(depth, head_depths[s]);
							swap(coll_id, head_ids[s]);
							swap(store, head_stores[s]);
							swap(normal, head_normals[s]);
							swap(ABC, head_ABCs[s]);
						}
					}
					fill_counters += FillHeadOne;

#if (DEBUG_HIERARCHICAL & 0x4) != 0
					printf("%d - %d %d - count: %d - sorted: %f %d - %f %d - %f %d - %f %d\n", warp.thread_rank(), pixpos.x, pixpos.y,
						fill_counters, head_depths[0], head_ids[0], head_depths[1], head_ids[1], head_depths[2], head_ids[2], head_depths[3], head_ids[3]);
#endif
				}
			}
			if constexpr (MidSortWindow != 8)
			{
				mid_front += 4;
			}
			fill_counters -= 4 * FillMidOne;
			halfwarp.sync();
		};

	auto pushPullThroughMid = [&](bool checkvalid)
		{
			// prepare depth for shfl
			float3 tail_depth_info[3];
			int load_id = tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x];
			if (!checkvalid || load_id != -1)
			{
				tail_depth_info[0] = make_float3(cov3Ds_inv[3 * load_id]);
				tail_depth_info[1] = make_float3(cov3Ds_inv[3 * load_id + 1]);
				tail_depth_info[2] = make_float3(cov3Ds_inv[3 * load_id + 2]);
			}
			else
			{
				tail_depth_info[0] = tail_depth_info[1] = tail_depth_info[2] = { 0,0,0 };
			}
#if (DEBUG_HIERARCHICAL & 0x2) != 0
			printf("%d - %d %d - mid loading %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, load_id);
#endif

			// take out 4 x 4 elements from tail and move into mid
			for (int mid = 0; mid < 4; ++mid)
			{
				// the tail is the same for everyone in the half warp
				if (checkvalid && (fill_counters & FillTailMask) == 0)
					break;

				// take 4 from tail to mid
				int tid = 4 * mid + (warp.thread_rank() % 4);
				int coll_id = tail_ids[block.thread_index().y][block.thread_index().z][tid];

				float depth = depthAlongRay(halfwarp.shfl(tail_depth_info[0], tid),
					halfwarp.shfl(tail_depth_info[1], tid),
					halfwarp.shfl(tail_depth_info[2], tid),
					tail_and_mid_viewdir[block.thread_index().y][block.thread_index().z][1 + block.thread_index().x / 4]);

				// note: we can only get invalid during draining here
				if (checkvalid && coll_id == -1)
				{
					depth = FLT_MAX;
				}

#if (DEBUG_HIERARCHICAL & 0x2) != 0
				printf("%d - %d %d %d - mid new depth %d %f (%f)\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, tid, coll_id, depth,
					tail_depths[block.thread_index().y][block.thread_index().z][tid]);
#endif
				if constexpr (MidSortWindow == 8)
				{
					// local sort first into front 4 slots (which are empty for sure)
					shflSortLocal2Shared<4>(head_group, warp.thread_rank() % 4, depth, coll_id,
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4]);
					head_group.sync();

					coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4];
					depth = mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4];

#if (DEBUG_HIERARCHICAL & 0x2) != 0
					printf("%d - %d %d %d - mid local %d sort %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, warp.thread_rank() % 4, coll_id, depth);
#endif


					// we do not need the exact count as we only got invalid during draining
					fill_counters += 4 * FillMidOne;

					// we are not culling here, so we always have data after the first
					// if ( (fill_counters & FillMidMask) > 4 * FillMidOne)
					if (mid != 0 || ((fill_counters & FillMidMask) > 4 * FillMidOne))
					{
						// sort mid					
						mergeSortRegToSmem<4>(head_group,
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4] + 4,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4] + 4,
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
							depth, coll_id);
						head_group.sync();
#if (DEBUG_HIERARCHICAL & 0x2) != 0
						printf("%d - %d %d %d - sorted into mid %d:  %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, warp.thread_rank() % 4,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4],
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4]);
						printf("%d - %d %d %d - sorted into mid %d:  %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, 4 + warp.thread_rank() % 4,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4],
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4]);
#endif
						front4OneFromMid(false);
					}
					else
					{
						// move mid
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4] = coll_id;
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4] = depth;
						// no need to sync here  as shfl of the next iteration will take care of it
					}
				}
				else
				{
					// local sort first using shfl
					int offset = shflRankingLocal<4>(head_group, warp.thread_rank() % 4, depth);
					uint32_t sort_mid_offset = mid_access(MidSortWindow - 4 + offset);
					uint32_t my_mid_offset = mid_access(MidSortWindow - 4 + warp.thread_rank() % 4);
					mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][sort_mid_offset] = coll_id;
					mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][sort_mid_offset] = depth;

					head_group.sync();

					coll_id = mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][my_mid_offset];
					depth = mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][my_mid_offset];

#if (DEBUG_HIERARCHICAL & 0x2) != 0
					printf("%d - %d %d %d - mid local %d sort %d %f \n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, warp.thread_rank() % 4, coll_id, depth);
#endif
					// we do not need the exact count as we only got invalid during draining
					fill_counters += 4 * FillMidOne;

					// merge sort with existing
					mergeSortInto<4, MidSortWindow - 4>(head_group, warp.thread_rank() % 4, depth, coll_id,
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4],
						mid_access);

#if (DEBUG_HIERARCHICAL & 0x2) != 0
					for (int j = 0; j < MidSortWindow; j += 4)
					{
						int access = mid_access(j + warp.thread_rank() % 4);
						printf("%d - %d %d %d - sorted into mid %d (%d from %d):  %d %f \n", 
							warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x / 4, 
							j + warp.thread_rank() % 4, access, mid_front,
							mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][access],
							mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][access]);
					}
#endif

					// run front if we are full
					if ((fill_counters & FillMidMask) > (MidSortWindow - 4) * FillMidOne)
					{
						front4OneFromMid(false);
					}
				}

				if (checkvalid)
				{
					fill_counters -= min(4 * FillTailOne, FillTailMask & fill_counters);
				}
			}
			if (!checkvalid)
			{
				fill_counters -= 16 * FillTailOne;
			}

		};

	// run through elements and continue to push in and blend out
	for (int progress = range.x; progress < range.y; progress += WARP_SIZE)
	{
		if (!warp.any(active))
			break;

#if DEBUG_HIERARCHICAL != 0
		//if (progress - range.x > 64)
		//	return;
#endif

		// fill new data into tail (last 32 elements) for both tail lists
		// and determine actual elements added and adjust count
		float4 in_conic_opacity;
		float2 in_point_xy;

		int load_id = -1;
		const int tid = progress + warp.thread_rank();
		if (tid < range.y)
		{
			load_id = point_list[tid];
		}

		if (load_id != -1 && CULL_ALPHA)
		{
			in_conic_opacity = conic_opacity[load_id];
			in_point_xy = points_xy_image[load_id];
		}

#if (DEBUG_HIERARCHICAL & 0x1) != 0
		printf("%d - %d %d - loading %d\n", warp.thread_rank(), pixpos.x, pixpos.y, load_id);
#endif

		uint32_t halfs_culled_mask = 0U;
		for (int half = 0; half < 2; ++half)
		{
			int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);
			if (load_id != -1)
			{
				// cull against tail tile
				if (CULL_ALPHA)
				{
					// tile boundaries
					const glm::vec2 tail_rect_min = { static_cast<float>(block.group_index().x * BLOCK_X + 4 * xid), static_cast<float>(block.group_index().y * BLOCK_Y + 4 * block.thread_index().z) };
					const glm::vec2 tail_rect_max = { tail_rect_min.x + 3.0f, tail_rect_min.y + 3.0f };

					glm::vec2 max_pos;
					float power = max_contrib_power_rect_gaussian_float<3, 3>(in_conic_opacity, in_point_xy, tail_rect_min, tail_rect_max, max_pos);

					float alpha = min(0.99f, in_conic_opacity.w * exp(-power));
					if (alpha < 1.0f / 255.0f)
						halfs_culled_mask |= (0x1U << half);
				}
			}
		}

		float3 in_depth_info[3];
		if (load_id != -1 && (!CULL_ALPHA || !(halfs_culled_mask == 0x3))) // if culling and not both halfs culled
		{
			in_depth_info[0] = make_float3(cov3Ds_inv[3 * load_id]);
			in_depth_info[1] = make_float3(cov3Ds_inv[3 * load_id + 1]);
			in_depth_info[2] = make_float3(cov3Ds_inv[3 * load_id + 2]);
		}	

		for (int half = 0; half < 2; ++half)
		{
			int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);
			float depth = FLT_MAX;

			if (load_id != -1)
			{
				// if not culled, compute depth
				if (!CULL_ALPHA || !(halfs_culled_mask & (0x1U << half)))
				{
					depth = depthAlongRay(in_depth_info[0], in_depth_info[1], in_depth_info[2], tail_and_mid_viewdir[xid][block.thread_index().z][0]);
				}
			}
			
			tail_depths[xid][block.thread_index().z][32 + warp.thread_rank()] = depth;
			tail_ids[xid][block.thread_index().z][32 + warp.thread_rank()] = depth == FLT_MAX ? -1 : load_id;
#if (DEBUG_HIERARCHICAL & 0x1) != 0
			printf("(%d) %d - %d %d %d - %d : %f\n", half, warp.thread_rank(), xid, block.thread_index().z, 0, load_id, depth);
#endif
		}
		// local sort the 32 elements with half warp from shared memory
		batcherSort<32>(halfwarp, tail_depths[block.thread_index().y][block.thread_index().z] + 32, tail_ids[block.thread_index().y][block.thread_index().z] + 32);
		// sync comes through shfl below

#if (DEBUG_HIERARCHICAL & 0x1) != 0
		printf("batcher sort %d/%d: %f %d\n", block.thread_index().y, halfwarp.thread_rank(), tail_depths[block.thread_index().y][block.thread_index().z][32 + halfwarp.thread_rank()], tail_ids[block.thread_index().y][block.thread_index().z][32 + halfwarp.thread_rank()]);
		printf("batcher sort %d/%d: %f %d\n", block.thread_index().y, 16 + halfwarp.thread_rank(), tail_depths[block.thread_index().y][block.thread_index().z][32 + 16 + halfwarp.thread_rank()], tail_ids[block.thread_index().y][block.thread_index().z][32 + 16 + halfwarp.thread_rank()]);
#endif

		for (int half = 0; half < 2; ++half)
		{
			// merge sort if we have old data
			if ((warp.shfl(fill_counters, half * 16) & FillTailMask) != 0)
			{
				int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);

				float* d = tail_depths[xid][block.thread_index().z];
				int* id = tail_ids[xid][block.thread_index().z];

				float k = d[32 + warp.thread_rank()];
				int v = id[32 + warp.thread_rank()];
				// determine number of valid
				uint32_t count_valid = __popc(warp.ballot(v != -1));
				if (half == warp.thread_rank() / 16)
					fill_counters += count_valid * FillTailOne;
				mergeSortRegToSmem<32>(warp, d, id, d, id, k, v);

#if (DEBUG_HIERARCHICAL & 0x1) != 0
				warp.sync();
				printf("merge of %d (%d) sort %d: %f %d\n", xid, (fill_counters & FillTailMask) / FillTailOne, warp.thread_rank(), d[warp.thread_rank()], id[warp.thread_rank()]);
				printf("merge of %d (%d) sort %d: %f %d\n", xid, (fill_counters & FillTailMask) / FillTailOne, 32 + warp.thread_rank(), d[32 + warp.thread_rank()], id[32 + warp.thread_rank()]);
#endif
			}
			else
			{
				// copy data to the front
				int xid = half == 0 ? (block.thread_index().y & (~0x1)) : (block.thread_index().y | 0x1);
				float* d = tail_depths[xid][block.thread_index().z];
				int* id = tail_ids[xid][block.thread_index().z];
				d[warp.thread_rank()] = d[32 + warp.thread_rank()];
				int v = id[32 + warp.thread_rank()];
				id[warp.thread_rank()] = v;
				// determine number of valid
				uint32_t count_valid = __popc(warp.ballot(v != -1));
				if (half == warp.thread_rank() / 16)
					fill_counters += count_valid * FillTailOne;

#if (DEBUG_HIERARCHICAL & 0x1) != 0
				warp.sync();
				printf("copied of %d (%d) data %d: %f %d\n", xid, (fill_counters & FillTailMask) / FillTailOne, warp.thread_rank(), d[warp.thread_rank()], id[warp.thread_rank()]);
#endif
			}
		}

		for (int half = 0; half < 2; ++half)
		{
			if ((fill_counters & FillTailMask) > 32 * FillTailOne)
			{
				// take 16 elements out from mid
				pushPullThroughMid(false);
				halfwarp.sync();

				// move current data in tail (max 48)
				for (int i = 0; i < 3 - half; ++i)
				{
					tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x + i * 16] =
						tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x + (i + 1) * 16];
					tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x + i * 16] =
						tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x + (i + 1) * 16];
				}
				halfwarp.sync();
			}

		}
	}

	// debug
#if (DEBUG_HIERARCHICAL & 0x10) != 0
	printf("%d - %d %d %d - draining tail with %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
		(fill_counters & FillHeadMask) / FillHeadOne, (fill_counters & FillMidMask) / FillMidOne, (fill_counters & FillTailMask) / FillTailOne);
#endif

	if (warp.any(active))
	{
		if ((fill_counters & FillTailMask) != 0)
		{
			for (int half = 0; half < 2; ++half)
			{
				pushPullThroughMid(true);
#if (DEBUG_HIERARCHICAL & 0x10) != 0
				printf("%d - %d %d %d - pulled from mid %d  with %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
					half, (fill_counters & FillHeadMask) / FillHeadOne, (fill_counters & FillMidMask) / FillMidOne, (fill_counters & FillTailMask) / FillTailOne);
#endif
				if ((half == 0) && (fill_counters & FillTailMask) == 0)
					break;


				// move current data in tail (max 16)
				if (half == 0)
				{
					tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x] =
						tail_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x + 16];
					tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x] =
						tail_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x + 16];
				}
			}
		}

		// drain the remainder from mid
		if (warp.any(active))
		{
			if constexpr (MidSortWindow == 8)
			{
				if ((fill_counters & FillMidMask) != 0)
				{
					// mid still has data, but it is not at the right location, so move it
					mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4] =
						mid_ids[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4];
					mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][warp.thread_rank() % 4] =
						mid_depths[block.thread_index().y][block.thread_index().z][block.thread_index().x / 4][4 + warp.thread_rank() % 4];


					front4OneFromMid(true);

#if (DEBUG_HIERARCHICAL & 0x10) != 0
					printf("%d - %d %d %d - pulled took 4 from mid %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
						(fill_counters & FillHeadMask) / FillHeadOne, (fill_counters & FillMidMask) / FillMidOne, (fill_counters & FillTailMask) / FillTailOne);
#endif

				}
			}
			else
			{
#if (DEBUG_HIERARCHICAL & 0x10) != 0
				int deb_counter = 0;
#endif
				while ((fill_counters & FillMidMask) != 0)
				{
					front4OneFromMid(true);
#if (DEBUG_HIERARCHICAL & 0x10) != 0
					printf("%d - %d %d %d - pulled (%d) took 4 from mid %d %d %d\n", warp.thread_rank(), block.thread_index().y, block.thread_index().z, block.thread_index().x,
						deb_counter, (fill_counters& FillHeadMask) / FillHeadOne, (fill_counters& FillMidMask) / FillMidOne, (fill_counters& FillTailMask) / FillTailOne);
					++deb_counter;
#endif
				}
			}
			// drain front
			while (active && fill_counters != 0)
			{
				blend_one();
			}
	}
}


	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.

	if (pixpos.x < W && pixpos.y < H)
	{
		float3 o = {cam_pos->x, cam_pos->y, cam_pos->z};
		fin_function(pixpos, blend_data, debugType, range.y - range.x, o);

	}
}


template <int32_t CHANNELS, int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA = true, bool EXACT_DEPTH = false, bool ENABLE_DEBUG_VIZ = false>
__global__ void __launch_bounds__(16 * 16) sortGaussiansRayHierarchicalCUDA_forward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y, 
	const float far_plane,
	const bool include_alpha,
	const float* view2gaussian,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugType,
	float* __restrict__ out_color)
{
	constexpr uint2 debug_target_pixel = {500, 500};
	// int num_blends = 0;
	struct BlendData
	{
		float T;
		float T_opa;
		float C[CHANNELS*2];
		uint32_t contributor = 0;
		float opacity;
		float depth;
		float distortion;
		float dist1;
		float dist2;
		float extent_loss;
		uint32_t max_contributor{0};
		uint32_t blend_contributor{0};
		float3 ray_dir;

		float debug;
	};

	auto prep_function = [&](bool inside, const uint2& pixpos, const float3 ray_dir)
		{
			BlendData bd;
			bd.ray_dir = ray_dir;
			bd.T = 1.0f;
			bd.T_opa = 1.0f;
			for (int ch = 0; ch < CHANNELS*2; ++ch)
			{
				bd.C[ch] = 0.0f;
			}
			bd.distortion = 0.f;
			bd.depth = 0.f;
			bd.opacity = 0.f;
			bd.dist1 = 0.f;
			bd.dist2 = 0.f;
			bd.extent_loss = 0.f;
			bd.debug = 0.f;
#if (DEBUG_HIERARCHICAL & 0x200) != 0
			if(pixpos.x == debug_target_pixel.x && pixpos.y == debug_target_pixel.y)
			{
				printf("+++++++++++++++++++++++++++++++++++++++++++\n");
			}
#endif
			return bd;
		};
	auto store_function = [](const uint2&, int coll_id, float G, float alpha, float depth)
		{
			return alpha;
		};
	auto blend_function = [&](const uint2& pixpos, BlendData& blend_data, int id, float alpha, float t, const float* view2gaussian_j, float2 ray, DebugVisualization debugType, float3 normal_, float4 ABC_)
		{
			const float normal[3] = {normal_.x, normal_.y, normal_.z};
			[[maybe_unused]] const float AA = ABC_.x;
			[[maybe_unused]] const float BB = ABC_.y;
			[[maybe_unused]] const float CC = ABC_.z;

			float test_T = blend_data.T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				return false;
			}

			blend_data.blend_contributor++;

			// TODO: consider using vectors and better loads?
			for (int ch = 0; ch < CHANNELS; ch++)
				blend_data.C[ch] += features[id * CHANNELS + ch] * alpha * blend_data.T;

			// NDC mapping is taken from 2DGS paper, please check here https://arxiv.org/pdf/2403.17888.pdf
			const float max_t = t;
			const float mapped_max_t = (far_plane * max_t - far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * max_t);
			
			// normalize normal
			float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] + 1e-7);
			const float normal_normalized[3] = { -normal[0] / length, -normal[1] / length, -normal[2] / length };

			// distortion loss is taken from 2DGS paper, please check https://arxiv.org/pdf/2403.17888.pdf
			float A = 1-blend_data.T;
			float error = mapped_max_t * mapped_max_t * A + blend_data.dist2 - 2 * mapped_max_t * blend_data.dist1;
			blend_data.distortion += error * alpha * blend_data.T;

			blend_data.dist1 += mapped_max_t * alpha * blend_data.T;
			blend_data.dist2 += mapped_max_t * mapped_max_t * alpha * blend_data.T;

			// normal
			for (int ch = 0; ch < CHANNELS; ch++)
				blend_data.C[CHANNELS + ch] += normal_normalized[ch] * alpha * blend_data.T;
			
			// depth and alpha
            if (blend_data.T > 0.5f)
            {
				if constexpr (EXACT_DEPTH) {
					if (test_T < 0.5f) {
						float Fp = (-0.5/(blend_data.T*conic_opacity[id].w)) + (1.0f/conic_opacity[id].w);
						float con = CC + 2 * logf(Fp);
						// why does this need to be abs?
						// TODO: new formulation with variance along the Gaussian
						float disc = abs(BB*BB - 4* AA*(con));
						float median_t = sqrtf(disc + 1e-9);

						// TODO: can we somehow scrap fminf
						// -BB, 1/2A are always positive (experiments)
						// median_t is always positive, due to being the sqrt of a positive nmber - only this one makes sense
						median_t = (-BB - median_t)/2.0f/AA;
						blend_data.depth = median_t;
					}
					else {
						blend_data.depth = t;
					}
				}
				else {
					blend_data.depth = t;
				}
				blend_data.max_contributor++;
            }
			else {
				// TODO: test if t* < t
				// if so, we have an issue
				float alpha_point = alpha;
				if (t > blend_data.depth) {
					float min_value = (AA * blend_data.depth * blend_data.depth + BB * blend_data.depth + CC);
					float p = -0.5f * min_value;
					if (p > 0.0f){
						p = 0.0f;
					}
	
					alpha_point = min(0.99f, ABC_.w * exp(p));
				}
				blend_data.opacity += alpha_point * blend_data.T_opa;
				blend_data.T_opa *= (1 - alpha_point);
			}

			// const float C = (far_plane * NEAR_PLANE * sqrtf(2 * logf(255 * ABC_.w))) / (far_plane - NEAR_PLANE);
			// float NDCspan = C * 4 * AA * sqrtf(AA) / (BB * BB);

			const float FN = (far_plane * NEAR_PLANE) / (far_plane - NEAR_PLANE);
			float C = (CC - 2 * logf(255 * ABC_.w));
			float extent = sqrtf(abs(BB*BB - 4* AA*C) + 1e-9);
			float NDCspan = FN * (2 * AA * extent) / (BB * BB);

			if (include_alpha) {
				blend_data.extent_loss += blend_data.T * alpha * NDCspan;
			}
			else {
				blend_data.extent_loss += blend_data.T * NDCspan;
			}
			
#if (DEBUG_HIERARCHICAL & 0x200) != 0
			if(pixpos.x == debug_target_pixel.x && pixpos.y == debug_target_pixel.y)
			{
				glm::mat3 inv;
				inv[0][0] = cov3Ds_inv[3*id].x;
				inv[0][1] = cov3Ds_inv[3*id].y;
				inv[0][2] = cov3Ds_inv[3*id].z;
				inv[1][0] = cov3Ds_inv[3*id].y;
				inv[1][1] = cov3Ds_inv[3*id+1].x;
				inv[1][2] = cov3Ds_inv[3*id+1].y;
				inv[2][0] = cov3Ds_inv[3*id].z;
				inv[2][1] = cov3Ds_inv[3*id+1].y;
				inv[2][2] = cov3Ds_inv[3*id+1].z;

				glm::vec3 ray_dir(blend_data.ray_dir.x, blend_data.ray_dir.y, blend_data.ray_dir.z);

				float sigma = glm::dot(ray_dir, inv * ray_dir);
				sigma = 1.0f/sigma;
				
				printf("t: %f, alpha: %f, sigma: %f\n", t, alpha, sigma);
			}
#endif
			// if the depth is larger than the current depth we're looking at, evaluate at the current position
			// if (blend_data.contributor > blend_data.max_contributor) {

			// }
			//++++++++++++++++GOF

			blend_data.T = test_T;

			return true;
		};
	auto fin_function = [&](const uint2& pixpos, BlendData& blend_data, DebugVisualization debugType, int range, float3 o)
		{		
			uint32_t pix_id = W * pixpos.y + pixpos.x;
			// A, D, and D^2
			final_T[pix_id] = blend_data.T;
			final_T[pix_id + H * W] = blend_data.dist1;
			final_T[pix_id + 2 * H * W] = blend_data.dist2;
			final_T[pix_id + 3 * H * W] = blend_data.T_opa;

			n_contrib[pix_id] = blend_data.max_contributor;

			// blend_data.extent_loss /= abs(blend_data.depth + 1e-9);

			if constexpr (!ENABLE_DEBUG_VIZ)
			{
				for (int ch = 0; ch < CHANNELS; ch++)
					out_color[ch * H * W + pix_id] = blend_data.C[ch] + blend_data.T * bg_color[ch];
				// normal
				for (int ch = 0; ch < CHANNELS; ch++){
					out_color[(CHANNELS + ch) * H * W + pix_id] = blend_data.C[CHANNELS+ch];
				}
				// depth and alpha
				out_color[DEPTH_OFFSET * H * W + pix_id] = blend_data.depth;
				out_color[ALPHA_OFFSET * H * W + pix_id] = blend_data.opacity;
				out_color[DISTORTION_OFFSET * H * W + pix_id] = blend_data.distortion;
				out_color[OPACITY_LOSS_OFFSET * H * W + pix_id] = blend_data.extent_loss;
			}
			else
			{
				outputDebugVis(debugType, out_color, pix_id, blend_data.blend_contributor, blend_data.T, blend_data.depth, blend_data.opacity, blend_data.distortion, blend_data.extent_loss, blend_data.debug, range, blend_data.max_contributor,H, W);
			}
#if (DEBUG_HIERARCHICAL & 0x200) != 0
			if(pixpos.x == debug_target_pixel.x && pixpos.y == debug_target_pixel.y)
			{
				printf("+++++++++++++++++++++++++++++++++++++++++++\n");
			}
#endif
#ifdef DEBUG_OPACITY_FIELD
			if (pixpos.x < 10 && pixpos.y < 10) {
				float d = blend_data.C[CHANNELS * 2];
				float3 depth = {
					o.x + d * blend_data.ray_dir.x,
					o.y + d * blend_data.ray_dir.y,
					o.z + d * blend_data.ray_dir.z
				};
				printf("[%d, %d]: depth: %.3f, depth point %.3f %.3f %.3f\n", pixpos.y, pixpos.x, blend_data.C[CHANNELS * 2], depth.x, depth.y, depth.z);
			}
#endif
		};

	sortGaussiansRayHierarchicaEvaluation<HEAD_WINDOW, MID_WINDOW, CULL_ALPHA>(
		ranges, point_list, W, H, focal_x, focal_y, view2gaussian, points_xy_image, cov3Ds_inv, projmatrix_inv, cam_pos, conic_opacity, debugType,
		prep_function, store_function, blend_function, fin_function);
}

template <int32_t CHANNELS, int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA = true, bool EXACT_DEPTH = false, bool ENABLE_DEBUG_VIZ = false>
__global__ void __launch_bounds__(16 * 16) sortGaussiansRayHierarchicalCUDA_opacity(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y, 
	const float far_plane,
	const float* view2gaussian,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugType,
	float* __restrict__ out_color)
{
	constexpr uint2 debug_target_pixel = {500, 500};
	// int num_blends = 0;
	struct BlendData
	{
		float T_opa;
		float T;
		float opacity;
		float max_depth;

		float3 ray_dir;
		uint32_t contributor{0};
		uint32_t current_contributor{0};
		uint32_t max_contributor{0};
	};

	auto prep_function = [&](bool inside, const uint2& pixpos, const float3 ray_dir)
		{
			uint32_t pix_id = W * pixpos.y + pixpos.x;
			BlendData bd;
			bd.ray_dir = ray_dir;

			// needed for opacity evaluation
			bd.T_opa = 1.0f;
			bd.T = 1.f;

			bd.opacity = 0.f;

			// get the max contributor
			bd.max_depth = inside ? out_color[DEPTH_OFFSET * H * W + pix_id] : 0;
			bd.max_contributor = inside ? n_contrib[pix_id] : 0;

			return bd;
		};
	auto store_function = [](const uint2&, int coll_id, float G, float alpha, float depth)
		{
			return alpha;
		};
	auto blend_function = [&](const uint2& pixpos, BlendData& blend_data, int id, float alpha, float t, const float* view2gaussian_j, float2 ray, DebugVisualization debugType, float3 normal_, float4 ABC_)
		{
			// accumulate the opacity up to the current contributor
			if (++blend_data.current_contributor > blend_data.max_contributor) {
				return false;
			}

			float alpha_point = alpha;
			if (t > blend_data.max_depth) {
				[[maybe_unused]] const float AA = ABC_.x;
				[[maybe_unused]] const float BB = ABC_.y;
				[[maybe_unused]] const float CC = ABC_.z;

				float min_value = (AA * blend_data.max_depth * blend_data.max_depth + BB * blend_data.max_depth + CC);
				float p = -0.5f * min_value;
				if (p > 0.0f){
					p = 0.0f;
				}
				alpha_point = min(0.99f, ABC_.w * exp(p));
			}

			blend_data.opacity += alpha_point * blend_data.T_opa;
			blend_data.T_opa *= (1 - alpha_point);

			return true;
		};
	auto fin_function = [&](const uint2& pixpos, BlendData& blend_data, DebugVisualization debugType, int range, float3 o)
		{		
			uint32_t pix_id = W * pixpos.y + pixpos.x;
			// we already have everything, except
			// - final opacity
			// - final T_opa
			float T_opa_rest = final_T[pix_id + 3 * H * W];
			float opa_rest = out_color[ALPHA_OFFSET * H * W + pix_id];

			// final opacity = blend_data.opacity + T_opa_k * opa_rest
			float opacity = blend_data.opacity + blend_data.T_opa * opa_rest;
			out_color[ALPHA_OFFSET * H * W + pix_id] = opacity;
			
			// final T_opa = T_opa_k * T_opa_rest
			final_T[pix_id + 3 * H * W] = blend_data.T_opa * T_opa_rest;

			if constexpr (ENABLE_DEBUG_VIZ)
			{
				outputDebugVis(debugType, out_color, pix_id, 0.0f, blend_data.T, blend_data.max_depth, opacity, 0.0f, 0.0f, 0.0f, range, blend_data.max_contributor,H, W);
			}
		};

	sortGaussiansRayHierarchicaEvaluation<HEAD_WINDOW, MID_WINDOW, CULL_ALPHA>(
		ranges, point_list, W, H, focal_x, focal_y, view2gaussian, points_xy_image, cov3Ds_inv, projmatrix_inv, cam_pos, conic_opacity, debugType,
		prep_function, store_function, blend_function, fin_function);
}


template <int32_t CHANNELS, int HEAD_WINDOW, int MID_WINDOW, bool CULL_ALPHA = true, bool DETACH_ALPHA = true>
__global__ void __launch_bounds__(16 * 16) sortGaussiansRayHierarchicalCUDA_backward(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y, 
	const float far_plane,
	const bool detach_alpha_extent,
	const bool include_alpha,
	const float* view2gaussian,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ pixel_colors,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* dL_dview2gaussian)
{
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;
	// int num_blends = 0;
	struct BlendData
	{
		float T_final;
		float dL_dpixel[CHANNELS];
		//++++++++++++GOF
		float dL_dnormal2D[3]; // Normal
		float dL_dmax_depth = 0;
		float max_depth = 0.f;
		float final_D;
		float final_D2;
		float final_A;
		float dL_dreg;
		float dL_dextent_loss;
		float extent_loss_rem;
		float last_alpha;
		float accum_normal_rec[3] = {0};
		float last_normal[3] = { 0 };
		float last_dL_dT = 0;

		float remaining_A;
		float remaining_D;
		float remaining_D2;

		// Opacity field loss
		float opacity_final;
		float T_opa_final;
		float T_opa;
		float opacity;
		float dL_dopacity;
		float dL_dt;
		float dt_dA;
		float dt_dB;

		uint32_t max_contributor;
		uint32_t contributor = 0;
		uint32_t current_contributor = 0;

		uint32_t depth_global_id{-1};
		float2 ray;

		//++++++++++++GOF
		float final_color[CHANNELS];
		float final_normal[CHANNELS];
		float T;
		float C[CHANNELS*2];
	};
	auto prep_function = [&](bool inside, const uint2& pixpos, const float3 raydir)
		{
			uint32_t pix_id = W * pixpos.y + pixpos.x;
			BlendData bd;
			bd.T = 1.0f;
			bd.T_final = inside ? final_Ts[pix_id] : 0;
			//++++++++++++GOF
			bd.final_D = inside ? final_Ts[pix_id + H * W] : 0;
			bd.final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
			bd.final_A = 1 - bd.T_final;
			bd.dL_dreg = inside ? dL_dpixels[DISTORTION_OFFSET * H * W + pix_id] : 0;

			// new opacity loss
			bd.extent_loss_rem = inside ? pixel_colors[OPACITY_LOSS_OFFSET * H * W + pix_id] : 0;
			bd.dL_dextent_loss = inside ? dL_dpixels[OPACITY_LOSS_OFFSET * H * W + pix_id] : 0;
			bd.dL_dt = 0.f;
			bd.dt_dA = 0.f;
			bd.dt_dB = 0.f;
			bd.ray = {raydir.x, raydir.y};

			bd.remaining_A = bd.final_A;
			bd.remaining_D = bd.final_D;
			bd.remaining_D2 = bd.final_D2;

			// Opacity field loss
			bd.opacity_final = inside ? pixel_colors[ALPHA_OFFSET * H * W + pix_id] : 0;
			bd.T_opa_final = inside ? final_Ts[pix_id + 3 * H * W] : 0;
			bd.T_opa = 1.0f;
			bd.opacity = 0.f;
			bd.dL_dopacity = inside ? dL_dpixels[ALPHA_OFFSET * H * W + pix_id] : 0;

			bd.max_contributor = inside ? n_contrib[pix_id] : 0;
			bd.max_depth = inside ? pixel_colors[DEPTH_OFFSET * H * W + pix_id] : 0;

			// blend_data.extent_loss /= (blend_data.depth + 1e-9);
			// depth is detached (for now)
			// bd.dL_dextent_loss /= (bd.max_depth + 1e-9);

			//++++++++++++GOF
			for (int ch = 0; ch < CHANNELS*2; ++ch)
			{
				bd.C[ch] = 0.f;
			}
			for (int ch = 0; ch < CHANNELS; ++ch)
			{
				if (inside)
				{
					bd.dL_dpixel[ch] = dL_dpixels[ch * H * W + pix_id];
					bd.final_color[ch] = pixel_colors[ch * H * W + pix_id] - bd.T_final * bg_color[ch];
				}
			}
			//++++++++++++GOF
			if(inside)
			{
				for (int i = 0; i < 3; i++)
				{	
					bd.dL_dnormal2D[i] = dL_dpixels[(CHANNELS+i) * H * W + pix_id];
					bd.final_normal[i] = pixel_colors[(CHANNELS+i) * H * W + pix_id];
				}
				bd.dL_dmax_depth = dL_dpixels[DEPTH_OFFSET * H * W + pix_id];
			}
			bd.last_alpha = 0;
			//++++++++++++GOF
				
			return bd;
		};
	auto store_function = [](const uint2&, int coll_id, float G, float alpha, float depth)
		{
			return G;
		};
	auto blend_function = [&](const uint2& pixpos, BlendData& blend_data, int global_id, float G, float t, const float* view2gaussian_j, float2 ray, DebugVisualization debugType, float3 normal_, float4 ABC_)
		{
			const float4 con_o = conic_opacity[global_id];

			//++++++++++++GOF
			const float normal[3] = {normal_.x, normal_.y, normal_.z};
			[[maybe_unused]] const float AA = ABC_.x;
			[[maybe_unused]] const float BB = ABC_.y;
			[[maybe_unused]] const float CC = ABC_.z;

			const float alpha = min(0.99f, con_o.w * G);
			float test_T = blend_data.T * (1.0f - alpha);
			if (test_T < 0.0001f)
			{
				return false;
			}			
			//++++++++++++GOF		

			// NDC mapping is taken from 2DGS paper, please check here https://arxiv.org/pdf/2403.17888.pdf
			const float max_t = t;
			const float mapped_max_t = (far_plane * max_t - far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * max_t);
			
			float dmax_t_dd = (far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * max_t * max_t);
			
			// normalize normal
			float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] + 1e-7);
			const float normal_normalized[3] = { -normal[0] / length, -normal[1] / length, -normal[2] / length};

			// ++num_blends;

			const float2 xy = points_xy_image[global_id];
			const float2 d = { xy.x - static_cast<float>(pixpos.x), xy.y - static_cast<float>(pixpos.y) };


			const float dchannel_dcolor = alpha * blend_data.T;


            //Gradients for depth supervision
			float dL_dA = 0.0f; //dL_dmin_value * (BB / AA) * (BB / AA) / 4.f; //0.0f
			float dL_dB = 0.0f; //dL_dmin_value * -BB / (2 *AA);//0.0f
			float dL_dC = 0.0f;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				const float c = colors[global_id * CHANNELS + ch];

				// reconstruct color up to this point
				blend_data.C[ch] += c * alpha * blend_data.T;
				// the contribution of all other gaussian coming after
				float accum_rec_ch = (blend_data.final_color[ch] - blend_data.C[ch]) / test_T;

				const float dL_dchannel = blend_data.dL_dpixel[ch];
				dL_dalpha += (c - accum_rec_ch) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * CHANNELS + ch]), dchannel_dcolor * dL_dchannel);
			}

			//++++++++++++GOF
			// gradient for the distoration loss is taken from 2DGS paper, please check https://arxiv.org/pdf/2403.17888.pdf
			float dL_dt = 0.0f;
			float dL_dmax_t = 0.0f;
			float dL_dweight = 0.0f;

			dL_dmax_t += 2.0f * (blend_data.T * alpha) * (mapped_max_t * blend_data.final_A - blend_data.final_D) * blend_data.dL_dreg * dmax_t_dd;

			// if weight is not detached
			if constexpr (!DETACH_ALPHA) {
				dL_dweight += (blend_data.final_D2 + mapped_max_t * mapped_max_t * blend_data.final_A - 2 * mapped_max_t * blend_data.final_D) * blend_data.dL_dreg;	

				blend_data.remaining_A -= blend_data.T * alpha;
				blend_data.remaining_D -= mapped_max_t * blend_data.T * alpha;
				blend_data.remaining_D2 -= mapped_max_t * mapped_max_t * blend_data.T * alpha;

				// in front-to-back order, we need T_i * (dL/d_wi - 1/test_T * \sum_{j=i+1}^N w_j dL_dw_j
				float dL_di_plus_one = (blend_data.final_D2 * blend_data.remaining_A + blend_data.final_A * blend_data.remaining_D2 - 2.f * blend_data.final_D * blend_data.remaining_D);
				dL_dalpha += dL_dweight - dL_di_plus_one / (test_T) * blend_data.dL_dreg;
			}
			
			float dL_dnormal_normalized[3] = {0};
			// // Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				// reconstruct color up to this point
				blend_data.C[ch+CHANNELS] += normal_normalized[ch] * alpha * blend_data.T;
				// the contribution of all other gaussian coming after
				blend_data.accum_normal_rec[ch] = (blend_data.final_normal[ch] - blend_data.C[ch+CHANNELS]) / test_T;

				dL_dalpha += (normal_normalized[ch] - blend_data.accum_normal_rec[ch]) * blend_data.dL_dnormal2D[ch];
				dL_dnormal_normalized[ch] = alpha * blend_data.T * blend_data.dL_dnormal2D[ch];
			}
			
			// float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] + 1e-7);
			// const float normal_normalized[3] = { -normal[0] / length, -normal[1] / length, -normal[2] / length};
			float dL_dlength = (dL_dnormal_normalized[0] * normal[0] + dL_dnormal_normalized[1] * normal[1] + dL_dnormal_normalized[2] * normal[2]);
			dL_dlength *= 1.f / (length * length);
			float dL_dnormal[3] = {
				(-dL_dnormal_normalized[0] + dL_dlength * normal[0]) / length,
				(-dL_dnormal_normalized[1] + dL_dlength * normal[1]) / length,
				(-dL_dnormal_normalized[2] + dL_dlength * normal[2]) / length
			};
			
			dL_dt = dL_dmax_t;

			float dL_do = 0.0f;
			if (++blend_data.current_contributor == blend_data.max_contributor) {
				dL_dt += blend_data.dL_dmax_depth;

				// blend_data.depth_global_id = global_id;
				// blend_data.dt_dA = BB / (2 * AA * AA);
				// blend_data.dt_dB = -1.f / (2 * AA);
#ifdef CORRECT_EXACT_DEPTH_GRAD
				float Fp = (-0.5/(blend_data.T*ABC_.w)) + (1.0f/ABC_.w);
				float con = CC + 2 * logf(Fp);
				// why does this need to be abs?
				// TODO: new formulation with variance along the Gaussian
				float disc = abs(BB*BB - 4* AA*(con));
				float median_t = sqrtf(disc + 1e-9);

				dL_dA += blend_data.dL_dmax_depth * (BB * BB - 2 * AA * (con)) / (median_t * 2 * AA * AA);
				dL_dB += blend_data.dL_dmax_depth * t / median_t;
				dL_dC += blend_data.dL_dmax_depth / median_t; 
#endif
			}

			dL_dalpha *= blend_data.T;
			
			// here, the depth of the Gaussian is larger than the depth of the pixel
			float alpha_point = 0.f;
			if (t > blend_data.max_depth) {
				// TODO: propagate gradient to AA, BB, CC directly
				float min_value = (AA * blend_data.max_depth * blend_data.max_depth + BB * blend_data.max_depth + CC);
				float p = -0.5f * min_value;
				if (p > 0.0f){
					p = 0.0f;
				}
				float expp = exp(p);
				alpha_point = min(0.99f, ABC_.w * expp);
				float dL_dalpha_point = (blend_data.T_opa - 1.f / (1 - alpha_point) * (blend_data.opacity_final - blend_data.opacity)) * blend_data.dL_dopacity;

				// derive gradients w.r.t. opacity
				// alpha_point = min(0.99f, ABC_.w * exp(p));
				dL_do += dL_dalpha_point * expp;

				// derive gradients w.r.t. AA,BB,CC
				// alpha_point = min(0.99f, ABC_.w * exp(p));
				float dL_dexpp = dL_dalpha_point * ABC_.w;
				float dL_dp = dL_dexpp * expp;

				// float p = -0.5f * min_value;
				float dL_dmin_value = dL_dp * -0.5f;

				// float min_value = (AA * blend_data.max_depth * blend_data.max_depth + BB * blend_data.max_depth + CC);
				dL_dA += dL_dmin_value * blend_data.max_depth * blend_data.max_depth;
				dL_dB += dL_dmin_value * blend_data.max_depth;
				dL_dC += dL_dmin_value;

				// dO_dt = dO_dalpha * dalpha_dt
				// float min_value = (AA * blend_data.max_depth * blend_data.max_depth + BB * blend_data.max_depth + CC);
				blend_data.dL_dt += dL_dalpha_point * (-alpha_point * (AA * blend_data.max_depth + BB / 2));
			}
			// otherwise (eval at max contrib)
			else {
				alpha_point = alpha;
				// just add the gradient to alpha, the computation will take care of the rest
				dL_dalpha += (blend_data.T_opa - 1.f / (1 - alpha_point) * (blend_data.opacity_final - blend_data.opacity)) * blend_data.dL_dopacity;
			}

			// first, accumulate the opacity
			blend_data.opacity += alpha_point * blend_data.T_opa;
			blend_data.T_opa *= (1 - alpha_point);

			// const float C = (far_plane * NEAR_PLANE * sqrtf(2 * logf(255 * ABC_.w))) / (far_plane - NEAR_PLANE);

			// float dL_dNDC = blend_data.T * alpha * blend_data.dL_dextent_loss * C;

			// extent: C * 4 * AA^(3/2) / (BB * BB)
			// dL_dA += 6 * sqrtf(AA) / powf(BB, 2) * dL_dNDC;

			// // extent: C * 4 * AA^(3/2) / (BB * BB)
			// dL_dB += - 8 * AA * sqrtf(AA) / powf(BB, 3) * dL_dNDC;

			// float NDCspan = C * 4 * AA * sqrtf(AA) / (BB * BB);

			const float FN = (far_plane * NEAR_PLANE) / (far_plane - NEAR_PLANE);
			float C = (CC - 2 * logf(255 * ABC_.w));
			float extent = sqrtf(abs(BB*BB - 4* AA*C) + 1e-9);

			float dL_dNDC = blend_data.T * blend_data.dL_dextent_loss * FN;
			dL_dNDC /= (BB * BB * extent);

			if (include_alpha) {
				dL_dNDC *= alpha;
			}

			// extent: C * 4 * AA^(3/2) / (BB * BB)
			dL_dA += (2 * (BB * BB - 6 * AA * C)) * dL_dNDC;

			// extent: C * 4 * AA / (BB * BB)
			dL_dB += (16 * AA * AA * C - 2 * AA * BB * BB) / BB * dL_dNDC;

			// formula is below anyway
			dL_dC += (-4 * AA * AA) * dL_dNDC;

			float NDCspan = FN * (2 * AA * extent) / (BB * BB);

			if (!detach_alpha_extent) {
				dL_do += (8 * AA * AA) / ABC_.w * dL_dNDC;

				if (include_alpha) {
					blend_data.extent_loss_rem -= blend_data.T * alpha * NDCspan;		
					dL_dalpha += blend_data.T * (NDCspan - blend_data.extent_loss_rem / test_T) * blend_data.dL_dextent_loss;
			
				}
				else {
					blend_data.extent_loss_rem -= blend_data.T * NDCspan;		
					dL_dalpha += blend_data.extent_loss_rem / (1 - alpha) * blend_data.dL_dextent_loss;
				}
			}

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < CHANNELS; i++)
				bg_dot_dpixel += bg_color[i] * blend_data.dL_dpixel[i];
			dL_dalpha += (-blend_data.T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);
			const float abs_dL_dmean2D = abs(dL_dG * dG_ddelx * ddelx_dx) + abs(dL_dG * dG_ddely * ddely_dy);
            atomicAdd(&dL_dmean2D[global_id].z, abs_dL_dmean2D);


			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			// atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			// atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha + dL_do);

			//++++++++++++GOF
			const float dG_dpower = G;
			const float dL_dpower = dL_dG * dG_dpower;

			// // float power = -0.5f * min_value;
			const float dL_dmin_value = dL_dpower * -0.5f;
			// float min_value = -(BB*BB)/(4*AA) + CC;
			// const float dL_dA = dL_dmin_value * (BB*BB)/4 *  1. / (AA*AA);
			dL_dA += dL_dmin_value * (BB / AA) * (BB / AA) / 4.f;
			dL_dB += dL_dmin_value * -BB / (2 *AA);
			dL_dC += dL_dmin_value * 1.0f;

			dL_dA += dL_dt * BB / (2 * AA * AA);
			dL_dB += dL_dt * -1.f / (2 * AA);

			// const float normal[3] = { view2gaussian_j[0] * ray.x + view2gaussian_j[1] * ray.y + view2gaussian_j[2], 
			// 						view2gaussian_j[1] * ray.x + view2gaussian_j[3] * ray.y + view2gaussian_j[4],
			// 						view2gaussian_j[2] * ray.x + view2gaussian_j[4] * ray.y + view2gaussian_j[5]};

			// use AA, BB, CC so that the name is unique
			// float AA = ray.x * normal[0] + ray.y * normal[1] + normal[2];
			// float BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			// float CC = view2gaussian_j[9];
			dL_dnormal[0] += dL_dA * ray.x;
			dL_dnormal[1] += dL_dA * ray.y;
			dL_dnormal[2] += dL_dA;
			
			// write the gradients to global memory directly
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 0]), dL_dnormal[0] * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 1]), dL_dnormal[0] * ray.y + dL_dnormal[1] * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 2]), dL_dnormal[0] + dL_dnormal[2] * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 3]), dL_dnormal[1] * ray.y);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 4]), dL_dnormal[1] + dL_dnormal[2] * ray.y);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 5]), dL_dnormal[2]);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 6]), dL_dB * 2 * ray.x);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 7]), dL_dB * 2 * ray.y);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 8]), dL_dB * 2);
			atomicAdd(&(dL_dview2gaussian[global_id * 10 + 9]), dL_dC);
			//++++++++++++GOF

			blend_data.T = test_T;
#ifdef ENABLE_NAN_CHECKS
            if(isnan(dL_dnormal[0]) || isnan(dL_dnormal[1]) || isnan(dL_dnormal[2]))
            {
                printf("Normals(%f, %f, %f)\n", dL_dnormal[0], dL_dnormal[1], dL_dnormal[2]);
            }
            if(isnan(dL_dA) || isnan(dL_dB) || isnan(dL_dC) || isnan(blend_data.dL_dmax_depth))
            {
                printf("dABC(%f, %f, %f, %f, %f, %f)\n", dL_dA, dL_dB, dL_dC, dL_dt, blend_data.dL_dmax_depth, blend_data.max_depth);
            }
            if(isnan(dL_dmean2D[global_id].x) || isnan(dL_dmean2D[global_id].y) || isnan(dL_dmean2D[global_id].z))
            {
               printf("(%f, %f, %f)\n",dL_dmean2D[global_id].x, dL_dmean2D[global_id].y, dL_dmean2D[global_id].z);
            }
            for(int i = 0; i < 10; i++)
            {
               if(isnan(dL_dview2gaussian[global_id * 10 + i]))
               {
                   printf("dL_dview2gaussian %d : %f, %f, %f, %f)\n", i, dL_dview2gaussian[global_id * 10 + i]);
               }
            }
#endif
			return true;
		};
	auto fin_function = [&](const uint2& pixpos, BlendData& blend_data, DebugVisualization debugType, int range, float3 o)
		{
#ifdef DEBUG_OPACITY_FIELD
			float diff = blend_data.opacity_final - blend_data.opacity;
			if (abs(diff) > 1e-5)
				printf("%u, %u:\t O %.5f (%.3f - %.3f)\n", pixpos.x, pixpos.y, diff, blend_data.opacity_final, blend_data.opacity);
			float diff2 = blend_data.T_opa_final - blend_data.T_opa;
			if (abs(diff2) > 1e-5)
				printf("%u, %u:\t T %.5f (%.3f - %.3f)\n", pixpos.x, pixpos.y, diff2, blend_data.T_opa_final, blend_data.T_opa);
#endif
			// if (blend_data.depth_global_id != -1) {
			// 	// we need to compute the gradients for the dO/dA, dO/dB
			// 	float dL_dt = blend_data.dL_dt;

			// 	// gradient with respect to A,B,C
			// 	float dL_dA = blend_data.dt_dA * dL_dt;
			// 	float dL_dB = blend_data.dt_dB * dL_dt;

			// 	float dL_dnormal[3] = {
			// 		dL_dA * blend_data.ray.x,
			// 		dL_dA * blend_data.ray.y,
			// 		dL_dA
			// 	};

			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 0]), dL_dnormal[0] * blend_data.ray.x);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 1]), dL_dnormal[0] * blend_data.ray.y + dL_dnormal[1] * blend_data.ray.x);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 2]), dL_dnormal[0] + dL_dnormal[2] * blend_data.ray.x);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 3]), dL_dnormal[1] * blend_data.ray.y);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 4]), dL_dnormal[1] + dL_dnormal[2] * blend_data.ray.y);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 5]), dL_dnormal[2]);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 6]), dL_dB * 2 * blend_data.ray.x);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 7]), dL_dB * 2 * blend_data.ray.y);
			// 	atomicAdd(&(dL_dview2gaussian[blend_data.depth_global_id * 10 + 8]), dL_dB * 2);
			// }
			return;
		};

	sortGaussiansRayHierarchicaEvaluation<HEAD_WINDOW, MID_WINDOW, CULL_ALPHA>(
		ranges, point_list, W, H, focal_x, focal_y, view2gaussian, points_xy_image, cov3Ds_inv, projmatrix_inv, cam_pos, conic_opacity, DebugVisualization::Disabled,
		prep_function, store_function, blend_function, fin_function);
}