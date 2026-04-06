/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

#pragma once

#include "../auxiliary.h"
#include "stopthepop_common.cuh"

#include <cub/block/block_radix_sort.cuh>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;


template <uint32_t CHANNELS, int WINDOW_SIZE, bool ENABLE_DEBUG_VIZ = false>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
renderkBufferCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugVisualizationType,
	float* __restrict__ out_color)
{
    // number of elements each thread considers for reshuffling
	constexpr int PerThreadSortWindow = WINDOW_SIZE;

	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list..
	uint2 range;
	range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// setup view dir
	const glm::mat4 inverse_vp = loadMatrix4x4(projmatrix_inv);
    const float3 campos = *cam_pos;
    float3 dir = computeViewRay(inverse_vp, campos, pixf, W, H);

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	float C[CHANNELS] = { 0 };

	float sort_depths[PerThreadSortWindow];
	float sort_alphas[PerThreadSortWindow];
	int sort_ids[PerThreadSortWindow];
	int sort_num = 0;

	for (int i = 0; i < PerThreadSortWindow; ++i)
	{
		sort_depths[i] = FLT_MAX;
		// just to suppress warnings:
		sort_alphas[i] = 0;
		sort_ids[i] = -1;
	}

	[[maybe_unused]] float depth_accum = 0.f;
	[[maybe_unused]] float currentDepth = -FLT_MAX;
	[[maybe_unused]] float sortingErrorCount = 0.f;

	auto blend_one = [&]() {
			if (sort_num == 0)
				return;
			--sort_num;
			float test_T = T * (1 - sort_alphas[0]);

			if (test_T < 0.0001f) {
				done = true;
				return;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[sort_ids[0] * CHANNELS + ch] * sort_alphas[0] * T;

			T = test_T;

			for (int i = 1; i < PerThreadSortWindow; ++i)
			{
				sort_depths[i - 1] = sort_depths[i];
				sort_alphas[i - 1] = sort_alphas[i];
				sort_ids[i - 1] = sort_ids[i];
			}
			sort_depths[PerThreadSortWindow - 1] = FLT_MAX;
		};

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int all_done = __syncthreads_and(done);
		if (all_done)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if (sort_num == PerThreadSortWindow) 
				blend_one();

			if (done == true)
				break;

			// Keep track of current position in range
			contributor++;

			int coll_id = collected_id[j];
			if (coll_id < 0) {
				// since negative indices are at the end of list, 
				// we can skip everything from here one
				i = rounds;
				break;
			}

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = evaluate_opacity_factor(d.x, d.y, con_o);
			if (power < 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(-power));
			if (alpha < 1.0f / 255.0f)
				continue;

            float depth = depthAlongRay(make_float3(cov3Ds_inv[3 * coll_id]), 
                                        make_float3(cov3Ds_inv[3 * coll_id + 1]), 
                                        make_float3(cov3Ds_inv[3 * coll_id + 2]), 
                                        dir);

			// culling for behind the camera!
			if (depth < 0.0f)
				continue;

			// push alpha and depth into per thread sorted array
			#pragma unroll
			for (int s = 0; s < PerThreadSortWindow; ++s) 
			{
				if (depth < sort_depths[s]) 
				{
					swap(depth, sort_depths[s]);
					swap(coll_id, sort_ids[s]);
					swap(alpha , sort_alphas[s]);
				}
			}
			++sort_num;
		}
	}

	if (!done) {
		while (sort_num > 0)
			blend_one();
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		if constexpr (!ENABLE_DEBUG_VIZ)
		{
			final_T[pix_id] = T;
			n_contrib[pix_id] = contributor;
			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		}
	}
}

template <uint32_t C, uint32_t WINDOW_SIZE>
__global__ void __launch_bounds__(BLOCK_X* BLOCK_Y)
renderkBufferBackwardCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float* __restrict__ colors,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
    const float* __restrict__ bg_color,
	const float* __restrict__ pixel_colors,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// number of elements each thread considers for reshuffling
	constexpr int PerThreadSortWindow = WINDOW_SIZE;

	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y};

	const bool inside = pix.x < W && pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// setup view dir
	const glm::mat4 inverse_vp = loadMatrix4x4(projmatrix_inv);
    const float3 campos = *cam_pos;
    float3 dir = computeViewRay(inverse_vp, campos, pixf, W, H);

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = 1.0f;
	float acc_colors[C] = { 0 };

	float sort_depths[PerThreadSortWindow];
	float sort_Gs[PerThreadSortWindow];
	int sort_ids[PerThreadSortWindow];
	int sort_num = 0;

	for (int i = 0; i < PerThreadSortWindow; ++i)
	{
		sort_depths[i] = FLT_MAX;
		// just to suppress warnings:
		sort_Gs[i] = 0;
		sort_ids[i] = -1;
	}

	uint32_t contributor = 0;

	float dL_dpixel[C];
	float final_color[C];
	if (inside)
	{
		for (int i = 0; i < C; i++)
		{
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
			final_color[i] = pixel_colors[i * H * W + pix_id] - T_final * bg_color[i];
		}
	}

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	auto blend_one = [&]() {
		if (sort_num == 0)
			return;
		--sort_num;

		int global_id = sort_ids[0];
		float G = sort_Gs[0];

		const float2 xy = points_xy_image[global_id];
		const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
		const float4 con_o = conic_opacity[global_id];

		const float alpha = min(0.99f, con_o.w * G);

		float test_T = T * (1 - alpha);
		if(test_T  < 0.0001f){
			done = true;
			return;
		}
		const float dchannel_dcolor = alpha * T;

		// Propagate gradients to per-Gaussian colors and keep
		// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
		// pair).
		float dL_dalpha = 0.0f;
		for (int ch = 0; ch < C; ch++)
		{
			const float c = colors[global_id * C + ch];

			// reconstruct color up to this point
			acc_colors[ch] += c * alpha * T;
			// the contribution of all other gaussian coming after
			float accum_rec_ch = (final_color[ch] - acc_colors[ch]) / test_T;

			const float dL_dchannel = dL_dpixel[ch];
			dL_dalpha += (c - accum_rec_ch) * dL_dchannel;
			// Update the gradients w.r.t. color of the Gaussian. 
			// Atomic, since this pixel is just one of potentially
			// many that were affected by this Gaussian.
			atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);

		}
		dL_dalpha *= T;

		// Account for fact that alpha also influences how much of
		// the background color is added if nothing left to blend
		float bg_dot_dpixel = 0;
		for (int i = 0; i < C; i++)
			bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
		dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


		// Helpful reusable temporary variables
		const float dL_dG = con_o.w * dL_dalpha;
		const float gdx = G * d.x;
		const float gdy = G * d.y;
		const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
		const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

		// Update gradients w.r.t. 2D mean position of the Gaussian
		atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
		atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

		// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
		atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
		atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
		atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

		// Update gradients w.r.t. opacity of the Gaussian
		atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);

		T = test_T;

		for (int i = 1; i < PerThreadSortWindow; ++i)
		{
			sort_depths[i - 1] = sort_depths[i];
			sort_Gs[i - 1] = sort_Gs[i];
			sort_ids[i - 1] = sort_ids[i];
		}
		sort_depths[PerThreadSortWindow - 1] = FLT_MAX;
	};


	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int all_done = __syncthreads_and(done);
		if (all_done)
			break;

		// Load auxiliary data into shared memory, start in the FRONT
		block.sync();
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			if (sort_num == PerThreadSortWindow)
				blend_one();
				
			if (done)
				break;

			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor++;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			int coll_id = collected_id[j];

            float depth = depthAlongRay(make_float3(cov3Ds_inv[3 * coll_id]), 
                                        make_float3(cov3Ds_inv[3 * coll_id + 1]), 
                                        make_float3(cov3Ds_inv[3 * coll_id + 2]), 
                                        dir);

			// culling for behind the camera!
			if (depth < 0.0f)
				continue;

			int id = collected_id[j];

			// push into per thread sorted array
#pragma unroll
			for (int s = 0; s < PerThreadSortWindow; ++s)
			{
				if (depth < sort_depths[s])
				{
					swap(depth, sort_depths[s]);
					swap(id, sort_ids[s]);
					swap(G, sort_Gs[s]);
				}
			}
			++sort_num;
		}
	}
	if (!done) {
		while (sort_num > 0)
			blend_one();
	}
}


template <uint32_t CHANNELS, bool ENABLE_DEBUG_VIZ = false>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderSortedFullCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ cov3Ds_inv,
	const float* __restrict__ projmatrix_inv,
	const float3* __restrict__ cam_pos,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugVisualizationType,
	float* __restrict__ out_color)
{
	constexpr int SortOverlapPerThread = 3;
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };

	constexpr int SortPerThread = SortOverlapPerThread + 1;

	// setting up the radix sort
	typedef cub::BlockRadixSort<float, BLOCK_X, SortPerThread, int, 4, true, cub::BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeFourByte, BLOCK_Y> BlockRadixSort;

	// stores the sorted indices to the data in shared memory
	union SMem {
		typename BlockRadixSort::TempStorage temp_storage;
		struct {
			int collected_id[BLOCK_SIZE];
			float2 collected_xy[BLOCK_SIZE];
			float4 collected_conic_opacity[BLOCK_SIZE];
			float collected_depth[BLOCK_SIZE];
		};
	};
	__shared__ SMem smem;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// iterate over all x/y for this tile
	for (uint32_t x = pix_min.x; x < pix_max.x; x++) {
		for (uint32_t y = pix_min.y; y < pix_max.y; y++) {
			// Initialize helper variables
			float2 pixf = { (float)x, (float)y };
			uint32_t pix_id = W * y + x;
			bool done = false;
			int toDo = range.y - range.x;

			// setup view dir
			const glm::mat4 inverse_vp = loadMatrix4x4(projmatrix_inv);
			const float3 campos = *cam_pos;
			float3 dir = computeViewRay(inverse_vp, campos, pixf, W, H);

			[[maybe_unused]] float depth_accum = 0.f;
			[[maybe_unused]] float currentDepth = -FLT_MAX;
			[[maybe_unused]] float sortingErrorCount = 0.f;

			float T = 1.0f;
			uint32_t contributor = 0;
			uint32_t last_contributor = 0;
			float C[CHANNELS] = { 0 };

			// dont touch out of bounds pixels
			if (!(x < W && y < H)) {
				continue;
			}

			// load the first round
			float key_t[SortPerThread];
			int value_t[SortPerThread];

			for (int i = 0; i < SortPerThread - 1; ++i) {
				int idx = range.x + i * BLOCK_SIZE + block.thread_rank();
				if (idx < range.y)
				{
					int coll_id = point_list[idx];
					key_t[i + 1] = depthAlongRay(make_float3(cov3Ds_inv[3 * coll_id]), 
												 make_float3(cov3Ds_inv[3 * coll_id + 1]), 
												 make_float3(cov3Ds_inv[3 * coll_id + 2]), 
												 dir);
					value_t[i + 1] = coll_id;
				}
				else {
					key_t[i + 1] = FLT_MAX;
					value_t[i + 1] = -1;
				}
			}
			

			// generate a sorted list of size min(BLOCK_SIZE, toDo)
			for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE) {
				int progress = (i + SortPerThread - 1) * BLOCK_SIZE + block.thread_rank();

				// if in range (i.e. thread has an underlying gaussian) compute t
				// else, write magic values for sorting
				if (range.x + progress < range.y)
				{
					// store information in shared memory
					int coll_id = point_list[range.x + progress];
					key_t[0] = depthAlongRay(make_float3(cov3Ds_inv[3 * coll_id]), 
											 make_float3(cov3Ds_inv[3 * coll_id + 1]), 
											 make_float3(cov3Ds_inv[3 * coll_id + 2]), 
											 dir);
					value_t[0] = coll_id;
				}
				else {
					key_t[0] = FLT_MAX;
					value_t[0] = -1;
				}
				block.sync();
				// sort values
				BlockRadixSort(smem.temp_storage).SortBlockedToStriped(key_t, value_t);
				// write to shared
			
				block.sync();

				smem.collected_id[block.thread_rank()] = value_t[0];
				smem.collected_xy[block.thread_rank()] = points_xy_image[value_t[0]];
				smem.collected_conic_opacity[block.thread_rank()] = conic_opacity[value_t[0]];
				smem.collected_depth[block.thread_rank()] = key_t[0];

				block.sync();

				// first thread computes output rgb
				if (block.thread_rank() == 0) {
					for (int idx = 0; !done && idx < min(BLOCK_SIZE, toDo); idx++)
					{
						if (smem.collected_id[idx] == -1) {
							break;
						}

						// rest of this code is identical to vanilla 3DGS
						contributor++;

						// Resample using conic matrix (cf. "Surface 
						// Splatting" by Zwicker et al., 2001)
						float2 xy = smem.collected_xy[idx];
						float2 d = { xy.x - pixf.x, xy.y - pixf.y };
						float4 con_o = smem.collected_conic_opacity[idx];
						float power = evaluate_opacity_factor(d.x, d.y, con_o);
						if (power < 0.0f) {
							continue;
						}

						// Eq. (2) from 3D Gaussian splatting paper.
						// Obtain alpha by multiplying with Gaussian opacity
						// and its exponential falloff from mean.
						// Avoid numerical instabilities (see paper appendix).
						float alpha = min(0.99f, con_o.w * exp(-power));
						if (alpha < 1.0f / 255.0f) {
							continue;
						}

						float test_T = T * (1 - alpha);
						if (test_T < 0.0001f)
						{
							done = true;
							continue;
						}

						// Eq. (3) from 3D Gaussian splatting paper.
						for (int ch = 0; ch < CHANNELS; ch++)
							C[ch] += features[smem.collected_id[idx] * CHANNELS + ch] * alpha * T;

						T = test_T;

						// Keep track of last range entry to update this
						// pixel.
						last_contributor = contributor;
					}
				}
				block.sync();
			}
			if (block.thread_rank() == 0) {
				// first thread has rgb/T stored, store output color
				if constexpr (!ENABLE_DEBUG_VIZ)
				{
					final_T[pix_id] = T;
					n_contrib[pix_id] = last_contributor;
					for (int ch = 0; ch < CHANNELS; ch++)
						out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
				}
			}
		}
	}
}