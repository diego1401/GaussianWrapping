/*
 * Copyright (C) 2024, Graz University of Technology
 * This code is licensed under the MIT license (see LICENSE.txt in this folder for details)
 */

#pragma once

#include "../auxiliary.h"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__forceinline__ __device__ glm::mat3 computeInvCov3D(const glm::vec3& scale, const glm::vec4& rot,  float scale_modifier = 1.0f)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	// TODO: handle numerical instabilities which might occur
	// effectively thickens gaussians
	S[0][0] = 1.f / (scale_modifier * max(1e-3f, scale.x));
	S[1][1] = 1.f / (scale_modifier * max(1e-3f, scale.y));
	S[2][2] = 1.f / (scale_modifier * max(1e-3f, scale.z));

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute (inverse of) 3D world covariance matrix Sigma
	return glm::transpose(M) * M;
}


template<class VEC3>
__device__ inline float depthAlongRay(const float3& first, const float3& second, const float3& third, const VEC3& viewdir)
{
	float3 viewdir_inv_cov = { first.x * viewdir.x + first.y * viewdir.y + first.z * viewdir.z,
							   first.y * viewdir.x + second.x * viewdir.y + second.y * viewdir.z,
							   first.z * viewdir.x + second.y * viewdir.y + second.z * viewdir.z };
	float num = (third.x * viewdir.x + third.y * viewdir.y + third.z * viewdir.z);
	float den = viewdir_inv_cov.x * viewdir.x + viewdir_inv_cov.y * viewdir.y + viewdir_inv_cov.z * viewdir.z;
	float rcp_den = __frcp_rn(max(0.00001f, den)); // since inv-cov is positive definite, den has to be > 0
	float depth = num * rcp_den;
	return depth;
}

template<class VEC3>
__device__ inline float depthAlongRay(const float4* cov3Ds_inv, const VEC3& viewdir)
{
	// symmetrix 3x3 matrix described by 6 values
	float4 first = cov3Ds_inv[0];
	float4 second = cov3Ds_inv[1];
	float4 third = cov3Ds_inv[2];

	return depthAlongRay<VEC3>(float3{ first.x, first.y, first.z }, float3{ second.x, second.y, second.z }, float3{ third.x, third.y, third.z }, viewdir);
}

template<class VEC3>
__device__ inline VEC3 computeViewRay(const glm::mat4 inverse_vp, const VEC3& campos, const float2& pix, const int W, const int H)
{
	const glm::vec3 p_world = pix2world(glm::vec2(pix.x, pix.y), W, H, inverse_vp);
	const glm::vec3 viewdir = glm::normalize(p_world - glm::vec3(campos.x, campos.y, campos.z));
	return { viewdir.x, viewdir.y, viewdir.z };
}

__device__ inline float evaluate_opacity_factor(const float dx, const float dy, const float4 co) 
{
	return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

__device__ inline float evaluate_opacity(const float dx, const float dy, const float4 co) 
{
	return co.w * expf(-evaluate_opacity_factor(dx, dy, co));
}

template<uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian(
	const float4 co, 
	const float2 mean, 
	const glm::vec2 rect_min,
	const glm::vec2 rect_max,
	glm::vec2& max_pos)
{
	//const float x_min_diff = rect_min.x - mean.x;
	const bool x_left = mean.x < rect_min.x; // x_min_diff > 0.0f;
	const bool in_x_range = !(x_left || mean.x > rect_max.x);

	// const float y_min_diff = rect_min.y - mean.y;
	const bool y_above =  mean.y < rect_min.y; // y_min_diff > 0.0f;
	const bool in_y_range = !(y_above || mean.y > rect_max.y);

	max_pos = {mean.x, mean.y};
	float max_contrib_power = 0.0f;

	if (!(in_y_range && in_x_range))
	{
		const glm::vec2 p = {
			x_left ?  rect_min.x : rect_max.x,
			y_above ? rect_min.y : rect_max.y
		};
		const glm::vec2 d = {
			x_left ?  float(PATCH_WIDTH)  : -float(PATCH_WIDTH), // copysign(float(PATCH_WIDTH), x_min_diff),
			y_above ? float(PATCH_HEIGHT) : -float(PATCH_HEIGHT) // copysign(float(PATCH_HEIGHT), y_min_diff)
		};

		const glm::vec2 diff = max_pos - p; // mean - p
		const glm::vec2 t_opt = {
			in_y_range ? 0.0f : __saturatef((d.x * co.x * diff.x + d.x * co.y * diff.y) / (d.x * d.x * co.x)),
			in_x_range ? 0.0f : __saturatef((d.y * co.y * diff.x + d.y * co.z * diff.y) / (d.y * d.y * co.z))
		};
		max_pos = p + t_opt * d;
		
		const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
		max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
	}

	return max_contrib_power;
}

template<uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
	const float4 co, 
	const float2 mean, 
	const glm::vec2 rect_min,
	const glm::vec2 rect_max,
	glm::vec2& max_pos)
{
	const float x_min_diff = rect_min.x - mean.x;
	const float x_left = x_min_diff > 0.0f;
	// const float x_left = mean.x < rect_min.x;
	const float not_in_x_range = x_left + (mean.x > rect_max.x);

	const float y_min_diff = rect_min.y - mean.y;
	const float y_above =  y_min_diff > 0.0f;
	// const float y_above = mean.y < rect_min.y;
	const float not_in_y_range = y_above + (mean.y > rect_max.y);

	max_pos = {mean.x, mean.y};
	float max_contrib_power = 0.0f;

	if ((not_in_y_range + not_in_x_range) > 0.0f)
	{
		const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
		const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

		const float dx = copysign(float(PATCH_WIDTH), x_min_diff);
		const float dy = copysign(float(PATCH_HEIGHT), y_min_diff);

		const float diffx = mean.x - px;
		const float diffy = mean.y - py;

		const float rcp_dxdxcox = __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x); // = 1.0 / (dx*dx*co.x)
		const float rcp_dydycoz = __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z); // = 1.0 / (dy*dy*co.z)

		const float tx = not_in_y_range * __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
		const float ty = not_in_x_range * __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);
		max_pos = {px + tx * dx, py + ty * dy};
		
		const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
		max_contrib_power = evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
	}

	return max_contrib_power;
}

template<bool LOAD_BALANCING, uint32_t SEQUENTIAL_TILE_THRESH = 32>
__device__ inline int computeTilebasedCullingTileCount(const bool active, 
	const float4 co_init, 
	const float2 xy_init, 
	const float opacity_power_threshold_init,
	const uint2 rect_min_init, 
	const uint2 rect_max_init)
{
	const int32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * (rect_max_init.x - rect_min_init.x);

	int tile_count = 0;
	if (active)
	{
		const uint32_t rect_width = (rect_max_init.x - rect_min_init.x);
		for (int tile_idx = 0; tile_idx < tile_count_init && (!LOAD_BALANCING || tile_idx < SEQUENTIAL_TILE_THRESH); tile_idx++)
		{
			const int y = (tile_idx / rect_width) + rect_min_init.y;
			const int x = (tile_idx % rect_width) + rect_min_init.x;

			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};

			glm::vec2 max_pos;
			float max_opac_factor = max_contrib_power_rect_gaussian_float<BLOCK_X-1, BLOCK_Y-1>(co_init, xy_init, tile_min, tile_max, max_pos);
			tile_count += (max_opac_factor <= opacity_power_threshold_init);
		}
	}

	if (!LOAD_BALANCING)
		return tile_count;
	
	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	const uint32_t warp_idx = cg::this_thread_block().thread_rank() / WARP_SIZE;

	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);
	if (remaining_threads == 0)
		return tile_count;

	const uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++)
	{
		const uint32_t i = __fns(remaining_threads, 0, n+1); // find lane index of next remaining thread

		const uint2 rect_min = make_uint2(__shfl_sync(WARP_MASK, rect_min_init.x, i), __shfl_sync(WARP_MASK, rect_min_init.y, i));
		const uint2 rect_max = make_uint2(__shfl_sync(WARP_MASK, rect_max_init.x, i), __shfl_sync(WARP_MASK, rect_max_init.y, i));
		const float2 xy = { __shfl_sync(WARP_MASK, xy_init.x, i), __shfl_sync(WARP_MASK, xy_init.y, i) };

		const float4 co = {
			__shfl_sync(WARP_MASK, co_init.x, i),
			__shfl_sync(WARP_MASK, co_init.y, i),
			__shfl_sync(WARP_MASK, co_init.z, i),
			__shfl_sync(WARP_MASK, co_init.w, i),
		};
		const float opacity_power_threshold = __shfl_sync(WARP_MASK, opacity_power_threshold_init, i);


		const uint32_t rect_width = (rect_max.x - rect_min.x);
		const uint32_t rect_tile_count = (rect_max.y - rect_min.y) * rect_width;
		const uint32_t remaining_rect_tile_count = rect_tile_count - SEQUENTIAL_TILE_THRESH;

		const int32_t n_iterations = (remaining_rect_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		for (int it = 0; it < n_iterations; it++)
		{
			const int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			const int active_curr_it = tile_idx < rect_tile_count;

			const int y = (tile_idx / rect_width) + rect_min.y;
			const int x = (tile_idx % rect_width) + rect_min.x;

			const glm::vec2 tile_min = {x * BLOCK_X, y * BLOCK_Y};
			const glm::vec2 tile_max = {(x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1};

			glm::vec2 max_pos;
			const float max_opac_factor = max_contrib_power_rect_gaussian_float<BLOCK_X-1, BLOCK_Y-1>(co, xy, tile_min, tile_max, max_pos);

			const uint32_t tile_contributes = active_curr_it && max_opac_factor <= opacity_power_threshold;

			const uint32_t contributes_ballot = __ballot_sync(WARP_MASK, tile_contributes);
			const uint32_t n_contribute = __popc(contributes_ballot);

			tile_count += (i == lane_idx) * n_contribute;
		}
	}

	return tile_count;
}

__device__ void __forceinline__ outputDebugVis(DebugVisualization debugType, float* out_color, int pix_id, uint32_t contributor, float T, float depth, float opacity, float distortion, float extent_loss, float depth_extent, int range, uint32_t max_contributor, int H, int W)
{
	if (debugType == DebugVisualization::GaussianCountPerPixel)
	{
		out_color[pix_id] = contributor;
	}
	else if (debugType == DebugVisualization::Transmittance)
	{
		out_color[pix_id] = 1 - T;
	}
	else if (debugType == DebugVisualization::GaussianCountPerTile)
	{
		out_color[pix_id] = range;
	}
	else if (debugType == DebugVisualization::Depth)
	{
		out_color[pix_id] = depth;
	}
	else if (debugType == DebugVisualization::Opacity)
	{
		out_color[pix_id] = opacity;
	}
	else if (debugType == DebugVisualization::Distortion)
	{
		out_color[pix_id] = distortion;
	}
	else if (debugType == DebugVisualization::EXTENT_LOSS)
	{
		out_color[pix_id] = extent_loss;
	}
	else if (debugType == DebugVisualization::Depth_Index)
	{
		out_color[pix_id] = max_contributor;
	}
	else if (debugType == DebugVisualization::DEPTH_EXTENT)
	{
		out_color[pix_id] = depth_extent;
	}
}

template <CudaRasterizer::GlobalSortOrder SORT_ORDER>
__device__ inline glm::vec2 getPerTileDepthTargetPos(const glm::vec2 tile_center, const float2 xy, const float4 co, const glm::vec2 max_pos, const float max_opac_factor)
{
	glm::vec2 target_pos;
	if constexpr (SORT_ORDER == CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_MAXPOS)
	{
		target_pos = max_pos;
	}
	else if constexpr (SORT_ORDER == CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_CENTER)
	{
		target_pos = tile_center;
	}
	return target_pos;
}

template<bool TILE_BASED_CULLING, bool LOAD_BALANCING = true, CudaRasterizer::GlobalSortOrder SORT_ORDER = CudaRasterizer::GlobalSortOrder::VIEWSPACE_Z>
__global__ void duplicateWithKeys_extended(
	int P,
	const float2* __restrict__ points_xy,
	const float* __restrict__ depths,
	const float4* __restrict__ cov3Ds_inv,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ projmatrix,
	const float* __restrict__ projmatrix_inv,
	const glm::vec3* __restrict__ cam_pos,
	const int W, const int H,
	const uint32_t* __restrict__  offsets,
	uint64_t* __restrict__ gaussian_keys_unsorted,
	uint32_t* __restrict__ gaussian_values_unsorted,
	const int* __restrict__ radii,
	const float2* __restrict__ rects,
	dim3 grid)
{	
	auto block = cg::this_thread_block();
	auto warp = cg::tiled_partition<WARP_SIZE>(block);
	constexpr bool EVAL_MAX_CONTRIB_POS = TILE_BASED_CULLING || SORT_ORDER == CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_MAXPOS;
	constexpr bool PER_TILE_DEPTH = SORT_ORDER == CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_MAXPOS ||
									SORT_ORDER == CudaRasterizer::GlobalSortOrder::PER_TILE_DEPTH_CENTER;

#define RETURN_OR_INACTIVE() if constexpr(LOAD_BALANCING) { active = false; } else { return; }
//#define DUPLICATE_OPT_DEBUG

	uint32_t idx = cg::this_grid().thread_rank();
	bool active = true;
	if (idx >= P) {
		RETURN_OR_INACTIVE();
		idx = P - 1;
	}

	const int radius = radii[idx];
	if (radius <= 0) {
		RETURN_OR_INACTIVE();
	}

	if constexpr(LOAD_BALANCING)
		if (__ballot_sync(WARP_MASK, active) == 0)
			return;

	// Find this Gaussian's offset in buffer for writing keys/values.
	uint32_t off_init = (idx == 0) ? 0 : offsets[idx - 1];

	const int offset_to_init = offsets[idx];		
	const float global_depth_init = depths[idx];

	const float2 xy_init = points_xy[idx];
	const float2 rect_dims_init = rects[idx];

	__shared__ float2 s_xy[BLOCK_SIZE];
	__shared__ float2 s_rect_dims[BLOCK_SIZE];
	s_xy[block.thread_rank()] = xy_init;
	s_rect_dims[block.thread_rank()] = rect_dims_init;

	uint2 rect_min_init, rect_max_init;
	getRect(xy_init, rect_dims_init, rect_min_init, rect_max_init, grid);


	constexpr size_t SHMEM_SIZE_COV3D_INV = size_t(PER_TILE_DEPTH) * BLOCK_SIZE + size_t(!PER_TILE_DEPTH);
	__shared__ float4 s_cov3D_inv_first[SHMEM_SIZE_COV3D_INV];
	__shared__ float4 s_cov3D_inv_second[SHMEM_SIZE_COV3D_INV];
	__shared__ float4 s_cov3D_inv_third[SHMEM_SIZE_COV3D_INV];
	if (PER_TILE_DEPTH)
	{
		s_cov3D_inv_first[block.thread_rank()] = cov3Ds_inv[3 * idx + 0];
		s_cov3D_inv_second[block.thread_rank()] = cov3Ds_inv[3 * idx + 1];
		s_cov3D_inv_third[block.thread_rank()] = cov3Ds_inv[3 * idx + 2];
	}

	constexpr size_t SHMEM_SIZE_CONIC_OPACITY = size_t(EVAL_MAX_CONTRIB_POS) * BLOCK_SIZE + size_t(!EVAL_MAX_CONTRIB_POS);
	__shared__ float4 s_co[SHMEM_SIZE_CONIC_OPACITY];
	if (EVAL_MAX_CONTRIB_POS)
	{
		s_co[block.thread_rank()] = conic_opacity[idx];
	}

	constexpr uint32_t SEQUENTIAL_TILE_THRESH = 32U; // all tiles above this threshold will be computed cooperatively
	const uint32_t rect_width_init = (rect_max_init.x - rect_min_init.x);
	const uint32_t tile_count_init = (rect_max_init.y - rect_min_init.y) * rect_width_init;

	// Generate no key/value pair for invisible Gaussians
	if (tile_count_init == 0)	{
		RETURN_OR_INACTIVE();
	}

	const glm::mat4 inversed_vp = loadMatrix4x4(projmatrix_inv);
	const glm::vec4 inversed_vp0 = inversed_vp[0];
	const glm::vec4 inversed_vp1 = inversed_vp[1];
	const glm::vec4 inversed_vp3 = inversed_vp[3];

	const glm::vec3 cam_pos_tmp = *cam_pos;

	auto tile_function = [&](int x, int y,
							 const float2 xy,
							 const float global_depth,
							 const float4 co,
							 const float opacity_factor_threshold,
							 const float3 cov3D_inv1,
							 const float3 cov3D_inv2,
							 const float3 cov3D_inv3,
							 float& depth)
		{
			const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
			const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1);

			glm::vec2 max_pos;
			float max_opac_factor = 0.0f;
			if constexpr (EVAL_MAX_CONTRIB_POS)
			{
				max_opac_factor = max_contrib_power_rect_gaussian_float<BLOCK_X-1, BLOCK_Y-1>(co, xy, tile_min, tile_max, max_pos);
			}

			if constexpr (PER_TILE_DEPTH) 
			{
				const glm::vec2 tile_center = (tile_min + tile_max) * 0.5f;
				const glm::vec2 target_pos = getPerTileDepthTargetPos<SORT_ORDER>(tile_center, xy, co, max_pos, max_opac_factor);
				
				const glm::vec3 p_world = pix2world(target_pos, W, H, inversed_vp0, inversed_vp1, inversed_vp3);
				const glm::vec3 viewdir = glm::normalize(p_world - cam_pos_tmp);
				// TODO: this may destroy training
				// depth may be negative depending on how the camera is positioned, which will destroy our sorting due to mixing it with int
				depth = max(0.0f, depthAlongRay(cov3D_inv1, cov3D_inv2, cov3D_inv3, viewdir) + 8.0f);
			}
			else
			{
				depth = global_depth;
			}

			return (!TILE_BASED_CULLING) || max_opac_factor <= opacity_factor_threshold;
		};

	if (active)
	{
		float3 cov3D_inv_first_init, cov3D_inv_second_init, cov3D_inv_third_init;
		if (PER_TILE_DEPTH)
		{
			cov3D_inv_first_init = make_float3(s_cov3D_inv_first[block.thread_rank()]);
			cov3D_inv_second_init = make_float3(s_cov3D_inv_second[block.thread_rank()]);
			cov3D_inv_third_init = make_float3(s_cov3D_inv_third[block.thread_rank()]);
		}

		float4 co_init;
		float opacity_factor_threshold_init;
		if (EVAL_MAX_CONTRIB_POS)
		{
			co_init = s_co[block.thread_rank()];
			opacity_factor_threshold_init = logf(co_init.w / ALPHA_THRESHOLD);
		}

		for (uint32_t tile_idx = 0; tile_idx < tile_count_init && (!LOAD_BALANCING || tile_idx < SEQUENTIAL_TILE_THRESH); tile_idx++)
		{
			const int y = (tile_idx / rect_width_init) + rect_min_init.y;
			const int x = (tile_idx % rect_width_init) + rect_min_init.x;

			float depth;
			bool write_tile = tile_function(x, y, xy_init, global_depth_init, co_init, opacity_factor_threshold_init,
											cov3D_inv_first_init, cov3D_inv_second_init, cov3D_inv_third_init, depth);

			if (write_tile)
			{
				if (off_init < offset_to_init)
				{
					const uint32_t tile_id = y * grid.x + x;
					gaussian_values_unsorted[off_init] = idx;
					gaussian_keys_unsorted[off_init] = constructSortKey(tile_id, depth);
				}
				else
				{
#ifdef DUPLICATE_OPT_DEBUG
					printf("Error (sequential): Too little memory reserved in preprocess: off=%d off_to=%d idx=%d\n", off_init, offset_to_init, idx);
#endif
				}
				off_init++;
			}
		}

		// fill in missing keys - can happen due to float inaccuracies
		for (; off_init < offset_to_init && (!LOAD_BALANCING || tile_count_init <= SEQUENTIAL_TILE_THRESH); off_init++)
		{
			gaussian_values_unsorted[off_init] = static_cast<uint32_t>(-1);
			gaussian_keys_unsorted[off_init] = constructSortKey(INVALID_TILE_ID, FLT_MAX);
		}
	}

#undef RETURN_OR_INACTIVE

	if (!LOAD_BALANCING)
		return;
	
	const uint32_t idx_init = idx;
	const uint32_t lane_idx = cg::this_thread_block().thread_rank() % WARP_SIZE;
	const uint32_t warp_idx = cg::this_thread_block().thread_rank() / WARP_SIZE;
	unsigned int lane_mask_allprev_excl = 0xFFFFFFFFU >> (WARP_SIZE - lane_idx);

	const int32_t compute_cooperatively = active && tile_count_init > SEQUENTIAL_TILE_THRESH;
	const uint32_t remaining_threads = __ballot_sync(WARP_MASK, compute_cooperatively);
	if (remaining_threads == 0)
		return;

	uint32_t n_remaining_threads = __popc(remaining_threads);
	for (int n = 0; n < n_remaining_threads && n < WARP_SIZE; n++)
	{
		int i = __fns(remaining_threads, 0, n+1); // find lane index of next remaining thread

		uint32_t idx_coop = __shfl_sync(WARP_MASK, idx_init, i);
		uint32_t off_coop = __shfl_sync(WARP_MASK, off_init, i);
		
		const uint32_t offset_to = __shfl_sync(WARP_MASK, offset_to_init, i);
		const float global_depth = __shfl_sync(WARP_MASK, global_depth_init, i);

		const float2 xy = s_xy[warp.meta_group_rank() * WARP_SIZE + i];
		const float2 rect_dims = s_rect_dims[warp.meta_group_rank() * WARP_SIZE + i];

		float3 cov3D_inv_first, cov3D_inv_second, cov3D_inv_third;
		if (PER_TILE_DEPTH)
		{
			cov3D_inv_first = make_float3(s_cov3D_inv_first[warp.meta_group_rank() * WARP_SIZE + i]);
			cov3D_inv_second = make_float3(s_cov3D_inv_second[warp.meta_group_rank() * WARP_SIZE + i]);
			cov3D_inv_third = make_float3(s_cov3D_inv_third[warp.meta_group_rank() * WARP_SIZE + i]);
		}

		float4 co;
		float opacity_factor_threshold;
		if (EVAL_MAX_CONTRIB_POS)
		{
			co = s_co[warp.meta_group_rank() * WARP_SIZE + i];
			opacity_factor_threshold = logf(co.w / ALPHA_THRESHOLD);
		}

		uint2 rect_min, rect_max;
		getRect(xy, rect_dims, rect_min, rect_max, grid);

		const uint32_t rect_width = (rect_max.x - rect_min.x);
		const uint32_t tile_count = (rect_max.y - rect_min.y) * rect_width;
		const uint32_t remaining_tile_count = tile_count - SEQUENTIAL_TILE_THRESH;
		
		const int32_t n_iterations = (remaining_tile_count + WARP_SIZE - 1) / WARP_SIZE;
		for (int it = 0; it < n_iterations; it++)
		{
			int tile_idx = it * WARP_SIZE + lane_idx + SEQUENTIAL_TILE_THRESH;
			int active_curr_it = tile_idx < tile_count;

			int y = (tile_idx / rect_width) + rect_min.y;
			int x = (tile_idx % rect_width) + rect_min.x;

			float depth;
			bool write_tile = tile_function(x, y, xy, global_depth, co, opacity_factor_threshold,
											cov3D_inv_first, cov3D_inv_second, cov3D_inv_third, depth);
			
			const uint32_t write = active_curr_it && write_tile;

			uint32_t n_writes, write_offset;
			if constexpr (!TILE_BASED_CULLING)
			{
				n_writes = WARP_SIZE;
				write_offset = off_coop + lane_idx;
			}
			else
			{
				const uint32_t write_ballot = __ballot_sync(WARP_MASK, write);
				n_writes = __popc(write_ballot);

				const uint32_t write_offset_it = __popc(write_ballot & lane_mask_allprev_excl);
				write_offset = off_coop + write_offset_it;
			}

			if (write)
			{
				if (write_offset < offset_to)
				{
					const uint32_t tile_id = y * grid.x + x;
					gaussian_values_unsorted[write_offset] = idx_coop;
					gaussian_keys_unsorted[write_offset] = constructSortKey(tile_id, depth);
				}
#ifdef DUPLICATE_OPT_DEBUG
				else
				{
					printf("Error (parallel): Too little memory reserved in preprocess: off=%d off_to=%d idx=%d tile_count=%d it=%d | x=%d y=%d rect=(%d %d - %d %d)\n", 
						   write_offset, offset_to, idx_coop, tile_count, it, x, y, rect_min.x, rect_min.y, rect_max.x, rect_max.y);
				}
#endif
			}
			off_coop += n_writes;
		}

		__syncwarp();

		// fill in missing keys - can happen due to float inaccuracies or larger threshold in preprocess
		for (int off_coop_i = off_coop + lane_idx; off_coop_i < offset_to; off_coop_i += WARP_SIZE)
		{
			gaussian_values_unsorted[off_coop_i] = static_cast<uint32_t>(-1);
			gaussian_keys_unsorted[off_coop_i] = constructSortKey(INVALID_TILE_ID, FLT_MAX);
		}
	}
}

__device__ inline glm::vec3 colormapMagma(float x)
{
	const glm::vec3 c_magma[] = {
		{-0.002136485053939582f, -0.000749655052795221f, -0.005386127855323933f},
		{0.2516605407371642f, 0.6775232436837668f, 2.494026599312351f},
		{8.353717279216625f, -3.577719514958484f, 0.3144679030132573f},
		{-27.66873308576866f, 14.26473078096533f, -13.64921318813922f},
		{52.17613981234068f, -27.94360607168351f, 12.94416944238394f},
		{-50.76852536473588f, 29.04658282127291f, 4.23415299384598f},
		{18.65570506591883f, -11.48977351997711f, -5.601961508734096f}
	};
	x = glm::clamp(x, 0.f, 1.f);
	glm::vec3 res = (c_magma[0]+x*(c_magma[1]+x*(c_magma[2]+x*(c_magma[3]+x*(c_magma[4]+x*(c_magma[5]+c_magma[6]*x))))));
	return glm::vec3(
		glm::clamp(res[0], 0.f, 1.f),
		glm::clamp(res[1], 0.f, 1.f),
		glm::clamp(res[2], 0.f, 1.f)
	);
}

// supporting the TURBO depth colormap of google (https://blog.research.google/2019/08/turbo-improved-rainbow-colormap-for.html?m=1)
// somewhat adapted from https://gist.github.com/mikhailov-work/ee72ba4191942acecc03fe6da94fc73f
__device__ inline glm::vec3 colormapTurbo(float x) {
	float turbo_srgb_floats[256][3] = {{0.18995,0.07176,0.23217},{0.19483,0.08339,0.26149},{0.19956,0.09498,0.29024},{0.20415,0.10652,0.31844},{0.20860,0.11802,0.34607},{0.21291,0.12947,0.37314},{0.21708,0.14087,0.39964},{0.22111,0.15223,0.42558},{0.22500,0.16354,0.45096},{0.22875,0.17481,0.47578},{0.23236,0.18603,0.50004},{0.23582,0.19720,0.52373},{0.23915,0.20833,0.54686},{0.24234,0.21941,0.56942},{0.24539,0.23044,0.59142},{0.24830,0.24143,0.61286},{0.25107,0.25237,0.63374},{0.25369,0.26327,0.65406},{0.25618,0.27412,0.67381},{0.25853,0.28492,0.69300},{0.26074,0.29568,0.71162},{0.26280,0.30639,0.72968},{0.26473,0.31706,0.74718},{0.26652,0.32768,0.76412},{0.26816,0.33825,0.78050},{0.26967,0.34878,0.79631},{0.27103,0.35926,0.81156},{0.27226,0.36970,0.82624},{0.27334,0.38008,0.84037},{0.27429,0.39043,0.85393},{0.27509,0.40072,0.86692},{0.27576,0.41097,0.87936},{0.27628,0.42118,0.89123},{0.27667,0.43134,0.90254},{0.27691,0.44145,0.91328},{0.27701,0.45152,0.92347},{0.27698,0.46153,0.93309},{0.27680,0.47151,0.94214},{0.27648,0.48144,0.95064},{0.27603,0.49132,0.95857},{0.27543,0.50115,0.96594},{0.27469,0.51094,0.97275},{0.27381,0.52069,0.97899},{0.27273,0.53040,0.98461},{0.27106,0.54015,0.98930},{0.26878,0.54995,0.99303},{0.26592,0.55979,0.99583},{0.26252,0.56967,0.99773},{0.25862,0.57958,0.99876},{0.25425,0.58950,0.99896},{0.24946,0.59943,0.99835},{0.24427,0.60937,0.99697},{0.23874,0.61931,0.99485},{0.23288,0.62923,0.99202},{0.22676,0.63913,0.98851},{0.22039,0.64901,0.98436},{0.21382,0.65886,0.97959},{0.20708,0.66866,0.97423},{0.20021,0.67842,0.96833},{0.19326,0.68812,0.96190},{0.18625,0.69775,0.95498},{0.17923,0.70732,0.94761},{0.17223,0.71680,0.93981},{0.16529,0.72620,0.93161},{0.15844,0.73551,0.92305},{0.15173,0.74472,0.91416},{0.14519,0.75381,0.90496},{0.13886,0.76279,0.89550},{0.13278,0.77165,0.88580},{0.12698,0.78037,0.87590},{0.12151,0.78896,0.86581},{0.11639,0.79740,0.85559},{0.11167,0.80569,0.84525},{0.10738,0.81381,0.83484},{0.10357,0.82177,0.82437},{0.10026,0.82955,0.81389},{0.09750,0.83714,0.80342},{0.09532,0.84455,0.79299},{0.09377,0.85175,0.78264},{0.09287,0.85875,0.77240},{0.09267,0.86554,0.76230},{0.09320,0.87211,0.75237},{0.09451,0.87844,0.74265},{0.09662,0.88454,0.73316},{0.09958,0.89040,0.72393},{0.10342,0.89600,0.71500},{0.10815,0.90142,0.70599},{0.11374,0.90673,0.69651},{0.12014,0.91193,0.68660},{0.12733,0.91701,0.67627},{0.13526,0.92197,0.66556},{0.14391,0.92680,0.65448},{0.15323,0.93151,0.64308},{0.16319,0.93609,0.63137},{0.17377,0.94053,0.61938},{0.18491,0.94484,0.60713},{0.19659,0.94901,0.59466},{0.20877,0.95304,0.58199},{0.22142,0.95692,0.56914},{0.23449,0.96065,0.55614},{0.24797,0.96423,0.54303},{0.26180,0.96765,0.52981},{0.27597,0.97092,0.51653},{0.29042,0.97403,0.50321},{0.30513,0.97697,0.48987},{0.32006,0.97974,0.47654},{0.33517,0.98234,0.46325},{0.35043,0.98477,0.45002},{0.36581,0.98702,0.43688},{0.38127,0.98909,0.42386},{0.39678,0.99098,0.41098},{0.41229,0.99268,0.39826},{0.42778,0.99419,0.38575},{0.44321,0.99551,0.37345},{0.45854,0.99663,0.36140},{0.47375,0.99755,0.34963},{0.48879,0.99828,0.33816},{0.50362,0.99879,0.32701},{0.51822,0.99910,0.31622},{0.53255,0.99919,0.30581},{0.54658,0.99907,0.29581},{0.56026,0.99873,0.28623},{0.57357,0.99817,0.27712},{0.58646,0.99739,0.26849},{0.59891,0.99638,0.26038},{0.61088,0.99514,0.25280},{0.62233,0.99366,0.24579},{0.63323,0.99195,0.23937},{0.64362,0.98999,0.23356},{0.65394,0.98775,0.22835},{0.66428,0.98524,0.22370},{0.67462,0.98246,0.21960},{0.68494,0.97941,0.21602},{0.69525,0.97610,0.21294},{0.70553,0.97255,0.21032},{0.71577,0.96875,0.20815},{0.72596,0.96470,0.20640},{0.73610,0.96043,0.20504},{0.74617,0.95593,0.20406},{0.75617,0.95121,0.20343},{0.76608,0.94627,0.20311},{0.77591,0.94113,0.20310},{0.78563,0.93579,0.20336},{0.79524,0.93025,0.20386},{0.80473,0.92452,0.20459},{0.81410,0.91861,0.20552},{0.82333,0.91253,0.20663},{0.83241,0.90627,0.20788},{0.84133,0.89986,0.20926},{0.85010,0.89328,0.21074},{0.85868,0.88655,0.21230},{0.86709,0.87968,0.21391},{0.87530,0.87267,0.21555},{0.88331,0.86553,0.21719},{0.89112,0.85826,0.21880},{0.89870,0.85087,0.22038},{0.90605,0.84337,0.22188},{0.91317,0.83576,0.22328},{0.92004,0.82806,0.22456},{0.92666,0.82025,0.22570},{0.93301,0.81236,0.22667},{0.93909,0.80439,0.22744},{0.94489,0.79634,0.22800},{0.95039,0.78823,0.22831},{0.95560,0.78005,0.22836},{0.96049,0.77181,0.22811},{0.96507,0.76352,0.22754},{0.96931,0.75519,0.22663},{0.97323,0.74682,0.22536},{0.97679,0.73842,0.22369},{0.98000,0.73000,0.22161},{0.98289,0.72140,0.21918},{0.98549,0.71250,0.21650},{0.98781,0.70330,0.21358},{0.98986,0.69382,0.21043},{0.99163,0.68408,0.20706},{0.99314,0.67408,0.20348},{0.99438,0.66386,0.19971},{0.99535,0.65341,0.19577},{0.99607,0.64277,0.19165},{0.99654,0.63193,0.18738},{0.99675,0.62093,0.18297},{0.99672,0.60977,0.17842},{0.99644,0.59846,0.17376},{0.99593,0.58703,0.16899},{0.99517,0.57549,0.16412},{0.99419,0.56386,0.15918},{0.99297,0.55214,0.15417},{0.99153,0.54036,0.14910},{0.98987,0.52854,0.14398},{0.98799,0.51667,0.13883},{0.98590,0.50479,0.13367},{0.98360,0.49291,0.12849},{0.98108,0.48104,0.12332},{0.97837,0.46920,0.11817},{0.97545,0.45740,0.11305},{0.97234,0.44565,0.10797},{0.96904,0.43399,0.10294},{0.96555,0.42241,0.09798},{0.96187,0.41093,0.09310},{0.95801,0.39958,0.08831},{0.95398,0.38836,0.08362},{0.94977,0.37729,0.07905},{0.94538,0.36638,0.07461},{0.94084,0.35566,0.07031},{0.93612,0.34513,0.06616},{0.93125,0.33482,0.06218},{0.92623,0.32473,0.05837},{0.92105,0.31489,0.05475},{0.91572,0.30530,0.05134},{0.91024,0.29599,0.04814},{0.90463,0.28696,0.04516},{0.89888,0.27824,0.04243},{0.89298,0.26981,0.03993},{0.88691,0.26152,0.03753},{0.88066,0.25334,0.03521},{0.87422,0.24526,0.03297},{0.86760,0.23730,0.03082},{0.86079,0.22945,0.02875},{0.85380,0.22170,0.02677},{0.84662,0.21407,0.02487},{0.83926,0.20654,0.02305},{0.83172,0.19912,0.02131},{0.82399,0.19182,0.01966},{0.81608,0.18462,0.01809},{0.80799,0.17753,0.01660},{0.79971,0.17055,0.01520},{0.79125,0.16368,0.01387},{0.78260,0.15693,0.01264},{0.77377,0.15028,0.01148},{0.76476,0.14374,0.01041},{0.75556,0.13731,0.00942},{0.74617,0.13098,0.00851},{0.73661,0.12477,0.00769},{0.72686,0.11867,0.00695},{0.71692,0.11268,0.00629},{0.70680,0.10680,0.00571},{0.69650,0.10102,0.00522},{0.68602,0.09536,0.00481},{0.67535,0.08980,0.00449},{0.66449,0.08436,0.00424},{0.65345,0.07902,0.00408},{0.64223,0.07380,0.00401},{0.63082,0.06868,0.00401},{0.61923,0.06367,0.00410},{0.60746,0.05878,0.00427},{0.59550,0.05399,0.00453},{0.58336,0.04931,0.00486},{0.57103,0.04474,0.00529},{0.55852,0.04028,0.00579},{0.54583,0.03593,0.00638},{0.53295,0.03169,0.00705},{0.51989,0.02756,0.00780},{0.50664,0.02354,0.00863},{0.49321,0.01963,0.00955},{0.47960,0.01583,0.01055}};
	
	float interp = glm::clamp(x * 255.f, 0.f, 255.f);
	int floor = x > 0 ? (int)interp : 0;
	int ceil = floor >= 255 ? 255 : floor + 1;
	float diff = interp - floor;

	return glm::vec3(
		glm::clamp(turbo_srgb_floats[floor][0] + (turbo_srgb_floats[ceil][0] - turbo_srgb_floats[floor][0]) * diff, 0.f, 1.f),
		glm::clamp(turbo_srgb_floats[floor][1] + (turbo_srgb_floats[ceil][1] - turbo_srgb_floats[floor][1]) * diff, 0.f, 1.f),
		glm::clamp(turbo_srgb_floats[floor][2] + (turbo_srgb_floats[ceil][2] - turbo_srgb_floats[floor][2]) * diff, 0.f, 1.f)
	);
}