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

#include "rasterizer_impl.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"
#include "stopthepop/stopthepop_common.cuh"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

void applyDebugVisualization(
	DebugVisualizationData& debugVis,
	int width, int height,
	CudaRasterizer::ImageState& imgState,
	CudaRasterizer::BinningState& binningState,
	const float2* means2D,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float* scales,
	const float* rotations,
	float* out_color,
	bool debug
)
{
	if (debugVis.type != DebugVisualization::Disabled)
	{
		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
		float* d_min_max; // GPU pointer for the result
		cudaMalloc((void**)&d_min_max, sizeof(float) * 2);

		int N = width * height;

		cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, out_color, d_min_max + 1, N);
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
		cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, out_color, d_min_max, N);
		cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, out_color, d_min_max + 1, N);

		std::array<float, 2> min_max_contribution_count;
		cudaMemcpy(min_max_contribution_count.data(), d_min_max, 2 * sizeof(float), cudaMemcpyDeviceToHost);

		float value = 0;
		if (debugVis.debugX >= 0 && debugVis.debugX < width && debugVis.debugY >= 0 && debugVis.debugY < height)
		{
			uint32_t pix_id = width * debugVis.debugY + debugVis.debugX;
			cudaMemcpy(&value, out_color + pix_id, sizeof(float), cudaMemcpyDeviceToHost);
		}

		// Statistics
		// Avg, STD
		std::vector<float> data(N);
		cudaMemcpy(data.data(), out_color, sizeof(float) * N, cudaMemcpyDeviceToHost);

		float sum = std::accumulate(data.begin(), data.end(), 0.0f, std::plus<float>());
		float average = sum / static_cast<float>(N);
		float std = std::sqrt(std::accumulate(data.begin(), data.end(), 0.f, [average](float v, float n) {
				return v + ((n - average) * (n - average));
				}) / static_cast<float>(N));

		if (debugVis.printing_enabled) {
			std::cout << std::fixed << std::setprecision(debugVis.precision);
			std::cout << "\033[31m" << toString(debugVis.type) << "\033[0m" << " for (" << debugVis.debugX << ", " << debugVis.debugY <<
			"): value=" << value << ", min=" << min_max_contribution_count[0] << ", max=" << min_max_contribution_count[1] << ", avg=" << average << ", std=" << std << std::endl;	
		}
		
		CHECK_CUDA(FORWARD::render_debug(debugVis, width * height, out_color, d_min_max), debug)
		
		cudaFree(d_min_max);
	}
 }

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	const glm::vec3 mean3D(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	const glm::mat4x3 viewmatrix_mat = loadMatrix4x3(viewmatrix);

	glm::vec3 p_view;
	present[idx] = in_frustum(idx, mean3D, viewmatrix_mat, false, p_view);
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	bool valid_tile = currtile != INVALID_TILE_ID;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			if (valid_tile) 
				ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1 && valid_tile)
		ranges[currtile].y = L;
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyPixelRanges(const int L, const uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint32_t currtile = point_list_keys[idx] >> 32;

	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileLaunches(const int T, const uint2* ranges, uint32_t* tiles_to_launch)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;

	uint2 range = ranges[idx];
	int todo = range.y - range.x;

	if (todo > 0)
	{
		tiles_to_launch[idx] = (uint32_t)((todo + BLOCK_SIZE - 1) / BLOCK_SIZE);
	}
	else {
		tiles_to_launch[idx] = 0;
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void writeTileLaunches(	const int T, const uint32_t* tiles_to_write, const uint32_t* tile_write_location_inc,
									uint2* tile_idx_to_gaussian_range)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= T)
		return;

	uint32_t num_writes = tiles_to_write[idx];
	uint2* location = tile_idx_to_gaussian_range;

	if (idx != 0) {
		// we computed an inclusive scan, hence, we just start writing from index 0 here
		location += tile_write_location_inc[idx-1];
	}

	for (uint32_t i = 0; i < num_writes; i++) {
		location[i].x = idx;
		location[i].y = i * 256;
	}

#ifdef DEBUG_TILE_LAUNCHES
	printf("Thread %u writing %u starting from location (%p + %u)\n", idx, num_writes, tile_idx_to_gaussian_range,tile_write_location_inc[idx-1]);
#endif
}


// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

__global__ void compute_filter_3d(
    int P, // num_points
    int C, // num_cameras
    const glm::vec3* means3D,
    const float* viewmatrices,
    const int W, const int H,
    const float focal_x, const float focal_y,
    float* filter_3D) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= P) return;
    
    float distance = 100000.f;

    glm::vec4 mean3D(means3D[idx], 1.0f);

    for (int i = 0; i < C; i++) {
        glm::mat4x3 viewmatrix_mat = loadMatrix4x3(viewmatrices + i * 16);
        glm::vec3 p_view = viewmatrix_mat * mean3D;

        float x = p_view.x / p_view.z * focal_x + W / 2.f;
        float y = p_view.y / p_view.z * focal_y + H / 2.f;

        if (x >= -0.15f * W &&
            x <=  1.15f * W &&
            y >= -0.15f * H &&
            y <=  1.15f * H &&
            p_view.z > 0.2f
        ) 
        {
            distance = fminf(distance, p_view.z);
        }
    }
	if (distance != 100000.f)
		filter_3D[idx] = distance / focal_x * powf(0.2f, 0.5f);

}


void CudaRasterizer::Rasterizer::Compute3DFilter(
    int P, // num_points
    int C, // num_cameras
    const float* means3D,
    const float* viewmatrices,
    const int W, const int H,
    const float focal_x, const float focal_y,
    float* filter_3D)
{
	int num_blocks = (P + 255) / 256;
	dim3 block(256, 1, 1);
	dim3 grid(num_blocks, 1, 1);
	compute_filter_3d<<<grid, block>>>(P, C, (glm::vec3*)means3D, viewmatrices, W, H, focal_x, focal_y, filter_3D);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P, bool requires_cov3D_inv)
{
	GeometryState geom;

	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.rects2D, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.view2gaussian, P * 10, 128);
	if (requires_cov3D_inv)
		obtain(chunk, geom.cov3D_inv, P * 3, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N * 4, 128);
	obtain(chunk, img.n_contrib, N * 2, 128);
	obtain(chunk, img.ranges, N, 128);
	obtain(chunk, img.point_ranges, N, 128);
	obtain(chunk, img.tile_launch_ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::PointBinningState CudaRasterizer::PointBinningState::fromChunk(char*& chunk, size_t PN)
{
	PointBinningState binning;
	obtain(chunk, binning.point_list, PN, 128);
	obtain(chunk, binning.point_list_unsorted, PN, 128);
	obtain(chunk, binning.point_list_keys, PN, 128);
	obtain(chunk, binning.point_list_keys_unsorted, PN, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, PN);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::PointState CudaRasterizer::PointState::fromChunk(char*& chunk, size_t P)
{
	PointState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.points2D, P, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}
// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const SplattingSettings splatting_settings,
	DebugVisualizationData& debugVisualization,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* view2gaussian_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* inv_viewprojmatrix,
	const float* filter_3d,
	const float* cam_pos,
	const float tan_fovx, const float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii,
	bool debug)
{
	static Timer timer({ "FW Preprocess", "FW Render", "FW Opacity" });
	timer.setActive(debugVisualization.timing_enabled);

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	bool requires_cov3D_inv = splatting_settings.sort_settings.requiresDepthAlongRay();
	size_t chunk_size = required<GeometryState>(P, requires_cov3D_inv);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, requires_cov3D_inv);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}
	timer();

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		view2gaussian_precomp,
		filter_3d,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.rects2D,
		splatting_settings,
		debugVisualization,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.cov3D_inv,
		geomState.view2gaussian,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	FORWARD::duplicate(
		P,
		geomState.means2D,
		geomState.conic_opacity,
		radii,
		geomState.rects2D,
		geomState.point_offsets,
		geomState.depths,
		geomState.cov3D_inv,
		splatting_settings,
		projmatrix,
		inv_viewprojmatrix,
		cam_pos,
		width, height,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		tile_grid);
	CHECK_CUDA(, debug)

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)

	timer();

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* view2gaussian = view2gaussian_precomp != nullptr ? view2gaussian_precomp : geomState.view2gaussian;
	CHECK_CUDA(FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		splatting_settings,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		geomState.means2D,
		view2gaussian,
		means3D,
		geomState.cov3D_inv,
		inv_viewprojmatrix,
		(glm::vec3*)cam_pos,
		feature_ptr,
		geomState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		debugVisualization,
		out_color), debug)

		timer();

		if (splatting_settings.render_opacity && 
			(splatting_settings.sort_settings.sort_mode == SortMode::HIERARCHICAL || (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL && debugVisualization.type == DebugVisualization::Opacity)) 
		) {
			CHECK_CUDA(FORWARD::render_opacity(
				tile_grid, block,
				imgState.ranges,
				splatting_settings,
				binningState.point_list,
				width, height,
				focal_x, focal_y,
				geomState.means2D,
				view2gaussian,
				means3D,
				geomState.cov3D_inv,
				inv_viewprojmatrix,
				(glm::vec3*)cam_pos,
				feature_ptr,
				geomState.depths,
				geomState.conic_opacity,
				imgState.accum_alpha,
				imgState.n_contrib,
				background,
				debugVisualization,
				out_color), debug)					
		}

	timer();

	std::vector<std::pair<std::string, float>> timings;
	timer.syncAddReport(timings);

	if (timings.size() > 0)
	{
		std::stringstream ss;
		ss << "Timings: \n";
		for (auto const& x : timings)
			ss << " - " << x.first << ": " << x.second << "ms\n";
		std::cout << ss.str() << std::endl;
	}

	applyDebugVisualization(
		debugVisualization,
		width, height,
		imgState, binningState, geomState.means2D,
		viewmatrix, projmatrix, cam_pos,
		scales, rotations,
		out_color,
		debug
	);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const SplattingSettings splatting_settings,
	const float* means3D,
	const float* shs,
    const float* opacities,
	const float* colors_precomp,
	const float* view2gaussian_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* inv_viewprojmatrix,
	const float* filter_3d,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const float* pixel_colors,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot,
	float* dL_dview2gaussian,
	bool debug)
{
	bool requires_cov3D_inv = splatting_settings.sort_settings.requiresDepthAlongRay();
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P, requires_cov3D_inv);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	const float* view2gaussian_ptr = (view2gaussian_precomp != nullptr) ? view2gaussian_precomp : geomState.view2gaussian;
	CHECK_CUDA(BACKWARD::render(
		tile_grid, block,
		imgState.ranges,
		splatting_settings,
		binningState.point_list,
		width, height,
		focal_x, focal_y,
		background,
		geomState.means2D,
		geomState.cov3D_inv,
		inv_viewprojmatrix,
		(glm::vec3*)cam_pos,
		geomState.conic_opacity,
		color_ptr,
		view2gaussian_ptr,
		viewmatrix,
		imgState.accum_alpha,
		imgState.n_contrib,
		pixel_colors,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor,
		dL_dview2gaussian), debug)

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	CHECK_CUDA(BACKWARD::preprocess(P, D, M,
		splatting_settings.proper_ewa_scaling,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		opacities,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		view2gaussian_ptr,
		filter_3d,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)cam_pos,
		(float3*)dL_dmean2D,
		dL_dconic,
		dL_dview2gaussian,
		dL_dopacity,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot), debug)
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect_GOF(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

__global__ void createWithKeys(
	int PN,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* points_keys_unsorted,
	uint32_t* points_values_unsorted,
	dim3 grid,
	int WIDTH)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= PN)
		return;

	float depth = depths[idx];
	if (depth > 0.f)
	{
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
	
		// determine the pixel that the point is in
		const float2 p = points_xy[idx];
		uint32_t x = min(grid.x - 1, max((uint32_t)0, (uint32_t) floorf(p.x / BLOCK_X)));
		uint32_t y = min(grid.y - 1, max((uint32_t)0, (uint32_t) floorf(p.y / BLOCK_Y)));

// #ifdef DEBUG_INTEGRATE
// 		printf("pixel %f/%f -> Tile %d/%d\n", p.x, p.y, x, y);
// #endif
	
		// key is the tile_id
		uint64_t key = y * grid.x + x;
		key <<= 32;
		key |= *((uint32_t*)&depths[idx]);
	
		points_keys_unsorted[off] = key;
		points_values_unsorted[off] = idx;
	}
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::integrate(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> pointBuffer,
	std::function<char* (size_t)> point_binningBuffer,
	const int PN, const int P, int D, int M,
	const float* background,
	const int width, int height,
	SplattingSettings splatting_settings,
	DebugVisualizationData& debugVisualization,
	const float* points3D,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* view2gaussian_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* inv_viewprojmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii, // remove 
	float* out_alpha_integrated,
	float* out_color_integrated,
	bool debug)
{
	static Timer timer({ "Preprocess Points", "Preprocess Gaussians", "Integrate" }, 25);
	timer.setActive(true);
	timer();

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P, true);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, true);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X, 
		(height + BLOCK_Y - 1) / BLOCK_Y, 
	1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	size_t point_chunk_size = required<PointState>(PN);
	char* point_chunkptr = pointBuffer(point_chunk_size);
	PointState pointState = PointState::fromChunk(point_chunkptr, PN);
	// Run preprocessing per-Point (transformation)
	CHECK_CUDA(FORWARD::PreprocessPoints(
		PN, D, M,
		points3D,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		pointState.points2D,
		out_alpha_integrated,
		pointState.depths,
		tile_grid,
		pointState.tiles_touched,
		out_alpha_integrated, 
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Points
	// E.g., [1, 1, 0, 1, 0] -> [1, 2, 2, 3, 3]
	// TODO: 	could this be easier with atomic adds?
	//			could we just directly write into tiles? and do the rest in pytorch?
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(pointState.scanning_space, pointState.scan_size, pointState.tiles_touched, pointState.point_offsets, PN), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_integrated;
	CHECK_CUDA(cudaMemcpy(&num_integrated, pointState.point_offsets + PN - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

#ifdef DEBUG_INTEGRATE
	printf("We're integrating %d/%d points\n", num_integrated, PN);
#endif

	size_t point_binning_chunk_size = required<PointBinningState>(num_integrated);
	char* point_binning_chunkptr = point_binningBuffer(point_binning_chunk_size);
	PointBinningState point_binningState = PointBinningState::fromChunk(point_binning_chunkptr, num_integrated);
	
	// For each point to be integrated, produce adequate [ tile ] key 
	// and corresponding Point indices to be sorted
	createWithKeys << <(PN + 255) / 256, 256 >> > (
		PN,
		pointState.points2D,
		pointState.depths,
		pointState.point_offsets,
		point_binningState.point_list_keys_unsorted,
		point_binningState.point_list_unsorted,
		tile_grid, width)
	CHECK_CUDA(, debug)

	// Sort complete list of (duplicated) Point indices by keys
	int num_tiles = tile_grid.x * tile_grid.y;
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		point_binningState.list_sorting_space,
		point_binningState.sorting_size,
		point_binningState.point_list_keys_unsorted, point_binningState.point_list_keys,
		point_binningState.point_list_unsorted, point_binningState.point_list,
		num_integrated, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.point_ranges, 0, num_tiles * sizeof(uint2)), debug);
	CHECK_CUDA(cudaMemset(imgState.tile_launch_ranges, 0, num_tiles * sizeof(uint32_t)), debug);
	identifyPixelRanges << <(num_integrated + 255) / 256, 256 >> > (
		num_integrated,
		point_binningState.point_list_keys,
		imgState.point_ranges);

	identifyTileLaunches << <(num_tiles + 255) / 256, 256 >> > (
		num_tiles,
		imgState.point_ranges,
		imgState.tile_launch_ranges);

	uint2* tile_tile_mapping;
#ifdef DEBUG_TILE_LAUNCHES
	cudaDeviceSynchronize();

	// write to the cpu for debugging
	std::vector<uint32_t> data(num_tiles, 0);
	cudaMemcpy(data.data(), imgState.tile_launch_ranges, sizeof(uint32_t) * num_tiles, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_tiles; i++) {
		if (i % tile_grid.x == 0) {printf("\n");}
		printf("%u ", data.at(i));
	}
#endif
	// storage for the output
	uint32_t* tile_launches_sum;
	cudaMalloc((void**)&tile_launches_sum, num_tiles * sizeof(uint32_t));
	
	// Determine temporary device storage requirements
	void     *d_temp_storage = nullptr;
	size_t   temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(
	d_temp_storage, temp_storage_bytes,
	imgState.tile_launch_ranges, tile_launches_sum, num_tiles);

	// Allocate temporary storage
	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	// Run exclusive prefix sum
	cub::DeviceScan::InclusiveSum(
	d_temp_storage, temp_storage_bytes,
	imgState.tile_launch_ranges, tile_launches_sum, num_tiles);
#ifdef DEBUG_TILE_LAUNCHES
	cudaDeviceSynchronize();
	// write to the cpu for debugging
	std::vector<uint32_t> data2(num_tiles, 0);
	cudaMemcpy(data2.data(), tile_launches_sum, sizeof(uint32_t) * num_tiles, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_tiles; i++) {
		if (i % tile_grid.x == 0) {printf("\n");}
		printf("%u ", data2.at(i));
	}

	cudaDeviceSynchronize();
#endif

	// hom many tile/tile mapping combinations
	int num_tile_mappings;
	cudaMemcpy(&num_tile_mappings, tile_launches_sum + num_tiles - 1, sizeof(int), cudaMemcpyDeviceToHost);
#ifdef DEBUG_TILE_LAUNCHES
	printf("\nwe doing %d/%d tiles\n", num_tile_mappings, num_tiles);
#endif
	// memory for the mapping buffer
	cudaMalloc((void**)&tile_tile_mapping, sizeof(uint2) * num_tile_mappings);
	
	// launch kernel to write
	writeTileLaunches << <(num_tiles + 255) / 256, 256 >> > (
		num_tiles,
		imgState.tile_launch_ranges,
		tile_launches_sum,
		tile_tile_mapping);
#ifdef DEBUG_TILE_LAUNCHES
	cudaDeviceSynchronize();
	std::vector<uint32_t> data3(num_tile_mappings * 2, 0);
	cudaMemcpy(data3.data(), tile_tile_mapping, sizeof(uint2) * num_tile_mappings, cudaMemcpyDeviceToHost);

	for (int i = 0; i < num_tile_mappings; i++) {
		if (i % tile_grid.x == 0) {printf("\n");}
		printf("%u ", data3.at(i));
	}
#endif
	timer();
	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		view2gaussian_precomp,
		nullptr,
		viewmatrix, projmatrix, 
		(glm::vec3*)cam_pos, 
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.rects2D,
		splatting_settings,
		debugVisualization,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.cov3D_inv,
		geomState.view2gaussian,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	FORWARD::duplicate(
		P,
		geomState.means2D,
		geomState.conic_opacity,
		radii,
		geomState.rects2D,
		geomState.point_offsets,
		geomState.depths,
		geomState.cov3D_inv,
		splatting_settings,
		projmatrix,
		inv_viewprojmatrix,
		cam_pos,
		width, height,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		tile_grid);
	CHECK_CUDA(, debug)

	// Sort complete list of (duplicated) Gaussian indices by keys
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	CHECK_CUDA(, debug)
	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges);
	CHECK_CUDA(, debug)
	
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	const float* cov3Ds = cov3D_precomp != nullptr ? cov3D_precomp : geomState.cov3D;
	const float* view2gaussian = view2gaussian_precomp != nullptr ? view2gaussian_precomp : geomState.view2gaussian;
	timer();

#ifdef OPT_TILE_LAUNCHES
	tile_grid = dim3(num_tile_mappings,1,1);
#endif
	DebugVisualizationData debug_data;
	CHECK_CUDA(FORWARD::integrate(
		tile_grid, block,
		imgState.ranges,
		imgState.point_ranges,
		tile_tile_mapping,
		binningState.point_list,
		point_binningState.point_list,
		binningState.point_list_keys,
		width, height, PN,
		focal_x, focal_y,
		pointState.points2D,
		feature_ptr,
		view2gaussian,
		cov3Ds,
		geomState.cov3D_inv,
		viewmatrix,
		inv_viewprojmatrix,
		(float3*)means3D,
		(float3*)scales,
		pointState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		(glm::vec3*)cam_pos,
		geomState.means2D,
		out_color,
		out_alpha_integrated,
		out_color_integrated, debug_data, splatting_settings), debug)

	timer();

	std::vector<std::pair<std::string, float>> timings;
	timer.syncAddReport(timings);

	if (timings.size() > 0)
	{
		std::stringstream ss;
		ss << "Timings: \n";
		for (auto const& x : timings)
			ss << " - " << x.first << ": " << x.second << "ms\n";
		std::cout << ss.str() << std::endl;
	}

	return num_integrated;
}

int CudaRasterizer::Rasterizer::computeTransmittance(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> pointBuffer,
	std::function<char* (size_t)> point_binningBuffer,
	const int PN, const int P, int D, int M,
	const int width, int height,
	const SplattingSettings splatting_settings,
	const float* points3D,
	const float* means3D,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* view2gaussian_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* inv_viewprojmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	int* radii,
	float* out_transmittance,
	bool debug)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P, true);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P, true);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid(
		(width + BLOCK_X - 1) / BLOCK_X,
		(height + BLOCK_Y - 1) / BLOCK_Y,
		1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	CHECK_CUDA(FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		nullptr,
		geomState.clamped,
		cov3D_precomp,
		geomState.rgb,
		view2gaussian_precomp,
		nullptr,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.rects2D,
		splatting_settings,
		{},
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.cov3D_inv,
		geomState.view2gaussian,
		nullptr,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	), debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size, geomState.tiles_touched, geomState.point_offsets, P), debug)

	int num_rendered;
	CHECK_CUDA(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	FORWARD::duplicate(
		P,
		geomState.means2D,
		geomState.conic_opacity,
		radii,
		geomState.rects2D,
		geomState.point_offsets,
		geomState.depths,
		geomState.cov3D_inv,
		splatting_settings,
		projmatrix,
		inv_viewprojmatrix,
		cam_pos,
		width, height,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		tile_grid);
	
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit), debug)

	size_t point_chunk_size = required<PointState>(PN);
	char* point_chunkptr = pointBuffer(point_chunk_size);
	PointState pointState = PointState::fromChunk(point_chunkptr, PN);

	CHECK_CUDA(FORWARD::PreprocessPoints(
		PN, D, M,
		points3D,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		pointState.points2D,
		out_transmittance,
		pointState.depths,
		tile_grid,
		pointState.tiles_touched,
		out_transmittance,
		prefiltered
	), debug)

	CHECK_CUDA(cub::DeviceScan::InclusiveSum(pointState.scanning_space, pointState.scan_size, pointState.tiles_touched, pointState.point_offsets, PN), debug)

	int num_integrated;
	CHECK_CUDA(cudaMemcpy(&num_integrated, pointState.point_offsets + PN - 1, sizeof(int), cudaMemcpyDeviceToHost), debug);

	size_t point_binning_chunk_size = required<PointBinningState>(num_integrated);
	char* point_binning_chunkptr = point_binningBuffer(point_binning_chunk_size);
	PointBinningState point_binningState = PointBinningState::fromChunk(point_binning_chunkptr, num_integrated);
	
	createWithKeys << <(PN + 255) / 256, 256 >> > (
		PN,
		pointState.points2D,
		pointState.depths,
		pointState.point_offsets,
		point_binningState.point_list_keys_unsorted,
		point_binningState.point_list_unsorted,
		tile_grid, width);
	
	CHECK_CUDA(cub::DeviceRadixSort::SortPairs(
		point_binningState.list_sorting_space,
		point_binningState.sorting_size,
		point_binningState.point_list_keys_unsorted, point_binningState.point_list_keys,
		point_binningState.point_list_unsorted, point_binningState.point_list,
		num_integrated, 0, 32 + bit), debug)

	CHECK_CUDA(cudaMemset(imgState.point_ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	CHECK_CUDA(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)), debug);
	CHECK_CUDA(cudaMemset(imgState.tile_launch_ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint32_t)), debug);

	identifyPixelRanges << <(num_integrated + 255) / 256, 256 >> > (
		num_integrated,
		point_binningState.point_list_keys,
		imgState.point_ranges);

	identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
		num_rendered,
		binningState.point_list_keys,
		imgState.ranges);

	int num_tiles = tile_grid.x * tile_grid.y;
	identifyTileLaunches << <(num_tiles + 255) / 256, 256 >> > (
		num_tiles,
		imgState.point_ranges,
		imgState.tile_launch_ranges);

	uint32_t* tile_launches_sum;
	cudaMalloc((void**)&tile_launches_sum, num_tiles * sizeof(uint32_t));
	
	void     *d_temp_storage = nullptr;
	size_t   temp_storage_bytes = 0;
	cub::DeviceScan::InclusiveSum(
	d_temp_storage, temp_storage_bytes,
	imgState.tile_launch_ranges, tile_launches_sum, num_tiles);

	cudaMalloc(&d_temp_storage, temp_storage_bytes);

	cub::DeviceScan::InclusiveSum(
	d_temp_storage, temp_storage_bytes,
	imgState.tile_launch_ranges, tile_launches_sum, num_tiles);

	int num_tile_mappings;
	cudaMemcpy(&num_tile_mappings, tile_launches_sum + num_tiles - 1, sizeof(int), cudaMemcpyDeviceToHost);

	uint2* tile_tile_mapping;
	cudaMalloc((void**)&tile_tile_mapping, sizeof(uint2) * num_tile_mappings);
	
	writeTileLaunches << <(num_tiles + 255) / 256, 256 >> > (
		num_tiles,
		imgState.tile_launch_ranges,
		tile_launches_sum,
		tile_tile_mapping);

	const float* view2gaussian = view2gaussian_precomp != nullptr ? view2gaussian_precomp : geomState.view2gaussian;
	const float* cov3Ds = cov3D_precomp != nullptr ? cov3D_precomp : geomState.cov3D;

#ifdef OPT_TILE_LAUNCHES
	tile_grid = dim3(num_tile_mappings,1,1);
#endif

	DebugVisualizationData debug_data;
	CHECK_CUDA(FORWARD::computeTransmittance(
		tile_grid, block,
		imgState.ranges,
		imgState.point_ranges,
		tile_tile_mapping,
		binningState.point_list,
		point_binningState.point_list,
		binningState.point_list_keys,
		width, height, PN,
		focal_x, focal_y,
		pointState.points2D,
		view2gaussian,
		cov3Ds,
		geomState.cov3D_inv,
		viewmatrix,
		inv_viewprojmatrix,
		(float3*)means3D,
		(float3*)scales,
		pointState.depths,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		(glm::vec3*)cam_pos,
		geomState.means2D,
		out_transmittance,
		splatting_settings.meshing_settings.transmittance_threshold,
		debug_data,
		splatting_settings), debug)

	cudaFree(tile_launches_sum);
	cudaFree(d_temp_storage);
	cudaFree(tile_tile_mapping);

	return num_integrated;
}



