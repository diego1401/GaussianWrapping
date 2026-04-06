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

#include "forward.h"
#include "forward_common.h"
#include "auxiliary.h"
#include "stopthepop/stopthepop_common.cuh"
#include "stopthepop/resorted_render.cuh"
#include "stopthepop/hierarchical_render.cuh"

#include <cooperative_groups.h>
namespace cg = cooperative_groups;


// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeysCUDA(
	int P,
	const float2* rects,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	const int* radii,
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

		getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint32_t tile_id = y * grid.x + x;
				gaussian_keys_unsorted[off] = constructSortKey(tile_id, depths[idx]);
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}


// TODO combined with computeCov3D to avoid redundant computation
// Forward method for creating a view to gaussian coordinate system transformation matrix
__device__ void computeView2Gaussian(const glm::vec3 scale, const float mod, const float3& mean, const glm::vec4 rot, const float* viewmatrix, float* view2gaussian)
{
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
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

	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f
	);

	// could be simplied by using pointer
	// viewmatrix is the world to view transformation matrix
	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);

	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;

	// inverse of Gaussian to view transform
	// glm::mat4 V2G_inverse = glm::inverse(G2V);
	// R = G2V[:, :3, :3]
	// t = G2V[:, :3, 3]
	
	// t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
	// V2G = torch.zeros((N, 4, 4), device='cuda')
	// V2G[:, :3, :3] = R.transpose(1, 2)
	// V2G[:, :3, 3] = t2
	// V2G[:, 3, 3] = 1.0
	glm::mat3 R_transpose = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);

	glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	glm::vec3 t2 = -R_transpose * t;

	// view2gaussian[0] = R_transpose[0][0];
	// view2gaussian[1] = R_transpose[0][1];
	// view2gaussian[2] = R_transpose[0][2];
	// view2gaussian[3] = 0.0f;
	// view2gaussian[4] = R_transpose[1][0];
	// view2gaussian[5] = R_transpose[1][1];
	// view2gaussian[6] = R_transpose[1][2];
	// view2gaussian[7] = 0.0f;
	// view2gaussian[8] = R_transpose[2][0];
	// view2gaussian[9] = R_transpose[2][1];
	// view2gaussian[10] = R_transpose[2][2];
	// view2gaussian[11] = 0.0f;
	// view2gaussian[12] = t2.x;
	// view2gaussian[13] = t2.y;
	// view2gaussian[14] = t2.z;
	// view2gaussian[15] = 1.0f;

    // precompute the value here to avoid repeated computations also reduce IO
	// v is the viewdirection and v^T is the transpose of v
	// o = position of the camera in the gaussian coordinate system
	// A = v^T @ R^T @ S^-1 @ S^-1 @ R @ v
	// B = o^T @ S^-1 @ S^-1 @ R @ v
	// C = o^T @ S^-1 @ S^-1 @ o
	// For the given caemra, t is fix and v depends on the pixel
	// therefore we can precompute A, B, C and use them in the forward pass
	// For A, we can precompute R^T @ S^-1 @ S^-1 @ R, which is a symmetric matrix and only store the upper triangle in 6 values
	// For B, we can precompute o^T @ S^-1 @ S^-1 @ R, which is a vector and store it in 3 values
	// and C is fixed, so we only need to store 1 value
	// Therefore, we only need to store 10 values in the view2gaussian matrix
	// S^-1 @ S^-1 is shared in A, B, C
	double3 s = {
		(double) scale.x * mod,
		(double) scale.y * mod,
		(double) scale.z * mod
	};
	double3 S_inv_square = {
		1.0f / ((double)s.x * s.x + 1e-7), 
		1.0f / ((double)s.y * s.y + 1e-7), 
		1.0f / ((double)s.z * s.z + 1e-7)};
	double C = t2.x * t2.x * S_inv_square.x + t2.y * t2.y * S_inv_square.y + t2.z * t2.z * S_inv_square.z;
	glm::mat3 S_inv_square_R = glm::mat3(
		S_inv_square.x * R_transpose[0][0], S_inv_square.y * R_transpose[0][1], S_inv_square.z * R_transpose[0][2],
		S_inv_square.x * R_transpose[1][0], S_inv_square.y * R_transpose[1][1], S_inv_square.z * R_transpose[1][2],
		S_inv_square.x * R_transpose[2][0], S_inv_square.y * R_transpose[2][1], S_inv_square.z * R_transpose[2][2]
	); 

	glm::vec3 B = t2 * S_inv_square_R;

	glm::mat3 Sigma = glm::transpose(R_transpose) * S_inv_square_R;

	// write to view2gaussian
	view2gaussian[0] = Sigma[0][0];
	view2gaussian[1] = Sigma[0][1];
	view2gaussian[2] = Sigma[0][2];
	view2gaussian[3] = Sigma[1][1];
	view2gaussian[4] = Sigma[1][2];
	view2gaussian[5] = Sigma[2][2];
	view2gaussian[6] = B.x;
	view2gaussian[7] = B.y;
	view2gaussian[8] = B.z;
	view2gaussian[9] = C;	
	
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C, bool TILE_BASED_CULLING, bool LOAD_BALANCING, bool ENABLE_DEBUG_VIZ=false>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* view2gaussian_precomp,
	const float* filter_3d,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* rects,
	const SplattingSettings splatting_settings,
	const DebugVisualizationData debug_data,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float4* cov3D_invs,
	float* view2gaussian,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
#define RETURN_OR_INACTIVE() if constexpr(TILE_BASED_CULLING && LOAD_BALANCING) { active = false; } else { return; }

	auto idx = cg::this_grid().thread_rank();
	bool active = true;
	if (idx >= P) {
		RETURN_OR_INACTIVE();
		idx = P - 1;
	}

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	const glm::vec3 mean3D(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	const glm::mat4x3 viewmatrix_mat = loadMatrix4x3(viewmatrix);

	// Perform near culling, quit if outside.
	glm::vec3 p_view;
	if (!in_frustum(idx, mean3D, viewmatrix_mat, prefiltered, p_view))
		RETURN_OR_INACTIVE();


	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// if filter_3d is nullptr, it was precomputed and baked into the representation
	// else, we nedd to compute it here (;
	glm::vec3 scale =  scales[idx];
	float opacity = opacities[idx];

	if (filter_3d != nullptr) {
		float filter_3d_sq = square(filter_3d[idx]);

		glm::vec3 scale_sq = glm::vec3(
			square(scale.x),
			square(scale.y),
			square(scale.z)
		);

		// 3D filter for scaling
		scale = glm::vec3{
			scale_sq.x + filter_3d_sq,
			scale_sq.y + filter_3d_sq,
			scale_sq.z + filter_3d_sq
		};

		float det1 = scale_sq.x * scale_sq.y * scale_sq.z;
		float det2 = scale.x * scale.y * scale.z;

		scale = glm::vec3{
			sqrtf(scale.x),
			sqrtf(scale.y),
			sqrtf(scale.z)
		};

		opacity = opacity * sqrtf(det1 / det2);
	}

	const glm::vec4 rot = rotations[idx];
	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scale, scale_modifier, rot, cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	glm::mat3 cov = computeCov2D(p_view, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix_mat);

	float det, convolution_scaling_factor;
	glm::vec3 cov2D = dilateCov2D(cov, splatting_settings.proper_ewa_scaling, det, convolution_scaling_factor);
	if (det == 0.0f)
		RETURN_OR_INACTIVE();

	// Invert covariance (EWA algorithm)
	float4 co = active ? computeConicOpacity(cov2D, opacity, det, convolution_scaling_factor) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

	if (co.w < ALPHA_THRESHOLD)
		RETURN_OR_INACTIVE();

	// Slightly higher threshold for tile-based culling; Otherwise, imprecisions could lead to more tiles in preprocess than in duplicate
	constexpr float alpha_threshold = TILE_BASED_CULLING ? ALPHA_THRESHOLD_PADDED : ALPHA_THRESHOLD;
	const float opacity_power_threshold = log(co.w / alpha_threshold);

	// Compute extent in screen space (by finding eigenvalues of 2D covariance matrix).
	const float extent = splatting_settings.culling_settings.tight_opacity_bounding ? min(3.33, sqrt(2.0f * opacity_power_threshold)) : 3.33f;

	const float min_lambda = 0.01f;
	const float mid = 0.5f * (cov2D.x + cov2D.z);
	const float lambda = mid + sqrt(max(min_lambda, mid * mid - det));
	const float radius = extent * sqrt(lambda);

	if (radius <= 0.0f)
		RETURN_OR_INACTIVE();

	// Transform point by projecting
	const glm::mat4 viewproj_mat = loadMatrix4x4(projmatrix);
	const glm::vec3 p_proj = world2ndc(mean3D, viewproj_mat);
	const float2 mean2D = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };

	uint2 rect_min, rect_max;
	float2 bb_center;
	float2 rect_dims;
	
	// Use extent to compute a bounding rectangle of screen-space tiles that this Gaussian overlaps with.
	// Quit if rectangle covers 0 tiles
	const float extent_x = min(splatting_settings.culling_settings.rect_bounding ? (extent * sqrt(cov2D.x)) : radius, radius);
	const float extent_y = min(splatting_settings.culling_settings.rect_bounding ? (extent * sqrt(cov2D.z)) : radius, radius);
	rect_dims = make_float2(extent_x, extent_y);
	bb_center = mean2D;

	getRect(bb_center, rect_dims, rect_min, rect_max, grid);	
	const int tile_count_rect = (rect_max.x - rect_min.x) * (rect_max.y - rect_min.y);
	if (tile_count_rect == 0)
		RETURN_OR_INACTIVE();

	const uint32_t WARP_MASK = 0xFFFFFFFFU;
	if constexpr(TILE_BASED_CULLING && LOAD_BALANCING)
		if (__ballot_sync(WARP_MASK, active) == 0) // early stop if whole warp culled
			return;
	
	int tile_count;
	if constexpr (TILE_BASED_CULLING)
		tile_count = computeTilebasedCullingTileCount<LOAD_BALANCING>(active, co, mean2D, opacity_power_threshold, rect_min, rect_max);
	else
		tile_count = tile_count_rect;


	if (tile_count == 0 || !active) // Cooperative threads no longer needed (after load balancing)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result;
		result = computeColorFromSH(idx, D, M, mean3D, *cam_pos, shs, clamped);

		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	if (cov3D_invs != nullptr)
	{
		const glm::vec3 mean3D(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
		glm::mat3 inv = computeInvCov3D(scale, rotations[idx], scale_modifier);

		// symmetric matrix, store six elements 
		// pack with Cov3dinv*(campos - mean) into 3 float4 for efficiency
		// we do have 3 elements leftover
		glm::vec3 upper = -inv * (*cam_pos - mean3D);
		cov3D_invs[3 * idx] = { inv[0][0], inv[0][1], inv[0][2], 0 };
		cov3D_invs[3 * idx + 1] = { inv[1][1], inv[1][2], inv[2][2], 0 };
		cov3D_invs[3 * idx + 2] = { upper.x, upper.y, upper.z, 0 };
	}

	// view to gaussian coordinate system
	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	const float* view2gaussian_;
	if (view2gaussian_precomp == nullptr)
	{
		// printf("view2gaussian_precomp is nullptr\n");
		computeView2Gaussian(scale, scale_modifier, p_orig, rot, viewmatrix, view2gaussian + idx * 10);
		
	} else {
		view2gaussian_ = view2gaussian_precomp + idx * 16;
	}

	if (splatting_settings.sort_settings.sort_order == GlobalSortOrder::MIN_VIEWSPACE_Z) {
		glm::mat3 T = glm::transpose(glm::mat3(viewmatrix_mat));

		float* c3d = cov3Ds + idx * 6;
		glm::mat3 Vrk = glm::mat3(
			c3d[0], c3d[1], c3d[2],
			c3d[1], c3d[3], c3d[4],
			c3d[2], c3d[4], c3d[5]);

		glm::mat3 zigma = glm::transpose(T) * glm::transpose(Vrk) * T;

#ifdef DEBUG_MIN_Z_BOUNDING
	printf("Gaussian %d: depth %f vs %f\n", idx,p_view.z, fmaxf(p_view.z - (sqrtf(zigma[2][2]) * 3.33f), 0.2f));
#endif
		// minimum z boundary
		depths[idx] = fmaxf(p_view.z - (sqrtf(zigma[2][2]) * extent), 0.2f);
	}
	else {
		depths[idx] = splatting_settings.sort_settings.sort_order == GlobalSortOrder::VIEWSPACE_Z ? p_view.z : glm::length(*cam_pos - mean3D);	
	}

	radii[idx] = (int) ceil(radius);
	rects[idx] = rect_dims;
	points_xy_image[idx] = bb_center;
	conic_opacity[idx] = co; // Inverse 2D covariance and opacity neatly pack into one float4
	tiles_touched[idx] = tile_count;
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool ENABLE_DEBUG_VIZ>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float far_plane,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* view2gaussian,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugVisualizationType,
	const glm::vec3* cam_pos,
	const glm::vec3* means3D,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t max_contributor = -1;
	float C[CHANNELS*2+2] = { 0 };

	float T_opa = 1.f;

	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};

	[[maybe_unused]] float depth_accum = 0.f;
	[[maybe_unused]] float currentDepth = -FLT_MAX;
	[[maybe_unused]] float sortingErrorCount = 0.f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;

			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			for (int ii = 0; ii < 10; ii++)
				collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			float4 con_o = collected_conic_opacity[j];
			float* view2gaussian_j = collected_view2gaussian + j * 10;
			
			float3 ray_point = { (pixf.x - W/2.) / focal_x, (pixf.y - H/2.) / focal_y, 1.0 };

			const float normal[3] = { 
				view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2], 
				view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
				view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
			};

			// use AA, BB, CC so that the name is unique
			double AA = ray_point.x * normal[0] + ray_point.y * normal[1] + normal[2];
			double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			float CC = view2gaussian_j[9];
			
			// t is the depth of the gaussian
			float t = -BB/(2*AA);
			// depth must be positive otherwise it is not valid and we skip it
			if (t <= NEAR_PLANE)
				continue;

			// the scale of the gaussian is 1.f / sqrt(AA)
			double min_value = -(BB/AA) * (BB/4.) + CC;

			float power = -0.5f * min_value;
			if (power > 0.0f){
				power = 0.0f;
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// NDC mapping is taken from 2DGS paper, please check here https://arxiv.org/pdf/2403.17888.pdf
			const float max_t = t;
			const float mapped_max_t = (far_plane * max_t - far_plane * NEAR_PLANE) / ((far_plane - NEAR_PLANE) * max_t);
			
			// normalize normal
			float length = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2] + 1e-7);
			const float normal_normalized[3] = { -normal[0] / length, -normal[1] / length, -normal[2] / length };

			// distortion loss is taken from 2DGS paper, please check https://arxiv.org/pdf/2403.17888.pdf
			float A = 1-T;
			float error = mapped_max_t * mapped_max_t * A + dist2 - 2 * mapped_max_t * dist1;
			distortion += error * alpha * T;
			
			dist1 += mapped_max_t * alpha * T;
			dist2 += mapped_max_t * mapped_max_t * alpha * T;

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++) {
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
				C[CHANNELS + ch] += normal_normalized[ch] * alpha * T;
			}

			// depth and alpha
			if (T > 0.5f){
				C[CHANNELS * 2] = t;
				max_contributor = contributor;
			}
			else {
				// eval at the depth
				float depth = C[CHANNELS * 2];
				float alpha_point = alpha;
				if (t > depth) {
					float min_value = (AA * depth * depth + BB * depth + CC);
					float p = -0.5f * min_value;
					if (p > 0.0f){
						p = 0.0f;
					}
					alpha_point = min(0.99f, con_o.w * exp(p));
				}

				T_opa *= (1 - alpha_point);
			}
			C[CHANNELS * 2 + 1] += alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		if constexpr (!ENABLE_DEBUG_VIZ)
		{
			const float distortion_before_normalized = distortion;
			// normalize
			distortion /= (1 - T) * (1 - T) + 1e-7;

			final_T[pix_id] = T;
			final_T[pix_id + H * W] = dist1;
			final_T[pix_id + 2 * H * W] = dist2;
			final_T[pix_id + 3 * H * W] = distortion;

			n_contrib[pix_id] = last_contributor;
			n_contrib[pix_id + H * W] = max_contributor;

			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

			// normal
			for (int ch = 0; ch < CHANNELS; ch++){
				out_color[(CHANNELS + ch) * H * W + pix_id] = C[CHANNELS+ch];
			}

			// depth and alpha
			out_color[DEPTH_OFFSET * H * W + pix_id] = C[CHANNELS * 2];
			out_color[ALPHA_OFFSET * H * W + pix_id] = T_opa;
			out_color[DISTORTION_OFFSET * H * W + pix_id] = distortion;
		}
		else {
			n_contrib[pix_id] = last_contributor;
			n_contrib[pix_id + H * W] = max_contributor;

			// depth and alpha
			out_color[DEPTH_OFFSET * H * W + pix_id] = C[CHANNELS * 2];
			out_color[ALPHA_OFFSET * H * W + pix_id] = T_opa;

			outputDebugVis(debugVisualizationType, out_color, pix_id, contributor, T,  C[CHANNELS * 2],  
				1 - T_opa, distortion, 0.f, 0.f,toDo, max_contributor, H, W);
		}
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool ENABLE_DEBUG_VIZ>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_opacity(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float far_plane,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* view2gaussian,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	DebugVisualization debugVisualizationType,
	const glm::vec3* cam_pos,
	const glm::vec3* means3D,
	float* __restrict__ out_color)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5f, (float)pix.y + 0.5f };

	float depth = out_color[DEPTH_OFFSET * H * W + pix_id];

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	uint32_t max_contributor = n_contrib[pix_id + H * W];
	float C[CHANNELS*2+2] = { 0 };

	float T_opa = 1.f;

	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};

	[[maybe_unused]] float depth_accum = 0.f;
	[[maybe_unused]] float currentDepth = -FLT_MAX;
	[[maybe_unused]] float sortingErrorCount = 0.f;

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;

			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			for (int ii = 0; ii < 10; ii++)
				collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			if (contributor > max_contributor) {
				done = true;
				continue;
			}

			float4 con_o = collected_conic_opacity[j];
			float* view2gaussian_j = collected_view2gaussian + j * 10;
			
			float3 ray_point = { (pixf.x - W/2.) / focal_x, (pixf.y - H/2.) / focal_y, 1.0 };

			const float normal[3] = { 
				view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2], 
				view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
				view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
			};

			// use AA, BB, CC so that the name is unique
			double AA = ray_point.x * normal[0] + ray_point.y * normal[1] + normal[2];
			double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
			float CC = view2gaussian_j[9];
			
			// t is the depth of the gaussian
			float t = -BB/(2*AA);
			// depth must be positive otherwise it is not valid and we skip it
			if (t <= NEAR_PLANE)
				continue;

			// the scale of the gaussian is 1.f / sqrt(AA)
			double min_value = -(BB/AA) * (BB/4.) + CC;

			float power = -0.5f * min_value;
			if (power > 0.0f){
				power = 0.0f;
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}



			float alpha_point = alpha;
			if (t > depth) {
				float min_value = (AA * depth * depth + BB * depth + CC);
				float p = -0.5f * min_value;
				if (p > 0.0f){
					p = 0.0f;
				}
				alpha_point = min(0.99f, con_o.w * exp(p));
			}

			T_opa *= (1 - alpha_point);

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// multiply with the other ones
		T_opa *= out_color[ALPHA_OFFSET * H * W + pix_id];
		float opacity_accum = 1 - T_opa;

		if constexpr (!ENABLE_DEBUG_VIZ)
		{
			const float distortion_before_normalized = distortion;
			// normalize
			distortion /= (1 - T) * (1 - T) + 1e-7;

			final_T[pix_id] = T;
			final_T[pix_id + H * W] = dist1;
			final_T[pix_id + 2 * H * W] = dist2;
			final_T[pix_id + 3 * H * W] = distortion;

			n_contrib[pix_id] = last_contributor;
			n_contrib[pix_id + H * W] = max_contributor;

			for (int ch = 0; ch < CHANNELS; ch++)
				out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

			// normal
			for (int ch = 0; ch < CHANNELS; ch++){
				out_color[(CHANNELS + ch) * H * W + pix_id] = C[CHANNELS+ch];
			}

			// depth and alpha
			out_color[DEPTH_OFFSET * H * W + pix_id] = C[CHANNELS * 2];
			out_color[ALPHA_OFFSET * H * W + pix_id] = opacity_accum;
			out_color[DISTORTION_OFFSET * H * W + pix_id] = distortion;


		}
		else {
			outputDebugVis(debugVisualizationType, out_color, pix_id, contributor, T,  C[CHANNELS * 2],  
				opacity_accum, distortion, 0.f, 0.f,toDo, max_contributor, H, W);
		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const SplattingSettings splatting_settings,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* view2gaussian,
	const float* means3D,
	const float4* cov3D_inv,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	const float* colors, // = feature_ptr
	const float* depths,
	const float4* conic_opacity,
	float* final_T, //= accum_alpha
	uint32_t* n_contrib,
	const float* bg_color,
	DebugVisualizationData& debugVisualization,
	float* out_color)
{

	if (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL)
	{
		#define CALL_VANILLA(ENABLE_DEBUG_VIZ) renderCUDA<NUM_CHANNELS, ENABLE_DEBUG_VIZ> <<<grid, block>>> ( \
			ranges, point_list, W, H, focal_x, focal_y, splatting_settings.far_plane, means2D, colors, view2gaussian,conic_opacity, final_T, \
			n_contrib, bg_color, debugVisualization.type, cam_pos, (glm::vec3*)means3D, out_color);

		if (debugVisualization.type == DebugVisualization::Disabled) {
			CALL_VANILLA(false);
		} else {
			CALL_VANILLA(true);
		}

		#undef CALL_VANILLA
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::PER_PIXEL_KBUFFER)
	{
		#define CALL_KBUFFER_DEBUG(WINDOW, ENABLE_DEBUG_VIZ) renderkBufferCUDA<NUM_CHANNELS, WINDOW, ENABLE_DEBUG_VIZ> <<<grid, block>>> (ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3*)cam_pos, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, out_color)
		#define CALL_KBUFFER(WINDOW) if (debugVisualization.type == DebugVisualization::Disabled) CALL_KBUFFER_DEBUG(WINDOW, false); else CALL_KBUFFER_DEBUG(WINDOW, true)
	

#ifdef STOPTHEPOP_FASTBUILD
		CALL_KBUFFER(16);
#else // STOPTHEPOP_FASTBUILD
		int window_size = splatting_settings.sort_settings.queue_sizes.per_pixel;
		if (window_size <= 1) 
			CALL_KBUFFER(1); 
		else if (window_size <= 2) 
			CALL_KBUFFER(2); 
		else if (window_size <= 4) 
			CALL_KBUFFER(4); 
		else if (window_size <= 8) 
			CALL_KBUFFER(8); 
		else if (window_size <= 12) 
			CALL_KBUFFER(12); 
		else if (window_size <= 16) 
			CALL_KBUFFER(16); 
		else if (window_size <= 20) 
			CALL_KBUFFER(20); 
		else 
			CALL_KBUFFER(24);
#endif // STOPTHEPOP_FASTBUILD
		
		#undef CALL_KBUFFER
		#undef CALL_KBUFFER_DEBUG
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::PER_PIXEL_FULL)
	{
		#define CALL_FULLSORT(ENABLE_DEBUG_VIZ) renderSortedFullCUDA<NUM_CHANNELS, ENABLE_DEBUG_VIZ> <<<grid, block>>> (ranges, point_list, W, H, means2D, cov3D_inv, projmatrix_inv, (float3*) cam_pos, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, out_color)
		
		if (debugVisualization.type == DebugVisualization::Disabled) {
			CALL_FULLSORT(false);
		} else {
			CALL_FULLSORT(true);
		}

		#undef CALL_FULLSORT
	}
	else if (splatting_settings.sort_settings.sort_mode == SortMode::HIERARCHICAL)
	{
#define CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, DEBUG, EXACT_DEPTH) sortGaussiansRayHierarchicalCUDA_forward<NUM_CHANNELS, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE, HIER_CULLING, EXACT_DEPTH, DEBUG><<<grid, {16, 4, 4}>>>( \
	ranges, point_list, W, H, focal_x, focal_y, splatting_settings.far_plane, splatting_settings.include_alpha,view2gaussian, means2D, cov3D_inv, projmatrix_inv, (float3 *)cam_pos, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, out_color)

#define CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, DEBUG) if (splatting_settings.exact_depth) { CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, DEBUG, true); } else { CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, DEBUG, false); }

#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE) if (debugVisualization.type == DebugVisualization::Disabled || debugVisualization.type == DebugVisualization::Opacity) { CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, false); } else { CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, true); }

#ifdef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#else // STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		case 8: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 8); break; } \
		case 16: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 16); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		case 12: { CALL_HIER_HEAD(HIER_CULLING, 12); break; } \
		case 20: { CALL_HIER_HEAD(HIER_CULLING, 20); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#endif // STOPTHEPOP_FASTBUILD

	if (splatting_settings.culling_settings.hierarchical_4x4_culling) {
		CALL_HIER_MID(true);
	} else {
		CALL_HIER_MID(false);
	}

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER
#undef CALL_HIER_EXACT_DEPTH
#undef CALL_HIER_DEBUG
	}
}

void FORWARD::render_opacity(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const SplattingSettings splatting_settings,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* means2D,
	const float* view2gaussian,
	const float* means3D,
	const float4* cov3D_inv,
	const float* projmatrix_inv,
	const glm::vec3* cam_pos,
	const float* colors, // = feature_ptr
	const float* depths,
	const float4* conic_opacity,
	float* final_T, //= accum_alpha
	uint32_t* n_contrib,
	const float* bg_color,
	DebugVisualizationData& debugVisualization,
	float* out_color)
{
	if (splatting_settings.sort_settings.sort_mode == SortMode::GLOBAL) {
		#define CALL_VANILLA(ENABLE_DEBUG_VIZ) renderCUDA_opacity<NUM_CHANNELS, ENABLE_DEBUG_VIZ> <<<grid, block>>> ( \
			ranges, point_list, W, H, focal_x, focal_y, splatting_settings.far_plane, means2D, colors, view2gaussian,conic_opacity, final_T, \
			n_contrib, bg_color, debugVisualization.type, cam_pos, (glm::vec3*)means3D, out_color);

		if (debugVisualization.type == DebugVisualization::Disabled) {
			CALL_VANILLA(false);
		} else {
			CALL_VANILLA(true);
		}
	}
	else {

#define CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EXACT_DEPTH, DEBUG) sortGaussiansRayHierarchicalCUDA_opacity<NUM_CHANNELS, HEAD_QUEUE_SIZE, MID_QUEUE_SIZE, HIER_CULLING, EXACT_DEPTH, DEBUG><<<grid, {16, 4, 4}>>>( \
	ranges, point_list, W, H, focal_x, focal_y, splatting_settings.far_plane, view2gaussian, means2D, cov3D_inv, projmatrix_inv, (float3 *)cam_pos, colors, conic_opacity, final_T, n_contrib, bg_color, debugVisualization.type, out_color)

#define CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EXACT_DEPTH) if (debugVisualization.type == DebugVisualization::Opacity) { CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EXACT_DEPTH, true); } else { CALL_HIER_DEBUG(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, EXACT_DEPTH, false); }

#define CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE) if (splatting_settings.exact_depth) { CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, true); } else { CALL_HIER_EXACT_DEPTH(HIER_CULLING, MID_QUEUE_SIZE, HEAD_QUEUE_SIZE, false); }

#ifdef STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#else // STOPTHEPOP_FASTBUILD
#define CALL_HIER_HEAD(HIER_CULLING, MID_QUEUE_SIZE) \
	switch (splatting_settings.sort_settings.queue_sizes.per_pixel) \
	{ \
		case 4: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 4); break; } \
		case 8: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 8); break; } \
		case 16: { CALL_HIER(HIER_CULLING, MID_QUEUE_SIZE, 16); break; } \
		default: { throw std::runtime_error("Not supported head queue size"); } \
	}

#define CALL_HIER_MID(HIER_CULLING) \
	switch (splatting_settings.sort_settings.queue_sizes.tile_2x2) \
	{ \
		case 8: { CALL_HIER_HEAD(HIER_CULLING, 8); break; } \
		case 12: { CALL_HIER_HEAD(HIER_CULLING, 12); break; } \
		case 20: { CALL_HIER_HEAD(HIER_CULLING, 20); break; } \
		default: { throw std::runtime_error("Not supported mid queue size"); } \
	}
#endif // STOPTHEPOP_FASTBUILD

	if (splatting_settings.culling_settings.hierarchical_4x4_culling) {
		CALL_HIER_MID(true);
	} else {
		CALL_HIER_MID(false);
	}

#undef CALL_HIER_MID
#undef CALL_HIER_HEAD
#undef CALL_HIER
#undef CALL_HIER_EXACT_DEPTH
#undef CALL_HIER_DEBUG
}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* view2gaussian_precomp,
	const float* filter_3d,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* rects,
	const SplattingSettings splatting_settings,
	const DebugVisualizationData debug_data,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float4* cov3D_invs,
	float* view2gaussians,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
#define PREPROCESS_CALL_DEBUG(TBC, LB, ENABLE_DEBUG_VIZ) \
	preprocessCUDA<NUM_CHANNELS, TBC, LB, ENABLE_DEBUG_VIZ> << <(P + 255) / 256, 256 >> > ( \
		P, D, M, \
		means3D, \
		scales, \
		scale_modifier, \
		rotations, \
		opacities, \
		shs, \
		clamped, \
		cov3D_precomp, \
		colors_precomp, \
		view2gaussian_precomp, \
		filter_3d, \
		viewmatrix,  \
		projmatrix, \
		cam_pos, \
		W, H, \
		tan_fovx, tan_fovy, \
		focal_x, focal_y, \
		radii, \
		rects, \
		splatting_settings, \
		debug_data, \
		means2D, \
		depths, \
		cov3Ds, \
		cov3D_invs, \
		view2gaussians, \
		rgb, \
		conic_opacity, \
		grid, \
		tiles_touched, \
		prefiltered \
		);
	
#define PREPROCESS_CALL(TBC, LB) if (debug_data.type != DebugVisualization::Disabled) { PREPROCESS_CALL_DEBUG(TBC, LB, true); } else { PREPROCESS_CALL_DEBUG(TBC, LB, false); }

	if (splatting_settings.culling_settings.tile_based_culling)
	{
		if (splatting_settings.load_balancing) {
			PREPROCESS_CALL(true, true);
		} else {
			PREPROCESS_CALL(true, false);
		}
	}
	else
	{
		if (splatting_settings.load_balancing) {
			PREPROCESS_CALL(false, true);
		} else {
			PREPROCESS_CALL(false, false);
		}
	}

#undef PREPROCESS_CALL
#undef PREPROCESS_CALL_DEBUG
}

void FORWARD::duplicate(int P,
						const float2 *means2D,
						const float4 *conic_opacity,
						const int *radii,
						const float2 *rects2D,
						const uint32_t *offsets,
						const float *depths,
						const float4 *cov3D_invs,
						const SplattingSettings splatting_settings,
						const float *projmatrix,
						const float *inv_viewprojmatrix,
						const float *cam_pos,
						const int W, int H,
						uint64_t *gaussian_keys_unsorted,
						uint32_t *gaussian_values_unsorted,
						dim3 grid)
{
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	#define CALL_DUPLICATE_EXTENDED(TBC, LB, SORT_ORDER) \
		duplicateWithKeys_extended<TBC, LB, SORT_ORDER> << <(P + 255) / 256, 256 >> > ( \
				P, \
				means2D, \
				depths, \
				cov3D_invs, \
				conic_opacity, \
				projmatrix, \
				inv_viewprojmatrix, \
				(glm::vec3*) cam_pos, \
				W, H, \
				offsets, \
				gaussian_keys_unsorted, \
				gaussian_values_unsorted, \
				radii, \
				rects2D, \
				grid)

	#define CALL_DUPLICATE_SORT_ORDER(SORT_ORDER) \
		if (splatting_settings.culling_settings.tile_based_culling) \
		{ \
			if (splatting_settings.load_balancing) { \
				CALL_DUPLICATE_EXTENDED(true, true, SORT_ORDER); \
			} else { \
				CALL_DUPLICATE_EXTENDED(true, false, SORT_ORDER); \
			} \
		} else { \
			if (splatting_settings.load_balancing) { \
				CALL_DUPLICATE_EXTENDED(false, true, SORT_ORDER); \
			} else { \
				CALL_DUPLICATE_EXTENDED(false, false, SORT_ORDER); \
			} \
		}

		switch (splatting_settings.sort_settings.sort_order)
		{
			case GlobalSortOrder::VIEWSPACE_Z:
			case GlobalSortOrder::MIN_VIEWSPACE_Z:
			case GlobalSortOrder::DISTANCE:
			{
				if (!splatting_settings.load_balancing && !splatting_settings.culling_settings.tile_based_culling)
				{
					duplicateWithKeysCUDA<<<(P + 255) / 256, 256>>>(
						P,
						rects2D,
						means2D,
						depths,
						offsets,
						gaussian_keys_unsorted,
						gaussian_values_unsorted,
						radii,
						grid);
				}
				else
				{
					// viewspace-z and distance treated equally
					CALL_DUPLICATE_SORT_ORDER(GlobalSortOrder::VIEWSPACE_Z);
				}
				break;
			}
			case GlobalSortOrder::PER_TILE_DEPTH_CENTER:
			{
				CALL_DUPLICATE_SORT_ORDER(GlobalSortOrder::PER_TILE_DEPTH_CENTER);
				break;
			}
			case GlobalSortOrder::PER_TILE_DEPTH_MAXPOS:
			{
				CALL_DUPLICATE_SORT_ORDER(GlobalSortOrder::PER_TILE_DEPTH_MAXPOS);
				break;
			}
		}
	#undef CALL_DUPLICATE_EXTENDED
	#undef CALL_DUPLICATE_SORT_ORDER
}

template<uint32_t CHANNELS, bool TURBO=false>
__global__ void render_debug_CUDA(int P, const float* __restrict__ min_max_contrib, float* out_color, bool debug_normalize, float debug_norm_min, float debug_norm_max)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float min = min_max_contrib[0];
	float max = min_max_contrib[1];

	if (debug_normalize)
	{
		min = debug_norm_min;
		max = debug_norm_max;
	}

	float alpha = 0.f;
	glm::vec3 output;
	if constexpr (TURBO) 
	{
		float T = (out_color + P)[idx];
		float alpha = (fminf(fmaxf(out_color[idx] + T * max, min), max) - min) / static_cast<float>(max - min);
		output = colormapTurbo(alpha);
	}
	else
	{
		float alpha = (fminf(fmaxf(out_color[idx], min), max) - min) / static_cast<float>(max - min);
		output = colormapMagma(alpha);
	}

	for (int ch = 0; ch < CHANNELS; ch++)
	{
		out_color[ch * P + idx] = output[ch];
	}
}

void FORWARD::render_debug(DebugVisualizationData& debugVisualization, int P, float* out_color, float* min_max_contrib)
{
	if (!debugVisualization.colormap) {
		return;
	}
	if (debugVisualization.type == DebugVisualization::Depth) {
		render_debug_CUDA<NUM_CHANNELS, true><<<(P + 255) / 256, 256>>>(
			P, min_max_contrib, out_color, debugVisualization.debug_normalize, debugVisualization.min, debugVisualization.max);
	}
	else {
		render_debug_CUDA<NUM_CHANNELS, false><<<(P + 255) / 256, 256>>>(
			P, min_max_contrib, out_color, debugVisualization.debug_normalize, debugVisualization.min, debugVisualization.max);
	}
}


// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool ALPHA_EARLY_STOP=true, bool MIN_Z_BOUNDING=true>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
computeTransmittanceCUDA(
	const uint2* __restrict__ gaussian_ranges,
	const uint2* __restrict__ point_ranges,
	const uint2* __restrict__ tile_tile_mapping,
	const uint32_t* __restrict__ gaussian_list,
	const uint32_t* __restrict__ point_list,
	const uint64_t* __restrict__ gaussian_depths,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points2D,
	const float* __restrict__ view2gaussian,
	const float* __restrict__ cov3Ds,
	const float* viewmatrix,
	const float3* __restrict__ means3D,
	const float3* __restrict__ scales,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	float* __restrict__ out_transmittance,
	float transmittance_threshold)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	// indicates which point we're currently treating
	const uint32_t thread_idx = block.thread_rank();

	// Tile index computation is different
	// 	- first, lookup in tile_tile_mapping for the actual tile_id (TTM.x)
	// 	- then, lookup for the start_range (TTM.y)
#ifdef OPT_TILE_LAUNCHES
	const uint32_t BLOCK_IDX = block.group_index().y * horizontal_blocks + block.group_index().x;
	// lookup for tile id
	const uint32_t TILE_IDX = tile_tile_mapping[BLOCK_IDX].x;

	uint2 p_range = point_ranges[TILE_IDX];
	p_range.x += tile_tile_mapping[BLOCK_IDX].y;
	// we do not do more than 256 points, no way in hell
	p_range.y = min(p_range.x + BLOCK_SIZE, p_range.y);
#else
	const uint32_t TILE_IDX = block.group_index().y * horizontal_blocks + block.group_index().x;

	uint2 p_range = point_ranges[TILE_IDX];
#endif
	const int p_rounds = ((p_range.y - p_range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int p_toDo = p_range.y - p_range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	// todo: pack depth and opacity with view2gaussian maybe?
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE]; // only need opacity
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];
	__shared__ float collected_gaussiandepth[BLOCK_SIZE];

	const uint2 range = gaussian_ranges[TILE_IDX];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// first loop: how often to iterate over Points
	for (int p_round = 0; p_round < p_rounds; p_round++, p_toDo -= BLOCK_SIZE)
	{
		float T = 1.0f;

		// get point info
		int p_progress = p_round * BLOCK_SIZE + thread_idx;

		float2 current_point;
		float current_point_depth;
		uint32_t point_id;
		float3 ray_point;

		if (p_range.x + p_progress < p_range.y)
		{
			point_id = point_list[p_range.x + p_progress];

			current_point = points2D[point_id];
			current_point_depth = depths[point_id];
			ray_point = { 
				(current_point.x - W/2.f) / focal_x, 
				(current_point.y - H/2.f) / focal_y, 
				1.0f 
			};
		}
		block.sync();

		int toDo = range.y - range.x;

		bool active = thread_idx < p_toDo;
		bool done = !active;

		// Iterate over batches until all done or range is complete
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// End if entire block votes that it is done rasterizing
			int num_done = __syncthreads_count(done);
			if (num_done == BLOCK_SIZE)
				break;

			// Collectively fetch per-Gaussian data from global to shared
			int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				int coll_id = gaussian_list[range.x + progress];
				collected_id[block.thread_rank()] = coll_id;
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				for (int ii = 0; ii < 10; ii++)
					collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
				if constexpr (MIN_Z_BOUNDING)
					collected_gaussiandepth[block.thread_rank()] = __uint_as_float(gaussian_depths[range.x + progress]);
			}
			block.sync();

			// Iterate over current batch
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				float4 con_o = collected_conic_opacity[j];
				float* view2gaussian_j = collected_view2gaussian + j * 10;

				if constexpr (MIN_Z_BOUNDING) {
					float gaussian_depth = collected_gaussiandepth[j];
					if (gaussian_depth > current_point_depth) {
						done = true;
						continue;
					}
				}

				const float normal[3] = { 
					view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2], 
					view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
					view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
				};

				// use AA, BB, CC so that the name is unique
				double AA = ray_point.x * normal[0] + ray_point.y * normal[1] + normal[2];
				double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
				float CC = view2gaussian_j[9];
				
				// t is the depth of the gaussian
				float t = -BB/(2*AA);
				// depth must be positive otherwise it is not valid and we skip it
				if (t > current_point_depth) {
					t = current_point_depth;
				}
				
				double min_value = (AA * t * t + BB * t + CC);

				float power = -0.5f * min_value;
				if (power > 0.0f){
					power = 0.0f;
				}

				float alpha_point = min(0.99f, con_o.w * exp(power));

				if (alpha_point < 1.0f / 255.0f) {
					continue;
				}

				T *= (1 - alpha_point);

				if constexpr (ALPHA_EARLY_STOP) {
					if (T < transmittance_threshold) {
						done = true;
					}
				}

			}
		}
		if (active) {
			out_transmittance[point_id] = T;
		}
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS, bool ALPHA_EARLY_STOP=true, bool MIN_Z_BOUNDING=true, bool RETURN_COLOR=true>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
integrateCUDA(
	const uint2* __restrict__ gaussian_ranges,
	const uint2* __restrict__ point_ranges,
	const uint2* __restrict__ tile_tile_mapping,
	const uint32_t* __restrict__ gaussian_list,
	const uint32_t* __restrict__ point_list,
	const uint64_t* __restrict__ gaussian_depths,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points2D,
	const float* __restrict__ features,
	const float* __restrict__ view2gaussian,
	const float* __restrict__ cov3Ds,
	const float* viewmatrix,
	const float3* __restrict__ means3D,
	const float3* __restrict__ scales,
	const float* __restrict__ depths,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_alpha_integrated,
	float* __restrict__ out_color_integrated)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;

	// indicates which point we're currently treating
	const uint32_t thread_idx = block.thread_rank();

	// Tile index computation is different
	// 	- first, lookup in tile_tile_mapping for the actual tile_id (TTM.x)
	// 	- then, lookup for the start_range (TTM.y)
#ifdef OPT_TILE_LAUNCHES
	const uint32_t BLOCK_IDX = block.group_index().y * horizontal_blocks + block.group_index().x;
	// lookup for tile id
	const uint32_t TILE_IDX = tile_tile_mapping[BLOCK_IDX].x;

	uint2 p_range = point_ranges[TILE_IDX];
	p_range.x += tile_tile_mapping[BLOCK_IDX].y;
	// we do not do more than 256 points, no way in hell
	p_range.y = min(p_range.x + BLOCK_SIZE, p_range.y);
	#ifdef DEBUG_TILE_LAUNCHES
		if (block.thread_rank() == 0) {
			printf("BLOCK/TILE %d/%d: range [%d,%d]\n", BLOCK_IDX, TILE_IDX, p_range.x, p_range.y);
		}
	#endif
#else
	const uint32_t TILE_IDX = block.group_index().y * horizontal_blocks + block.group_index().x;

	uint2 p_range = point_ranges[TILE_IDX];
#endif
	const int p_rounds = ((p_range.y - p_range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int p_toDo = p_range.y - p_range.x;

	assert(p_rounds <= 1);

#ifdef DEBUG_TILE_LAUNCHES
	assert(p_toDo <= 256);
#endif

#ifdef DEBUG_INTEGRATE
	if (thread_idx == 0) {
		printf("TILE %d: points rounds todo %d for %d points\n", TILE_IDX, p_rounds, p_toDo);
	}
#endif

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	// todo: pack depth and opacity with view2gaussian maybe?
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE]; // only need opacity
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 10];
	__shared__ float collected_gaussiandepth[BLOCK_SIZE];
	__shared__ float collected_color[BLOCK_SIZE * CHANNELS];

	const uint2 range = gaussian_ranges[TILE_IDX];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	// first loop: how often to iterate over Points
	for (int p_round = 0; p_round < p_rounds; p_round++, p_toDo -= BLOCK_SIZE)
	{
		float T = 1.0f;
		float alpha = 0.0f;
		float C[CHANNELS*2+2] = { 0.f };

		// get point info
		int p_progress = p_round * BLOCK_SIZE + thread_idx;

		float2 current_point;
		float current_point_depth;
		uint32_t point_id;
		float3 ray_point;

		if (p_range.x + p_progress < p_range.y)
		{
			point_id = point_list[p_range.x + p_progress];

			current_point = points2D[point_id];
			current_point_depth = depths[point_id];
			ray_point = { 
				(current_point.x - W/2.f) / focal_x, 
				(current_point.y - H/2.f) / focal_y, 
				1.0f 
			};
		}
		block.sync();

#ifdef DEBUG_INTEGRATE
		if (point_id == POINT_TO_DEBUG) {
			printf("Point %f/%f -> %f\n", current_point.x, current_point.y, current_point_depth);
			printf("Ray %f/%f/%f\n", (current_point.x - W/2.) / focal_x, (current_point.y - H/2.) / focal_y, 1.0f );
		}
#endif 

		int toDo = range.y - range.x;

#ifdef DEBUG_INTEGRATE
		if (block.thread_rank() == 0 && p_rounds > 1) {
			printf("TILE %d: Round %d/%d, points left: %d\n", TILE_IDX, p_round, p_rounds, p_toDo);
		}
#endif

		uint32_t contributor = 0;
		bool active = thread_idx < p_toDo;
		bool done = !active;

		// Iterate over batches until all done or range is complete
		for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
		{
			// End if entire block votes that it is done rasterizing
			int num_done = __syncthreads_count(done);
			if (num_done == BLOCK_SIZE)
				break;

			// Collectively fetch per-Gaussian data from global to shared
			int progress = i * BLOCK_SIZE + block.thread_rank();
			if (range.x + progress < range.y)
			{
				int coll_id = gaussian_list[range.x + progress];
				collected_id[block.thread_rank()] = coll_id;
				collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
				for (int ii = 0; ii < 10; ii++)
					collected_view2gaussian[10 * block.thread_rank() + ii] = view2gaussian[coll_id * 10 + ii];
				if constexpr (MIN_Z_BOUNDING)
					collected_gaussiandepth[block.thread_rank()] = __uint_as_float(gaussian_depths[range.x + progress]);
				if constexpr (RETURN_COLOR) {
					for (int ii = 0; ii < CHANNELS; ii++)
						collected_color[CHANNELS * thread_idx + ii] = features[CHANNELS * coll_id + ii];
				}
			}
			block.sync();

			// Iterate over current batch
			for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
			{
				// Keep track of current position in range
				contributor++;

#ifdef DEBUG_MIN_Z_BOUNDING
				if (thread_idx == 0 && BLOCK_IDX == 0) {
					printf("G %d/%d: \t%.4f\n", collected_id[j], contributor, collected_gaussiandepth[j]);
				}
#endif
				float4 con_o = collected_conic_opacity[j];
				float* view2gaussian_j = collected_view2gaussian + j * 10;

				if constexpr (MIN_Z_BOUNDING) {
					float gaussian_depth = collected_gaussiandepth[j];
					if (gaussian_depth > current_point_depth) {
						done = true;
						continue;
					}
				}

				const float normal[3] = { 
					view2gaussian_j[0] * ray_point.x + view2gaussian_j[1] * ray_point.y + view2gaussian_j[2], 
					view2gaussian_j[1] * ray_point.x + view2gaussian_j[3] * ray_point.y + view2gaussian_j[4],
					view2gaussian_j[2] * ray_point.x + view2gaussian_j[4] * ray_point.y + view2gaussian_j[5]
				};

				// use AA, BB, CC so that the name is unique
				double AA = ray_point.x * normal[0] + ray_point.y * normal[1] + normal[2];
				double BB = 2 * (view2gaussian_j[6] * ray_point.x + view2gaussian_j[7] * ray_point.y + view2gaussian_j[8]);
				float CC = view2gaussian_j[9];
				
				// t is the depth of the gaussian
				float t = -BB/(2*AA);
				// depth must be positive otherwise it is not valid and we skip it
				if (t > current_point_depth) {
					t = current_point_depth;
				}
				
				double min_value = (AA * t * t + BB * t + CC);

				float power = -0.5f * min_value;
				if (power > 0.0f){
					power = 0.0f;
				}

				float alpha_point = min(0.99f, con_o.w * exp(power));

#ifdef DEBUG_INTEGRATE
				if (point_id == POINT_TO_DEBUG) {
					printf("Gaussian %d: alpha %f/%f, T %f, depth %f\n", collected_id[j], alpha, alpha_point, T, t);
				}
#endif 

				if (alpha_point < 1.0f / 255.0f) {
					continue;
				}

				alpha += alpha_point * T;

				if constexpr (RETURN_COLOR) {
					float test_T = T * (1 - alpha_point);
					if (test_T >= 0.0001f) {
						for (int ch = 0; ch < CHANNELS; ch++) {
							C[ch] += collected_color[j * CHANNELS + ch] * alpha_point * T;
						}
					}
				}

				// hmm, isnt this already opa
				T *= (1 - alpha_point);

				if constexpr (ALPHA_EARLY_STOP && !RETURN_COLOR) {
					if (alpha > 0.5000001f) {
						done = true;
					}
				}

			}
		}
		if (active) {
			out_alpha_integrated[point_id] = fminf(alpha, out_alpha_integrated[point_id]);
#ifdef DEBUG_OPACITY_FIELD
			// filter out the really wrong one
			if (alpha < 0.5f) {
				printf("[%.2f, %.2f]: depth %.4f, dir %.3f/%.3f/%.3f\n", points2D[point_id].x, points2D[point_id].y, depths[point_id],
					(current_point.x - W/2.f) / focal_x, (current_point.y -H/2.f) / focal_y, 1.0f);
			}
#endif
			if constexpr (RETURN_COLOR) {
				for (int ch = 0; ch < CHANNELS; ch++)
					out_color_integrated[point_id * CHANNELS + ch] = C[ch] + T * bg_color[ch];
			}
		}
	}
}

void FORWARD::integrate(
	const dim3 grid, dim3 block,
	const uint2* gaussian_ranges,
	const uint2* point_ranges,
	const uint2* tile_tile_mapping,
	const uint32_t* gaussian_list,
	const uint32_t* point_list,
	const uint64_t* gaussian_depths,
	int W, int H, int PN,
	float focal_x, float focal_y,
	const float2* points2D,
	const float* colors,
	const float* view2gaussian,
	const float* cov3Ds,
	const float4* cov3D_inv,
	const float* viewmatrix,
	const float* projmatrix_inv,
	const float3* means3D,
	const float3* scales,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	// float* center_depth,
	// float4* center_alphas,
	const float* bg_color,
	const glm::vec3* cam_pos,
	const float2* means2D,
	float* out_color,
	float* out_alpha_integrated,
	float* out_color_integrated,
	DebugVisualizationData& debugVisualization,
	const SplattingSettings splatting_settings
	)
{
#define INTEGRATE_CALL(ALPHA_EARLY_STOP, MIN_Z_BOUND, RETURN_COLOR) integrateCUDA<NUM_CHANNELS, ALPHA_EARLY_STOP, MIN_Z_BOUND, RETURN_COLOR> << <grid, block >> > ( \
		gaussian_ranges,point_ranges,tile_tile_mapping,gaussian_list,point_list,gaussian_depths,W, H,focal_x, focal_y,points2D,colors,view2gaussian,cov3Ds,viewmatrix,means3D,scales,depths,conic_opacity,final_T,n_contrib,bg_color,out_color,out_alpha_integrated,out_color_integrated);

#define INTEGRATE_RETURN_COLOR(ALPHA_EARLY_STOP, MIN_Z_BOUND) if (splatting_settings.meshing_settings.return_color) INTEGRATE_CALL(ALPHA_EARLY_STOP, MIN_Z_BOUND, true) else INTEGRATE_CALL(ALPHA_EARLY_STOP, MIN_Z_BOUND, false)
#define INTEGRATE_Z_BOUND(ALPHA_EARLY_STOP) if (splatting_settings.sort_settings.sort_order == GlobalSortOrder::MIN_VIEWSPACE_Z) INTEGRATE_RETURN_COLOR(ALPHA_EARLY_STOP, true) else INTEGRATE_RETURN_COLOR(ALPHA_EARLY_STOP, false)

	if (splatting_settings.meshing_settings.alpha_early_stop) {
		INTEGRATE_Z_BOUND(true);
	}
	else {
		INTEGRATE_Z_BOUND(false);
	}

#undef INTEGRATE_Z_BOUND
#undef INTEGRATE_RETURN_COLOR
#undef INTEGRATE_CALL
}

template<int C>
__global__ void preprocessPointsCUDA(int PN, int D, int M,
	const float* points3D,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	float2* points2D,
	float* opacity_field,
	float* depths,
	const dim3 grid,
	uint32_t* tiles_touched,
	const float* alpha_integrated,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= PN)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	tiles_touched[idx] = 0;
	depths[idx] = -1.f;

#ifdef OPT_CULL_POINTS
	if (alpha_integrated[idx] < 0.4999f) {
		return;
	}
#endif

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum_GOF(idx, points3D, viewmatrix, projmatrix, prefiltered, p_view))
	{
		return;
	}

	float2 point_image = {focal_x * p_view.x / (p_view.z + 0.0000001f) + W/2., focal_y * p_view.y / (p_view.z + 0.0000001f) + H/2.};

	// If the point is outside the image, quit.
	if (point_image.x < 0 || point_image.x >= W || point_image.y < 0 || point_image.y >= H)
	{
		return;
	}
	//printf("Point is inside image\n");
	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	points2D[idx] = point_image;
	tiles_touched[idx] = 1;
}

void FORWARD::PreprocessPoints(int PN, int D, int M,
		const float* points3D,
		const float* viewmatrix,
		const float* projmatrix,
		const glm::vec3* cam_pos,
		const int W, int H,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		float2* points2D,
		float* opacity_field,
		float* depths,
		const dim3 grid,
		uint32_t* tiles_touched,
		float* alpha,
		bool prefiltered)
{

	preprocessPointsCUDA<NUM_CHANNELS> << <(PN + 255) / 256, 256 >> > (
		PN, D, M,
		points3D,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		points2D,
		opacity_field,
		depths,
		grid,
		tiles_touched,
		alpha,
		prefiltered
		);
	//for(int i = 0; i < 100; i++)
	//{
	//	printf("2D Point: %f %f\n", points2D[i].x, points2D[i].y);
	//}
}

void FORWARD::computeTransmittance(
	const dim3 grid, dim3 block,
	const uint2* gaussian_ranges,
	const uint2* point_ranges,
	const uint2* tile_tile_mapping,
	const uint32_t* gaussian_list,
	const uint32_t* point_list,
	const uint64_t* gaussian_depths,
	int W, int H, int PN,
	float focal_x, float focal_y,
	const float2* points2D,
	const float* view2gaussian,
	const float* cov3Ds,
	const float4* cov3D_inv,
	const float* viewmatrix,
	const float* projmatrix_inv,
	const float3* means3D,
	const float3* scales,
	const float* depths,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const glm::vec3* cam_pos,
	const float2* means2D,
	float* out_transmittance,
	float transmittance_threshold,
	DebugVisualizationData& debugVisualization,
	const SplattingSettings splatting_settings)
{
#define TRANSMITTANCE_CALL(ALPHA_EARLY_STOP, MIN_Z_BOUND) computeTransmittanceCUDA<NUM_CHANNELS, ALPHA_EARLY_STOP, MIN_Z_BOUND> << <grid, block >> > ( \
		gaussian_ranges,point_ranges,tile_tile_mapping,gaussian_list,point_list,gaussian_depths,W, H,focal_x, focal_y,points2D,view2gaussian,cov3Ds,viewmatrix,means3D,scales,depths,conic_opacity,final_T,n_contrib,out_transmittance, transmittance_threshold);

#define TRANSMITTANCE_Z_BOUND(ALPHA_EARLY_STOP) if (splatting_settings.sort_settings.sort_order == GlobalSortOrder::MIN_VIEWSPACE_Z) TRANSMITTANCE_CALL(ALPHA_EARLY_STOP, true) else TRANSMITTANCE_CALL(ALPHA_EARLY_STOP, false)

	if (splatting_settings.meshing_settings.alpha_early_stop) {
		TRANSMITTANCE_Z_BOUND(true);
	}
	else {
		TRANSMITTANCE_Z_BOUND(false);
	}

#undef TRANSMITTANCE_Z_BOUND
#undef TRANSMITTANCE_CALL
}
